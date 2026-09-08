"""Anthropic tool-calling agent loop with pin injection, permissions, and compaction."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from contextlib import suppress
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from orbweaver import __version__
from orbweaver.compact import (
    events_to_messages,
    live_events,
    maybe_compact,
    persist_tool_result,
    prompt_events,
    record_usage,
    rehydrate_messages,
    usage_input_tokens,
)
from orbweaver.config import settings
from orbweaver.image import format_image_read, hydrate_workspace_images, is_image_path
from orbweaver.memory import pinned_prompt, remember, rewrite_search_query
from orbweaver.permissions import TurnAborted, can_use_tool, denial_state_for
from orbweaver.permissions.injection_probe import probe_tool_output
from orbweaver.skills import workspace_skills_prompt
from orbweaver.store import Event, Job, Store, new_uuid
from orbweaver.lints import read_lints
from orbweaver.todos import inject_session_todos, persist_todos
from orbweaver.tooltext import format_read, format_webfetch

DEFAULT_MAX_ROUNDS = 48
LAST_ROUND_NUDGE = (
    "This is the last tool round of this turn. After these tool results, answer the user "
    "with what you have. Do not start new exploration. Spawn a subagent only if a narrower "
    "task can finish the work."
)
CONCLUDE_NUDGE = (
    "Tool-round budget exhausted. Answer the user now from the tool results you already "
    "have. Do not call tools."
)

TOOL_SPEC = [
    {
        "name": "Read",
        "description": (
            "Read a file as numbered lines, or an image as vision (jpg/png/webp/gif). "
            "Relative paths are the session workspace. Absolute paths in extra sandbox "
            "roots are auto-allowed; other host paths are classified. Use offset "
            "(1-based line, or negative from the end) and limit to page text; do not "
            "page files with Bash. The result says how to continue when truncated."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "offset": {
                    "type": "integer",
                    "description": "1-based starting line; negative counts from the end.",
                },
                "limit": {"type": "integer", "description": "Max lines to return (default 400)."},
            },
            "required": ["path"],
        },
    },
    {
        "name": "Write",
        "description": "Write a file in the session workspace.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"],
        },
    },
    {
        "name": "Delete",
        "description": (
            "Delete a file or directory in the session working set. Same deny/ask rules as "
            "Write: always-deny secrets (.env, keys), and paths outside the working set are "
            "blocked. Prefer this over Bash rm."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
    {
        "name": "ProposePatch",
        "description": "Propose a search-replace patch (also applied as a review overlay in VS Code).",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "old_string": {"type": "string"},
                "new_string": {"type": "string"},
            },
            "required": ["path", "new_string"],
        },
    },
    {
        "name": "Glob",
        "description": "List files matching a glob in the workspace.",
        "input_schema": {
            "type": "object",
            "properties": {"pattern": {"type": "string"}},
            "required": ["pattern"],
        },
    },
    {
        "name": "Grep",
        "description": (
            "Search file contents with a regex. Use | for alternation. glob limits the files "
            "(default **/*)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"pattern": {"type": "string"}, "glob": {"type": "string"}},
            "required": ["pattern"],
        },
    },
    {
        "name": "ReadLints",
        "description": (
            "Return diagnostics for files you just edited. Runs ORBWEAVER_LINTER when set "
            "(use {paths} or trailing paths). Otherwise a Python AST stub reports syntax "
            "errors. Paths default to recent Write/ProposePatch targets."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "paths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Workspace paths to check. Empty uses recently edited files.",
                }
            },
        },
    },
    {
        "name": "Bash",
        "description": (
            "Run a shell command in the workspace. Sandboxed by default: host files are readable, "
            "writes stay in the working set, network uses a domain allowlist, Unix sockets are "
            "denied unless granted. Host reads and allowlisted sockets/domains do not need "
            "escalation. If the sandbox blocks the command, ask the user before setting "
            "permissions to [\"full_network\"] (arbitrary internet) or [\"all\"] "
            "(host writes/docker/sudo). Those overrides pause for approval. "
            "unsandboxed true aliases [\"all\"]."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {"type": "string"},
                "unsandboxed": {"type": "boolean"},
                "permissions": {
                    "type": "array",
                    "items": {"type": "string", "enum": ["full_network", "all"]},
                },
            },
            "required": ["command"],
        },
    },
    {
        "name": "WebFetch",
        "description": (
            "HTTP GET a known URL and return extracted readable text (HTML is stripped). "
            "Find URLs with WebSearch first. Do not keep fetching nearby docs URLs when the "
            "result says the page is JavaScript-rendered."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"url": {"type": "string"}},
            "required": ["url"],
        },
    },
    {
        "name": "WebSearch",
        "description": (
            "Search the public web and return titled result URLs with snippets. Use this "
            "instead of guessing documentation URLs. Then WebFetch one or two promising links. "
            "Call independent searches in parallel in one round."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "max_results": {
                    "type": "integer",
                    "description": "How many hits to return (default 8, max 10).",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "MemorySearch",
        "description": "Search shared memory using the conversation plus current query.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
    {
        "name": "MemoryRemember",
        "description": "Store a fact in shared memory.",
        "input_schema": {
            "type": "object",
            "properties": {"text": {"type": "string"}, "pinned": {"type": "boolean"}},
            "required": ["text"],
        },
    },
    {
        "name": "MemoryPin",
        "description": "Pin a chunk or entity into always-injected working memory.",
        "input_schema": {
            "type": "object",
            "properties": {
                "kind": {"type": "string", "enum": ["chunk", "entity"]},
                "id": {"type": "string"},
            },
            "required": ["kind", "id"],
        },
    },
    {
        "name": "MemoryForget",
        "description": "Tombstone a memory chunk.",
        "input_schema": {
            "type": "object",
            "properties": {"id": {"type": "string"}},
            "required": ["id"],
        },
    },
    {
        "name": "AskUser",
        "description": "Ask the user a question and wait.",
        "input_schema": {
            "type": "object",
            "properties": {"question": {"type": "string"}},
            "required": ["question"],
        },
    },
    {
        "name": "TodoWrite",
        "description": (
            "Create or update the session todo list. Persisted on the session so compaction "
            "does not erase the plan. merge true updates by id; merge false replaces the list."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "todos": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "content": {"type": "string"},
                            "status": {
                                "type": "string",
                                "enum": ["pending", "in_progress", "completed", "cancelled"],
                            },
                        },
                        "required": ["content"],
                    },
                },
                "merge": {
                    "type": "boolean",
                    "description": "If true, merge by id into the existing list.",
                },
            },
            "required": ["todos"],
        },
    },
    {
        "name": "SpawnSubagent",
        "description": (
            "Spawn a nested agent with its own event stream to complete a focused task. "
            "Shares this workspace and memory. Returns a summary. Nested spawns are not allowed."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "task": {"type": "string"},
                "label": {"type": "string", "description": "Short name for the child run."},
            },
            "required": ["task"],
        },
    },
    {
        "name": "ScheduleTask",
        "description": (
            "Ask the host to schedule a job. due_at is ISO-8601. "
            "Optional recurrence: minute, hour, or day (also 'every hour'), "
            "or a 5-field cron expression (minute hour day-of-month month "
            "day-of-week), e.g. '0 9 * * mon'. Omit recurrence for a one-shot."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "due_at": {"type": "string"},
                "message": {"type": "string"},
                "recurrence": {
                    "type": "string",
                    "description": (
                        "minute/hour/day, or 5-field cron "
                        "(minute hour day month weekday)."
                    ),
                },
            },
            "required": ["due_at", "message"],
        },
    },
    {
        "name": "SendPhoto",
        "description": (
            "Send an image file from the workspace to the user on Telegram. "
            "Use this for screenshots, charts, or other images you wrote under attachments/."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "caption": {"type": "string"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "GenerateImage",
        "description": (
            "Generate an image from a text prompt and save it under attachments/. "
            "On Telegram sessions the file is also sent to the chat."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "prompt": {"type": "string"},
                "path": {
                    "type": "string",
                    "description": "Workspace path to write, default attachments/generated-<id>.png",
                },
                "caption": {"type": "string"},
            },
            "required": ["prompt"],
        },
    },
]

PROPOSE_PATCH_TOOL = "ProposePatch"
VSCODE_CHANNEL = "vscode"


def normalize_channel(value: str | None) -> str:
    raw = str(value or "").strip().lower().replace("_", "-").replace(" ", "")
    if raw in {"vscode", "vs-code", "visualstudiocode"}:
        return VSCODE_CHANNEL
    return raw


def resolve_channel(
    jsonld: dict[str, Any] | None = None,
    *,
    channel: str | None = None,
    extra: dict[str, Any] | None = None,
) -> str:
    """Client channel for tool/prompt filtering. Only vscode keeps ProposePatch."""
    if channel and str(channel).strip():
        return normalize_channel(channel)
    extra = extra or {}
    extra_ch = extra.get("channel") or extra.get("client")
    if extra_ch and str(extra_ch).strip():
        return normalize_channel(str(extra_ch))
    jsonld = jsonld or {}
    stored = jsonld.get("channel") or jsonld.get("client")
    if stored and str(stored).strip():
        return normalize_channel(str(stored))
    if jsonld.get("telegram_user_id") is not None or jsonld.get("telegram_chat_id") is not None:
        return "telegram"
    title = str(jsonld.get("title") or "").strip().lower()
    if title == "vscode" or title.startswith("vscode:") or title.startswith("vscode/"):
        return VSCODE_CHANNEL
    return ""


def channel_allows_proposepatch(channel: str) -> bool:
    return normalize_channel(channel) == VSCODE_CHANNEL


def tools_for_channel(
    channel: str,
    tool_spec: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    spec = list(tool_spec if tool_spec is not None else TOOL_SPEC)
    if channel_allows_proposepatch(channel):
        return spec
    return [t for t in spec if t.get("name") != PROPOSE_PATCH_TOOL]


async def session_tools(workspace=None, channel: str | None = None) -> list[dict[str, Any]]:
    """Built-in tools (channel-filtered) plus tools from configured MCP servers."""
    from orbweaver.mcp import mcp_tool_specs

    extra = await mcp_tool_specs(workspace)
    base = list(TOOL_SPEC) if channel is None else tools_for_channel(channel)
    return base + extra


def _events_to_messages(events: list[Event]) -> list[dict[str, Any]]:
    """Build Anthropic messages from a (possibly projected) event list."""
    return events_to_messages(events)


def _nudge_user(messages: list[dict[str, Any]], text: str) -> None:
    if messages and messages[-1].get("role") == "user":
        prev = messages[-1]["content"]
        if isinstance(prev, str):
            messages[-1]["content"] = prev + "\n\n" + text
            return
        if isinstance(prev, list):
            prev.append({"type": "text", "text": text})
            return
    messages.append({"role": "user", "content": text})


def _prompt_messages(events: list[Event], workspace, user_text: str) -> list[dict[str, Any]]:
    messages = events_to_messages(prompt_events(events))
    messages = hydrate_workspace_images(messages, workspace)
    messages = rehydrate_messages(messages, events, workspace)
    messages = inject_session_todos(messages, events)
    if not messages:
        messages = [{"role": "user", "content": user_text}]
    return messages


def static_system(channel: str = "") -> str:
    write_line = "In auto mode, in-project Write applies immediately. "
    if channel_allows_proposepatch(channel):
        write_line += "Prefer ProposePatch when a visible diff overlay helps the user. "
    return (
        f"You are Orbweaver, a coding agent. The running gateway is Orbweaver {__version__} "
        f"(semantic version). If asked what version is running, answer {__version__}. "
        "Use tools to read and patch the workspace. Sandboxed Bash can read host files; "
        "do not set permissions [\"all\"] just to inspect logs or journals. "
        f"{write_line}"
        "In auto mode, in-project Delete applies immediately. TodoWrite keeps the plan "
        "on this session across compaction. After edits, ReadLints for diagnostics. "
        "Use MemorySearch when past decisions might "
        "matter. Keep pins small. If the sandbox cannot run a command, ask the user "
        "before requesting permissions [\"full_network\"] or [\"all\"]. Hard denials "
        "stay blocked; do not route around them. Call independent tools in parallel in "
        "one round. Prefer Read offset/limit and Grep over Bash for paging files. "
        "Use WebSearch to find sources, then WebFetch a few result URLs; do not guess "
        "docs paths. Configured MCP servers appear as mcp_<server>_<tool> and use the "
        "same permission pipeline as other tools. Finish with a user-visible answer "
        "before the tool-round budget runs out; spawn a subagent for a long exploration "
        "instead of burning parent rounds."
    )


def build_agent_system(
    pins: str, extra: str = "", skills: str = "", channel: str = ""
) -> list[dict[str, Any]]:
    rest = pins or "(no pinned memory)"
    if skills:
        rest = rest + "\n\n" + skills
    if extra:
        rest = rest + "\n\n" + extra
    return [
        {"type": "text", "text": static_system(channel), "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": rest},
    ]


def _blocked_tool_result(decision) -> str:
    if decision.behavior == "ask":
        return (
            "This action needs user approval and was not executed. "
            f"Reason: {decision.reason}. Wait for the user to confirm, then retry."
        )
    return (
        f"Blocked by permission gate ({decision.fast_path}): {decision.reason}. "
        "Treat this boundary in good faith. Find a safer path; do not route around the block."
    )


async def run_tools(name: str, inp: dict[str, Any], ctx: dict[str, Any]) -> str:
    ws = ctx["workspace"]
    store: Store = ctx["store"]
    session_id: UUID = ctx["session_id"]
    if name == "Read":
        path = str(inp.get("path") or "")
        if is_image_path(path):
            return format_image_read(ws, path)
        try:
            raw = ws.read(path)
        except (OSError, PermissionError, UnicodeDecodeError, IsADirectoryError) as e:
            return f"error reading {inp.get('path')}: {e}"
        return format_read(raw, path=path, offset=inp.get("offset"), limit=inp.get("limit"))
    if name == "Write":
        ws.write(inp["path"], inp["content"])
        return f"wrote {inp['path']}"
    if name == "Delete":
        try:
            return ws.delete(inp["path"])
        except (OSError, PermissionError) as e:
            return f"error deleting {inp.get('path')}: {e}"
    if name == "ProposePatch":
        result = ws.propose_patch(inp["path"], inp.get("old_string") or "", inp["new_string"])
        return json.dumps(result)[:200_000]
    if name == "Glob":
        return "\n".join(ws.glob(inp["pattern"])[:200])
    if name == "Grep":
        return "\n".join(ws.grep(inp["pattern"], inp.get("glob") or "**/*"))
    if name == "ReadLints":
        return read_lints(ws, inp, ctx.get("events"))
    if name == "Bash":
        from orbweaver.permissions.pipeline import bash_permissions

        perms = sorted(bash_permissions(inp))
        return ws.bash(
            inp["command"],
            unsandboxed=bool(inp.get("unsandboxed")),
            permissions=perms,
        )
    if name == "WebFetch":
        import httpx

        r = httpx.get(inp["url"], timeout=20.0, follow_redirects=True)  # noqa: ASYNC210
        ctype = r.headers.get("content-type") or ""
        return format_webfetch(str(inp.get("url") or ""), r.status_code, ctype, r.text)
    if name == "WebSearch":
        from orbweaver.websearch import run_websearch

        return run_websearch(inp)
    if name == "MemorySearch":
        events = live_events(await store.list_events(session_id))
        already = set()
        for ev in events:
            if ev.kind == "MemoryRecall":
                already.update(ev.payload.get("chunk_ids") or [])
        q = rewrite_search_query(events, inp.get("query") or "")
        hits = await store.search_chunks(q, k=8)
        lines = []
        ids = []
        for c, score in hits:
            if str(c.id) in already:
                continue
            ids.append(str(c.id))
            lines.append(f"[{c.id} score={score:.3f}] {c.text}")
        return json.dumps({"chunk_ids": ids, "text": "\n".join(lines) or "(no hits)"})
    if name == "MemoryRemember":
        from orbweaver.store import PinBudgetError

        try:
            chunk = await remember(
                store, inp["text"], source="agent", pinned=bool(inp.get("pinned"))
            )
        except PinBudgetError as e:
            return f"pin rejected: {e}"
        return str(chunk.id)
    if name == "MemoryPin":
        from orbweaver.store import PinBudgetError

        try:
            await store.set_pinned(inp["kind"], UUID(inp["id"]), True)
        except PinBudgetError as e:
            return f"pin rejected: {e}"
        except (KeyError, ValueError) as e:
            return f"error: {e}"
        return "pinned"
    if name == "MemoryForget":
        await store.forget_chunk(UUID(inp["id"]))
        return "forgotten"
    if name == "AskUser":
        return json.dumps({"ask": inp["question"]})
    if name == "TodoWrite":
        todos = await persist_todos(store, session_id, inp)
        return json.dumps({"todos": todos})
    if name == "SpawnSubagent":
        from orbweaver.subagent import run_subagent

        return await run_subagent(inp, ctx)
    if name == "ScheduleTask":
        try:
            due = datetime.fromisoformat(str(inp["due_at"]).replace("Z", "+00:00"))  # noqa: FURB162
            if due.tzinfo is None:
                due = due.replace(tzinfo=UTC)
        except ValueError as e:
            return f"invalid due_at: {e}"
        rec = inp.get("recurrence")
        if rec:
            from orbweaver.channels.cron import _parse_recurrence

            if _parse_recurrence(str(rec), due) is None:
                return (
                    "invalid recurrence: use minute, hour, day "
                    "(or 'every hour'), or a 5-field cron expression "
                    "like '0 9 * * mon'"
                )
        job = Job(
            id=new_uuid(),
            due_at=due,
            payload={"message": inp.get("message") or ""},
            recurrence=rec,
            session_id=session_id,
        )
        await store.put_job(job)
        return json.dumps({"job_id": str(job.id), "due_at": due.isoformat()})
    if name == "SendPhoto":
        from orbweaver.channels.telegram import send_session_photo

        return await send_session_photo(ctx, inp)
    if name == "GenerateImage":
        from orbweaver.channels.telegram import generate_and_maybe_send

        return await generate_and_maybe_send(ctx, inp)
    if name.startswith("mcp_"):
        from orbweaver.mcp import call_mcp_tool

        return await call_mcp_tool(name, inp, ws)
    return f"unknown tool {name}"


class TurnCancelled(Exception):
    """Raised when the user stops or discards an in-flight turn."""

    def __init__(self, produced: list[Event] | None = None):
        super().__init__("turn cancelled")
        self.produced = produced or []


class TurnInjected(Exception):
    """Current LLM call aborted so a mid-turn follow-up can join this query."""


def _raise_if_cancelled(
    cancel: asyncio.Event | None, produced: list[Event] | None = None
) -> None:
    if cancel is not None and cancel.is_set():
        raise TurnCancelled(produced)


async def _await_or_cancel(
    coro,
    cancel: asyncio.Event | None,
    produced: list[Event] | None = None,
    inject: asyncio.Event | None = None,
):
    if cancel is None and inject is None:
        return await coro
    _raise_if_cancelled(cancel, produced)
    task = asyncio.create_task(coro)
    watchers: set[asyncio.Task] = set()
    if cancel is not None:
        watchers.add(asyncio.create_task(cancel.wait()))
    if inject is not None:
        watchers.add(asyncio.create_task(inject.wait()))
    if not watchers:
        return await task
    try:
        done, _pending = await asyncio.wait(
            {task, *watchers}, return_when=asyncio.FIRST_COMPLETED
        )
        if task not in done:
            task.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await task
            if cancel is not None and cancel.is_set():
                raise TurnCancelled(produced)
            raise TurnInjected()
        for w in watchers:
            w.cancel()
            with suppress(asyncio.CancelledError):
                await w
        return task.result()
    except asyncio.CancelledError:
        task.cancel()
        for w in watchers:
            w.cancel()
        with suppress(asyncio.CancelledError, Exception):
            await task
        raise


async def agent_turn(
    store: Store,
    session_id: UUID,
    user_text: str,
    workspace,
    workspace_kind: str = "local",
    emit: Callable[[dict[str, Any]], None] | None = None,
    cancel: asyncio.Event | None = None,
    turn_state: Any | None = None,
    resume: bool = False,
    headless: bool = False,
    tools: list[dict[str, Any]] | None = None,
    system_extra: str = "",
    max_rounds: int = DEFAULT_MAX_ROUNDS,
    subagent_depth: int = 0,
    images: list[dict[str, str]] | None = None,
    channel: str | None = None,
) -> list[Event]:
    _raise_if_cancelled(cancel)
    if resume:
        if turn_state is not None:
            for ev in reversed(await store.list_events(session_id)):
                if ev.kind == "user":
                    turn_state.user_seq = ev.seq
                    if not user_text:
                        user_text = str(ev.payload.get("text") or "")
                    break
    else:
        payload: dict[str, Any] = {"text": user_text}
        if images:
            payload["images"] = images
        user_ev = await store.append_event(session_id, "user", payload)
        if turn_state is not None:
            turn_state.user_seq = user_ev.seq
    produced: list[Event] = []

    def fire(ev: Event) -> None:
        produced.append(ev)
        if emit:
            emit({"kind": ev.kind, "payload": ev.payload, "id": str(ev.id), "seq": ev.seq})

    def check() -> None:
        _raise_if_cancelled(cancel, produced)

    def inject_event() -> asyncio.Event | None:
        return getattr(turn_state, "inject", None) if turn_state is not None else None

    async def record_abort(exc: TurnAborted) -> None:
        payload = dict(exc.payload)
        payload.setdefault("text", exc.message)
        fire(await store.append_event(session_id, "turn_aborted", payload))
        fire(await store.append_event(session_id, "assistant", {"text": exc.message}))

    check()
    if not settings.anthropic_api_key:
        ev = await store.append_event(
            session_id,
            "assistant",
            {
                "text": (
                    "ANTHROPIC_API_KEY is not set. Echo: "
                    + (user_text or "")[:500]
                    + "\nSet the key to enable the Claude tool loop."
                )
            },
        )
        fire(ev)
        return produced

    import anthropic

    headers = {}
    if settings.anthropic_workspace_id.strip():
        headers["anthropic-workspace-id"] = settings.anthropic_workspace_id.strip()
    client = anthropic.AsyncAnthropic(
        api_key=settings.anthropic_api_key, default_headers=headers or None
    )
    sess = await store.get_entity(session_id)
    resolved_channel = resolve_channel(sess.jsonld if sess else None, channel=channel)
    denial_state = denial_state_for(session_id)
    ctx = {
        "workspace": workspace,
        "store": store,
        "session_id": session_id,
        "workspace_kind": workspace_kind,
        "auto_write": True,
        "headless": headless,
        "denial_state": denial_state,
        "events": [],
        "cancel": cancel,
        "fire": fire,
        "subagent_depth": subagent_depth,
        "channel": resolved_channel,
    }
    pins = await pinned_prompt(store)
    system = build_agent_system(
        pins,
        system_extra,
        skills=workspace_skills_prompt(workspace),
        channel=resolved_channel,
    )
    active_tools = (
        tools if tools is not None else await session_tools(workspace, resolved_channel)
    )
    if not resume:
        await maybe_compact(
            store, session_id, client=client, workspace=workspace, system=system
        )
        if turn_state is not None:
            for ev in reversed(await store.list_events(session_id)):
                if ev.kind == "user":
                    turn_state.user_seq = ev.seq
                    break

    try:
        for round_i in range(max_rounds):
            check()
            inj = inject_event()
            if inj is not None:
                inj.clear()
            events = await store.list_events(session_id)
            ctx["events"] = events
            messages = _prompt_messages(events, workspace, user_text)
            if round_i == max_rounds - 1:
                _nudge_user(messages, LAST_ROUND_NUDGE)
            try:
                resp = await _await_or_cancel(
                    client.messages.create(
                        model=settings.orbweaver_model,
                        max_tokens=4096,
                        system=system,
                        tools=active_tools,
                        messages=messages,
                    ),
                    cancel,
                    produced,
                    inject=inj,
                )
            except TurnInjected:
                continue
            last_seq = events[-1].seq if events else 0
            record_usage(session_id, usage_input_tokens(getattr(resp, "usage", None)), last_seq)
            tool_uses = [b for b in resp.content if getattr(b, "type", None) == "tool_use"]
            texts = [b.text for b in resp.content if getattr(b, "type", None) == "text"]
            if texts:
                ev = await store.append_event(session_id, "assistant", {"text": "\n".join(texts)})
                fire(ev)
            check()
            if not tool_uses:
                break
            stop_after_ask = False
            for block in tool_uses:
                check()
                call_ev = await store.append_event(
                    session_id,
                    "tool_call",
                    {"id": block.id, "name": block.name, "input": block.input},
                )
                fire(call_ev)
                ctx["events"] = await store.list_events(session_id)
                try:
                    decision = await can_use_tool(block.name, dict(block.input), ctx)
                except TurnAborted as e:
                    await record_abort(e)
                    return produced
                fire(
                    await store.append_event(
                        session_id,
                        "permission_decision",
                        {
                            "tool_use_id": block.id,
                            "name": block.name,
                            "behavior": decision.behavior,
                            "reason": decision.reason,
                            "fast_path": decision.fast_path,
                        },
                    )
                )
                persisted_path = None
                if decision.behavior != "allow":
                    result = _blocked_tool_result(decision)
                    if decision.behavior == "ask":
                        stop_after_ask = True
                else:
                    result = await run_tools(block.name, dict(block.input), ctx)
                    result, persisted_path = persist_tool_result(
                        workspace, str(block.id), block.name, result
                    )
                    probed = await probe_tool_output(block.name, result)
                    result = probed["output"]
                    if probed.get("flagged"):
                        fire(
                            await store.append_event(
                                session_id,
                                "injection_warning",
                                {"tool_use_id": block.id, "name": block.name},
                            )
                        )
                check()
                kind = "MemoryRecall" if block.name == "MemorySearch" else "tool_result"
                payload = {"tool_use_id": block.id, "name": block.name, "content": result}
                if decision.behavior == "allow" and persisted_path:
                    payload["persisted_path"] = persisted_path
                if kind == "MemoryRecall":
                    try:
                        parsed = json.loads(result)
                        payload["chunk_ids"] = parsed.get("chunk_ids") or []
                        payload["text"] = parsed.get("text") or result
                    except json.JSONDecodeError:
                        payload["text"] = result
                res_ev = await store.append_event(session_id, kind, payload)
                fire(res_ev)
                if block.name == "ProposePatch" and decision.behavior == "allow":
                    fire(
                        await store.append_event(
                            session_id, "patch_proposal", {"tool_use_id": block.id, "content": result}
                        )
                    )
                if block.name == "ScheduleTask" and decision.behavior == "allow":
                    fire(
                        await store.append_event(
                            session_id, "schedule_request", {"input": dict(block.input)}
                        )
                    )
            if stop_after_ask:
                fire(
                    await store.append_event(
                        session_id,
                        "assistant",
                        {
                            "text": (
                                "I need your approval before continuing. "
                                "Reply with what you want done."
                            )
                        },
                    )
                )
                break
            await maybe_compact(
                store, session_id, client=client, workspace=workspace, system=system
            )
        else:
            events = await store.list_events(session_id)
            messages = _prompt_messages(events, workspace, user_text)
            _nudge_user(messages, CONCLUDE_NUDGE)
            try:
                resp = await _await_or_cancel(
                    client.messages.create(
                        model=settings.orbweaver_model,
                        max_tokens=4096,
                        system=system,
                        tools=[],
                        messages=messages,
                    ),
                    cancel,
                    produced,
                    inject=inject_event(),
                )
            except TurnInjected:
                return produced
            last_seq = events[-1].seq if events else 0
            record_usage(session_id, usage_input_tokens(getattr(resp, "usage", None)), last_seq)
            texts = [b.text for b in resp.content if getattr(b, "type", None) == "text"]
            if texts:
                fire(
                    await store.append_event(session_id, "assistant", {"text": "\n".join(texts)})
                )
            else:
                fire(
                    await store.append_event(
                        session_id,
                        "assistant",
                        {
                            "text": (
                                f"Stopped after {max_rounds} tool rounds without a final answer. "
                                "Continue in a follow-up, or spawn a subagent with a narrower task."
                            )
                        },
                    )
                )
        return produced
    except TurnAborted as e:
        await record_abort(e)
        return produced
    except TurnCancelled as e:
        e.produced = produced
        raise
