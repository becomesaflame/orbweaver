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
from orbweaver.memory import pinned_prompt, remember, rewrite_search_query
from orbweaver.permissions import TurnAborted, can_use_tool, denial_state_for
from orbweaver.permissions.injection_probe import probe_tool_output
from orbweaver.store import Event, Job, Store, new_uuid

TOOL_SPEC = [
    {
        "name": "Read",
        "description": "Read a file in the session workspace.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
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
        "description": "Search file contents for a string.",
        "input_schema": {
            "type": "object",
            "properties": {"pattern": {"type": "string"}, "glob": {"type": "string"}},
            "required": ["pattern"],
        },
    },
    {
        "name": "Bash",
        "description": "Run a shell command in the workspace. Sandboxed by default (no network). "
        "Set unsandboxed true only when network or host access is required; that path is classified.",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {"type": "string"},
                "unsandboxed": {"type": "boolean"},
            },
            "required": ["command"],
        },
    },
    {
        "name": "WebFetch",
        "description": "HTTP GET a URL and return text.",
        "input_schema": {
            "type": "object",
            "properties": {"url": {"type": "string"}},
            "required": ["url"],
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
        "description": "Ask the host to schedule a job (ISO-8601 due_at).",
        "input_schema": {
            "type": "object",
            "properties": {
                "due_at": {"type": "string"},
                "message": {"type": "string"},
                "recurrence": {"type": "string"},
            },
            "required": ["due_at", "message"],
        },
    },
]


def _events_to_messages(events: list[Event]) -> list[dict[str, Any]]:
    """Build Anthropic messages from a (possibly projected) event list."""
    return events_to_messages(events)


def static_system() -> str:
    return (
        f"You are Orbweaver, a coding agent. The running gateway is Orbweaver {__version__} "
        f"(semantic version). If asked what version is running, answer {__version__}. "
        "Use tools to read and patch the workspace. "
        "In auto mode, in-project Write applies immediately. Prefer ProposePatch when a "
        "visible diff overlay helps the user. Use MemorySearch when past decisions might "
        "matter. Keep pins small. If a tool is blocked, find a safer path; do not try to "
        "bypass the permission gate."
    )


def build_agent_system(pins: str, extra: str = "") -> list[dict[str, Any]]:
    rest = pins or "(no pinned memory)"
    if extra:
        rest = rest + "\n\n" + extra
    return [
        {"type": "text", "text": static_system(), "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": rest},
    ]


def _blocked_tool_result(decision) -> str:
    if decision.behavior == "ask":
        return (
            "This action needs user approval and was not executed. "
            f"Reason: {decision.reason}. Ask the user, or pick a safer approach."
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
        return ws.read(inp["path"])[:20000]
    if name == "Write":
        ws.write(inp["path"], inp["content"])
        return f"wrote {inp['path']}"
    if name == "ProposePatch":
        result = ws.propose_patch(inp["path"], inp.get("old_string") or "", inp["new_string"])
        return json.dumps(result)[:200_000]
    if name == "Glob":
        return "\n".join(ws.glob(inp["pattern"])[:200])
    if name == "Grep":
        return "\n".join(ws.grep(inp["pattern"], inp.get("glob") or "**/*"))
    if name == "Bash":
        unsandboxed = bool(inp.get("unsandboxed"))
        return ws.bash(inp["command"], unsandboxed=unsandboxed)
    if name == "WebFetch":
        import httpx

        r = httpx.get(inp["url"], timeout=20.0, follow_redirects=True)  # noqa: ASYNC210
        return r.text[:500_000]
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
        job = Job(
            id=new_uuid(),
            due_at=due,
            payload={"message": inp.get("message") or ""},
            recurrence=inp.get("recurrence"),
            session_id=session_id,
        )
        await store.put_job(job)
        return json.dumps({"job_id": str(job.id), "due_at": due.isoformat()})
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
    max_rounds: int = 24,
    subagent_depth: int = 0,
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
        user_ev = await store.append_event(session_id, "user", {"text": user_text})
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
    }
    pins = await pinned_prompt(store)
    system = build_agent_system(pins, system_extra)
    active_tools = tools if tools is not None else TOOL_SPEC
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
        for _ in range(max_rounds):
            check()
            inj = inject_event()
            if inj is not None:
                inj.clear()
            events = await store.list_events(session_id)
            ctx["events"] = events
            messages = events_to_messages(prompt_events(events))
            messages = rehydrate_messages(messages, events, workspace)
            if not messages:
                messages = [{"role": "user", "content": user_text}]
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
        return produced
    except TurnAborted as e:
        await record_abort(e)
        return produced
    except TurnCancelled as e:
        e.produced = produced
        raise
