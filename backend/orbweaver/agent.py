"""Anthropic tool-calling agent loop with pin injection and compaction."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any, Callable
from uuid import UUID

from orbweaver.config import settings
from orbweaver.memory import pinned_prompt, remember, rewrite_search_query
from orbweaver.store import Event, Job, Store, new_uuid
from orbweaver.tokens import estimate_tokens

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
        "description": "Propose a search-replace patch. Does not write until the user accepts in VS Code.",
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
        "description": "Run a shell command in the workspace.",
        "input_schema": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
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
        "description": "Record a subagent task (runs inline in v1).",
        "input_schema": {
            "type": "object",
            "properties": {"task": {"type": "string"}},
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
    messages: list[dict[str, Any]] = []
    pending_tool: list[dict[str, Any]] = []
    for ev in events:
        k = ev.kind
        p = ev.payload
        if k == "user":
            messages.append({"role": "user", "content": p.get("text") or p.get("content") or ""})
        elif k == "assistant":
            messages.append({"role": "assistant", "content": p.get("text") or ""})
        elif k == "tool_call":
            pending_tool.append(
                {
                    "type": "tool_use",
                    "id": p.get("id") or str(ev.id),
                    "name": p.get("name"),
                    "input": p.get("input") or {},
                }
            )
        elif k in {"tool_result", "MemoryRecall"}:
            if pending_tool:
                messages.append({"role": "assistant", "content": pending_tool})
                pending_tool = []
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": p.get("tool_use_id") or p.get("id") or "unknown",
                            "content": p.get("content") or p.get("text") or json.dumps(p)[:8000],
                        }
                    ],
                }
            )
        elif k == "UserCorrection":
            messages.append(
                {
                    "role": "user",
                    "content": f"User correction: {p.get('text') or json.dumps(p)}",
                }
            )
        elif k == "compact_summary":
            messages.append(
                {"role": "user", "content": f"[compacted earlier turns]\n{p.get('text')}"}
            )
    if pending_tool:
        messages.append({"role": "assistant", "content": pending_tool})
    return messages


def event_token_count(events: list[Event]) -> int:
    return estimate_tokens(json.dumps([e.payload for e in events], default=str))


async def maybe_compact(store: Store, session_id: UUID) -> Event | None:
    events = await store.list_events(session_id)
    budget = int(settings.event_budget * settings.compact_ratio)
    if event_token_count(events) <= budget:
        return None
    keep = max(8, len(events) // 5)
    dropped, kept = events[:-keep], events[-keep:]
    summary = "Summary of earlier events:\n" + "\n".join(
        f"- {e.kind}: {json.dumps(e.payload, default=str)[:240]}" for e in dropped[-40:]
    )
    await remember(store, summary, source=f"compact:{session_id}")
    summary_ev = Event(
        id=new_uuid(),
        session_id=session_id,
        seq=1,
        kind="compact_summary",
        payload={"text": summary},
    )
    new_events = [summary_ev]
    for i, e in enumerate(kept, start=2):
        e.seq = i
        new_events.append(e)
    await store.replace_events(session_id, new_events)
    return summary_ev


async def run_tools(name: str, inp: dict[str, Any], ctx: dict[str, Any]) -> str:
    ws = ctx["workspace"]
    store: Store = ctx["store"]
    session_id: UUID = ctx["session_id"]
    if name == "Read":
        return ws.read(inp["path"])[:20000]
    if name == "Write":
        if ctx.get("workspace_kind") == "local" and not ctx.get("auto_write"):
            return json.dumps({"status": "needs_approval", "path": inp["path"]})
        ws.write(inp["path"], inp["content"])
        return f"wrote {inp['path']}"
    if name == "ProposePatch":
        result = ws.propose_patch(inp["path"], inp.get("old_string") or "", inp["new_string"])
        return json.dumps(result)[:15000]
    if name == "Glob":
        return "\n".join(ws.glob(inp["pattern"])[:200])
    if name == "Grep":
        return "\n".join(ws.grep(inp["pattern"], inp.get("glob") or "**/*"))
    if name == "Bash":
        return ws.bash(inp["command"])
    if name == "WebFetch":
        import httpx

        r = httpx.get(inp["url"], timeout=20.0, follow_redirects=True)
        return r.text[:15000]
    if name == "MemorySearch":
        events = await store.list_events(session_id)
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
        return f"subagent task recorded (inline v1): {inp['task'][:500]}"
    if name == "ScheduleTask":
        try:
            due = datetime.fromisoformat(str(inp["due_at"]).replace("Z", "+00:00"))
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


async def agent_turn(
    store: Store,
    session_id: UUID,
    user_text: str,
    workspace,
    workspace_kind: str = "local",
    emit: Callable[[dict[str, Any]], None] | None = None,
) -> list[Event]:
    await store.append_event(session_id, "user", {"text": user_text})
    await maybe_compact(store, session_id)
    produced: list[Event] = []

    def fire(ev: Event) -> None:
        produced.append(ev)
        if emit:
            emit({"kind": ev.kind, "payload": ev.payload, "id": str(ev.id)})

    if not settings.anthropic_api_key:
        ev = await store.append_event(
            session_id,
            "assistant",
            {
                "text": (
                    "ANTHROPIC_API_KEY is not set. Echo: "
                    + user_text[:500]
                    + "\nSet the key to enable the Claude tool loop."
                )
            },
        )
        fire(ev)
        return produced

    import anthropic

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    ctx = {
        "workspace": workspace,
        "store": store,
        "session_id": session_id,
        "workspace_kind": workspace_kind,
        "auto_write": workspace_kind == "docker",
    }
    pins = await pinned_prompt(store)
    system = (
        "You are Orbweaver, a coding agent. Use tools to read and patch the workspace. "
        "Prefer ProposePatch for file edits so the user can accept/reject in VS Code. "
        "Use MemorySearch when past decisions might matter. Keep pins small.\n\n"
        + (pins or "(no pinned memory)")
    )

    for _ in range(16):
        events = await store.list_events(session_id)
        messages = _events_to_messages(events)
        if not messages:
            messages = [{"role": "user", "content": user_text}]
        resp = client.messages.create(
            model=settings.orbweaver_model,
            max_tokens=4096,
            system=system,
            tools=TOOL_SPEC,
            messages=messages,
        )
        tool_uses = [b for b in resp.content if getattr(b, "type", None) == "tool_use"]
        texts = [b.text for b in resp.content if getattr(b, "type", None) == "text"]
        if texts:
            ev = await store.append_event(session_id, "assistant", {"text": "\n".join(texts)})
            fire(ev)
        if not tool_uses:
            break
        for block in tool_uses:
            call_ev = await store.append_event(
                session_id,
                "tool_call",
                {"id": block.id, "name": block.name, "input": block.input},
            )
            fire(call_ev)
            result = await run_tools(block.name, dict(block.input), ctx)
            kind = "MemoryRecall" if block.name == "MemorySearch" else "tool_result"
            payload = {"tool_use_id": block.id, "name": block.name, "content": result}
            if kind == "MemoryRecall":
                try:
                    parsed = json.loads(result)
                    payload["chunk_ids"] = parsed.get("chunk_ids") or []
                    payload["text"] = parsed.get("text") or result
                except json.JSONDecodeError:
                    payload["text"] = result
            res_ev = await store.append_event(session_id, kind, payload)
            fire(res_ev)
            if block.name == "ProposePatch":
                fire(
                    await store.append_event(
                        session_id, "patch_proposal", {"tool_use_id": block.id, "content": result}
                    )
                )
            if block.name == "ScheduleTask":
                fire(
                    await store.append_event(
                        session_id, "schedule_request", {"input": dict(block.input)}
                    )
                )
        await maybe_compact(store, session_id)
    return produced
