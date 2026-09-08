"""Nested subagent runner: child Session, shared workspace and memory."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from orbweaver.store import SESSION_TYPE, Entity, Event, Store, new_uuid, session_at_id

try:
    from orbweaver.permissions.handoff import review_subagent_return
except ImportError:  # pragma: no cover
    review_subagent_return = None

CHILD_BLOCKED_TOOLS = frozenset({"SpawnSubagent", "AskUser", "ScheduleTask"})
SUBAGENT_MAX_ROUNDS = 48
RESULT_TEXT_CAP = 8000
DEFAULT_SUBAGENT_TYPE = "implement"
SUBAGENT_TYPES = frozenset({"explore", "implement", "shell"})
_TYPE_ALIASES = {
    "explore": "explore",
    "research": "explore",
    "implement": "implement",
    "impl": "implement",
    "general": "implement",
    "shell": "shell",
    "bash": "shell",
}
EXPLORE_TOOLS = frozenset(
    {
        "Read",
        "Glob",
        "Grep",
        "WebFetch",
        "WebSearch",
        "MemorySearch",
        "MemoryGraph",
        "ReadLints",
    }
)
SHELL_TOOLS = frozenset({"Bash", "Read", "Glob", "Grep", "ReadLints"})
_TYPE_TOOLS: dict[str, frozenset[str] | None] = {
    "explore": EXPLORE_TOOLS,
    "implement": None,
    "shell": SHELL_TOOLS,
}
SUBAGENT_SYSTEM_EXTRA = (
    "You are an Orbweaver subagent. Complete the assigned task using tools. "
    "Do not ask the user questions. Do not schedule jobs or spawn further subagents. "
    "Report a concise result when done. Shared memory is available via MemorySearch; "
    "pinned memory is already in this system prompt."
)
_TYPE_EXTRA = {
    "explore": (
        "You are an explore subagent. Read and search only. Do not write files, "
        "edit notebooks, or run shell commands."
    ),
    "implement": (
        "You are an implement subagent. Make the assigned change. Prefer patches. "
        "Do not spawn further agents."
    ),
    "shell": (
        "You are a shell subagent. Run commands for the assigned task. "
        "Do not write workspace files with Write; use Bash only when needed."
    ),
}


def is_subagent_session(entity: Entity | None) -> bool:
    if entity is None:
        return False
    jsonld = entity.jsonld or {}
    return jsonld.get("role") == "subagent" or bool(jsonld.get("parent_session"))


def normalize_subagent_type(raw: Any) -> str | None:
    spec = str(raw or "").strip().lower()
    if not spec:
        return DEFAULT_SUBAGENT_TYPE
    return _TYPE_ALIASES.get(spec)


def child_tool_spec(
    tool_spec: list[dict[str, Any]],
    subagent_type: str = DEFAULT_SUBAGENT_TYPE,
) -> list[dict[str, Any]]:
    allowed = _TYPE_TOOLS.get(subagent_type, None)
    out = [t for t in tool_spec if t.get("name") not in CHILD_BLOCKED_TOOLS]
    if allowed is None:
        return out
    return [t for t in out if t.get("name") in allowed]


def subagent_system_extra(subagent_type: str = DEFAULT_SUBAGENT_TYPE) -> str:
    extra = _TYPE_EXTRA.get(subagent_type) or SUBAGENT_SYSTEM_EXTRA
    return SUBAGENT_SYSTEM_EXTRA + " " + extra


def _last_assistant(events: list[Event]) -> str:
    for ev in reversed(events):
        if ev.kind == "assistant":
            return str(ev.payload.get("text") or "")
    return ""


async def _parent_event(ctx: dict[str, Any], kind: str, payload: dict[str, Any]) -> Event:
    store: Store = ctx["store"]
    parent_id: UUID = ctx["session_id"]
    ev = await store.append_event(parent_id, kind, payload)
    fire = ctx.get("fire")
    if callable(fire):
        fire(ev)
    return ev


async def _maybe_review_return(
    ctx: dict[str, Any],
    child_events: list[Event],
    payload: str,
) -> str:
    if review_subagent_return is None:
        return payload
    store: Store = ctx["store"]
    parent_events = await store.list_events(ctx["session_id"])
    child_tool_calls = [
        {"name": e.payload.get("name"), "input": e.payload.get("input") or {}}
        for e in child_events
        if e.kind == "tool_call"
    ]
    reviewed = await review_subagent_return(
        parent_events,
        child_tool_calls,
        payload,
        workspace=ctx.get("workspace"),
    )
    return str(reviewed.get("payload") or payload)


async def run_subagent(inp: dict[str, Any], ctx: dict[str, Any]) -> str:
    from orbweaver.agent import TurnCancelled, agent_turn, resolve_channel, session_tools

    if int(ctx.get("subagent_depth") or 0) >= 1:
        return "error: nested subagents are not allowed"

    task = str(inp.get("task") or "").strip()
    if not task:
        return "error: task is required"
    kind = normalize_subagent_type(inp.get("type") or inp.get("role"))
    if kind is None:
        return f"error: type must be one of {sorted(SUBAGENT_TYPES)}"
    label = str(inp.get("label") or "").strip()
    title = (label or task)[:80]

    store: Store = ctx["store"]
    parent_id: UUID = ctx["session_id"]
    parent = await store.get_entity(parent_id)
    workspace_uri = "workspace:default"
    workspace_kind = str(ctx.get("workspace_kind") or "local")
    parent_channel = resolve_channel(
        parent.jsonld if parent else None, channel=ctx.get("channel")
    )
    if parent and parent.jsonld:
        workspace_uri = str(parent.jsonld.get("workspace_uri") or workspace_uri)
        workspace_kind = str(parent.jsonld.get("workspace_kind") or workspace_kind)

    child_id = new_uuid()
    child_jsonld = {
        "@id": session_at_id(child_id),
        "@type": SESSION_TYPE,
        "workspace_uri": workspace_uri,
        "workspace_kind": workspace_kind,
        "title": title,
        "status": "active",
        "role": "subagent",
        "subagent_type": kind,
        "parent_session": session_at_id(parent_id),
        "created_at": datetime.now(UTC).isoformat(),
    }
    if parent_channel:
        child_jsonld["channel"] = parent_channel
    child = Entity(
        id=child_id,
        at_id=session_at_id(child_id),
        at_type=SESSION_TYPE,
        jsonld=child_jsonld,
    )
    await store.put_entity(child)
    await _parent_event(
        ctx,
        "subagent_started",
        {
            "child_session_id": str(child_id),
            "task": task[:500],
            "label": label or title,
            "type": kind,
        },
    )

    status = "ok"
    try:
        await agent_turn(
            store,
            child_id,
            task,
            ctx["workspace"],
            workspace_kind=workspace_kind,
            cancel=ctx.get("cancel"),
            headless=True,
            tools=child_tool_spec(
                await session_tools(ctx["workspace"], parent_channel), kind
            ),
            system_extra=subagent_system_extra(kind),
            max_rounds=SUBAGENT_MAX_ROUNDS,
            subagent_depth=1,
            channel=parent_channel or None,
        )
    except TurnCancelled:
        status = "cancelled"

    child_events = await store.list_events(child_id)
    if status == "ok" and any(e.kind == "turn_aborted" for e in child_events):
        status = "aborted"
    for ev in child_events:
        if ev.kind != "patch_proposal":
            continue
        payload = dict(ev.payload)
        payload["child_session_id"] = str(child_id)
        await _parent_event(ctx, "patch_proposal", payload)

    text = _last_assistant(child_events)[:RESULT_TEXT_CAP]
    await _parent_event(
        ctx,
        "subagent_finished",
        {"child_session_id": str(child_id), "status": status},
    )
    body = json.dumps(
        {
            "child_session_id": str(child_id),
            "status": status,
            "text": text,
        }
    )
    return await _maybe_review_return(ctx, child_events, body)
