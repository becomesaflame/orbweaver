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
SUBAGENT_SYSTEM_EXTRA = (
    "You are an Orbweaver subagent. Complete the assigned task using tools. "
    "Do not ask the user questions. Do not schedule jobs or spawn further subagents. "
    "Report a concise result when done. Search project files with WorkspaceSearch; "
    "shared memory is available via MemorySearch. Pinned memory is already in this "
    "system prompt."
)


def is_subagent_session(entity: Entity | None) -> bool:
    if entity is None:
        return False
    jsonld = entity.jsonld or {}
    return jsonld.get("role") == "subagent" or bool(jsonld.get("parent_session"))


def child_tool_spec(tool_spec: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [t for t in tool_spec if t.get("name") not in CHILD_BLOCKED_TOOLS]


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
    from orbweaver.agent import TurnCancelled, agent_turn, resolve_channel, tools_for_channel

    if int(ctx.get("subagent_depth") or 0) >= 1:
        return "error: nested subagents are not allowed"

    task = str(inp.get("task") or "").strip()
    if not task:
        return "error: task is required"
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
        {"child_session_id": str(child_id), "task": task[:500], "label": label or title},
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
            tools=child_tool_spec(tools_for_channel(parent_channel)),
            system_extra=SUBAGENT_SYSTEM_EXTRA,
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
