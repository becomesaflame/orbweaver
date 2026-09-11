"""Nested subagent runner: child Session, shared workspace and memory.

Children run inline (the parent waits for the result) or in the background as
an ``asyncio.Task`` registered on the parent turn's ``ctx["children"]``. A
process-wide semaphore caps how many children run at once; each child has a
wall-clock timeout and a tool-round budget. When the parent turn is cancelled
or ends, ``settle_children`` cancels or collects whatever is still running.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from orbweaver.config import settings
from orbweaver.store import SESSION_TYPE, Entity, Event, Store, new_uuid, session_at_id

try:
    from orbweaver.permissions.handoff import review_subagent_return
except ImportError:  # pragma: no cover
    review_subagent_return = None  # type: ignore[assignment,misc]

log = logging.getLogger(__name__)

CHILD_BLOCKED_TOOLS = frozenset({"SpawnSubagent", "SubagentWait", "AskUser", "ScheduleTask"})
SUBAGENT_MAX_ROUNDS = 24
RESULT_TEXT_CAP = 8000
# How long settle_children waits for a hard-cancelled child to finalize its events.
CANCEL_FINALIZE_S = 5.0
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
        "MemoryReflect",
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
    "Report a concise result when done. Search project files with WorkspaceSearch; "
    "shared memory is available via MemorySearch and MemoryReflect. Pinned memory is already in this "
    "system prompt."
)
_TYPE_EXTRA = {
    "explore": (
        "You are an explore subagent. Read and search only. Do not write files, "
        "edit notebooks, or run shell commands."
    ),
    "implement": (
        "You are an implement subagent. Make the assigned change. Prefer StrReplace "
        "for existing files. Do not spawn further agents."
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


@dataclass
class ChildRun:
    """One spawned child, tracked on the parent turn's ``ctx["children"]``."""

    child_id: UUID
    name: str
    background: bool
    cancel: asyncio.Event
    started: float = field(default_factory=time.monotonic)
    task: asyncio.Task[str] | None = None
    status: str = "running"
    # True once the parent model has seen the result (tool_result, SubagentWait,
    # or an injected subagent_result event).
    delivered: bool = False

    @property
    def id(self) -> str:
        return str(self.child_id)

    @property
    def finished(self) -> bool:
        return self.task is not None and self.task.done()

    @property
    def elapsed_s(self) -> float:
        return round(time.monotonic() - self.started, 2)

    def result(self) -> dict[str, Any]:
        """Result body for the model. Safe to call before the task is done."""
        base = {"subagent_id": self.id, "child_session_id": self.id, "name": self.name}
        if not self.finished:
            return {**base, "status": "running", "text": ""}
        task = self.task
        assert task is not None
        if task.cancelled():
            return {**base, "status": "cancelled", "text": "error: subagent was cancelled"}
        exc = task.exception()
        if exc is not None:
            return {**base, "status": "error", "text": f"error: {exc}"[:RESULT_TEXT_CAP]}
        raw = task.result()
        try:
            parsed = json.loads(raw)
        except (TypeError, ValueError):
            parsed = None
        if not isinstance(parsed, dict):
            return {**base, "status": self.status, "text": str(raw)[:RESULT_TEXT_CAP]}
        return {**base, **parsed}


_semaphore: asyncio.Semaphore | None = None
_semaphore_key: tuple[int, int] | None = None


def subagent_semaphore() -> asyncio.Semaphore:
    """Process-wide cap on concurrently running children (rebuilt if the limit or loop changes)."""
    global _semaphore, _semaphore_key
    limit = max(1, int(settings.orbweaver_max_concurrent_subagents))
    key = (id(asyncio.get_running_loop()), limit)
    if _semaphore is None or _semaphore_key != key:
        _semaphore = asyncio.Semaphore(limit)
        _semaphore_key = key
    return _semaphore


def reset_subagent_semaphore_for_tests() -> None:
    global _semaphore, _semaphore_key
    _semaphore = None
    _semaphore_key = None


def children_of(ctx: dict[str, Any]) -> dict[str, ChildRun]:
    kids = ctx.get("children")
    if not isinstance(kids, dict):
        kids = {}
        ctx["children"] = kids
    return kids


def _float_param(raw: Any, default: float) -> float:
    if raw is None or raw == "":
        return float(default)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return float(default)


def _int_param(raw: Any, default: int, *, lo: int = 1, hi: int = 200) -> int:
    if raw is None or raw == "":
        return int(default)
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return int(default)
    return max(lo, min(hi, value))


def _child_emit(ctx: dict[str, Any], child_id: UUID) -> Any:
    """Stream child events to the parent's subscribers as subagent_event (not persisted)."""
    parent_emit = ctx.get("emit")
    if not callable(parent_emit):
        return None

    def emit(msg: dict[str, Any]) -> None:
        parent_emit(
            {
                "kind": "subagent_event",
                "payload": {
                    "subagent_id": str(child_id),
                    "child_session_id": str(child_id),
                    "event": msg,
                },
            }
        )

    return emit


async def _finish_child(
    run: ChildRun, ctx: dict[str, Any], status: str, error: str = ""
) -> str:
    store: Store = ctx["store"]
    child_id = run.child_id
    child_events = await store.list_events(child_id)
    if status == "ok" and any(e.kind == "turn_aborted" for e in child_events):
        status = "aborted"
    for ev in child_events:
        if ev.kind != "patch_proposal":
            continue
        payload = dict(ev.payload)
        payload["child_session_id"] = str(child_id)
        payload["subagent_id"] = str(child_id)
        await _parent_event(ctx, "patch_proposal", payload)

    text = _last_assistant(child_events)
    if error:
        text = error + (f"\npartial result: {text}" if text else "")
    text = text[:RESULT_TEXT_CAP]
    run.status = status
    await _parent_event(
        ctx,
        "subagent_finished",
        {
            "subagent_id": str(child_id),
            "child_session_id": str(child_id),
            "name": run.name,
            "status": status,
            "elapsed_s": run.elapsed_s,
            "background": run.background,
        },
    )
    body = json.dumps(
        {
            "subagent_id": str(child_id),
            "child_session_id": str(child_id),
            "status": status,
            "text": text,
        }
    )
    return await _maybe_review_return(ctx, child_events, body)


async def _run_child(
    run: ChildRun,
    ctx: dict[str, Any],
    *,
    task: str,
    kind: str,
    workspace_kind: str,
    parent_channel: str,
    timeout_s: float,
    max_rounds: int,
) -> str:
    from orbweaver.agent import TurnCancelled, agent_turn, session_tools

    store: Store = ctx["store"]
    status = "ok"
    error = ""
    try:
        async with subagent_semaphore():
            tools = child_tool_spec(await session_tools(ctx["workspace"], parent_channel), kind)
            await asyncio.wait_for(
                agent_turn(
                    store,
                    run.child_id,
                    task,
                    ctx["workspace"],
                    workspace_kind=workspace_kind,
                    emit=_child_emit(ctx, run.child_id),
                    cancel=run.cancel,
                    headless=True,
                    tools=tools,
                    system_extra=subagent_system_extra(kind),
                    max_rounds=max_rounds,
                    subagent_depth=1,
                    channel=parent_channel or None,
                ),
                timeout=timeout_s if timeout_s > 0 else None,
            )
    except TurnCancelled:
        status = "cancelled"
    except TimeoutError:
        status = "timeout"
        error = f"error: subagent timed out after {timeout_s:g}s and was cancelled"
    except asyncio.CancelledError:
        # settle_children hard-cancelled us; still record what the child produced.
        status = "cancelled"
        error = "error: subagent was cancelled"
    except Exception as e:  # pragma: no cover - defensive; the loop records its own aborts
        log.exception("subagent %s failed", run.child_id)
        status = "error"
        error = f"error: {e}"
    return await _finish_child(run, ctx, status, error)


async def run_subagent(inp: dict[str, Any], ctx: dict[str, Any]) -> str:
    from orbweaver.agent import resolve_channel

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
    background = bool(inp.get("background"))
    timeout_s = _float_param(inp.get("timeout_s"), settings.orbweaver_subagent_timeout_s)
    max_rounds = _int_param(inp.get("max_rounds"), settings.orbweaver_subagent_max_rounds)

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
        "background": background,
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

    # Inline children share the parent's cancel flag (the parent is blocked on them).
    # Background children get their own so settle_children can stop them individually.
    parent_cancel = ctx.get("cancel")
    cancel = (
        asyncio.Event()
        if background or not isinstance(parent_cancel, asyncio.Event)
        else parent_cancel
    )
    run = ChildRun(child_id=child_id, name=label or title, background=background, cancel=cancel)
    children_of(ctx)[run.id] = run
    await _parent_event(
        ctx,
        "subagent_started",
        {
            "subagent_id": str(child_id),
            "child_session_id": str(child_id),
            "name": run.name,
            "task": task[:500],
            "label": label or title,
            "type": kind,
            "background": background,
            "timeout_s": timeout_s,
            "max_rounds": max_rounds,
        },
    )
    run.task = asyncio.create_task(
        _run_child(
            run,
            ctx,
            task=task,
            kind=kind,
            workspace_kind=workspace_kind,
            parent_channel=parent_channel,
            timeout_s=timeout_s,
            max_rounds=max_rounds,
        ),
        name=f"subagent-{child_id}",
    )
    if background:
        return json.dumps(
            {
                "subagent_id": str(child_id),
                "child_session_id": str(child_id),
                "status": "running",
                "name": run.name,
            }
        )
    try:
        return await run.task
    finally:
        run.delivered = True


def _resolve_child(children: dict[str, ChildRun], raw: Any) -> ChildRun | None:
    key = str(raw or "").strip()
    if not key:
        return None
    if key in children:
        return children[key]
    matches = [r for cid, r in children.items() if cid.startswith(key)]
    return matches[0] if len(matches) == 1 else None


async def wait_subagents(inp: dict[str, Any], ctx: dict[str, Any]) -> str:
    """SubagentWait: block until the named (default: all uncollected) children finish."""
    from orbweaver.agent import _await_or_cancel

    children = children_of(ctx)
    raw_ids = inp.get("ids") or []
    if isinstance(raw_ids, str):
        raw_ids = [raw_ids]
    unknown: list[str] = []
    targets: list[ChildRun] = []
    if raw_ids:
        for rid in raw_ids:
            run = _resolve_child(children, rid)
            if run is None:
                unknown.append(str(rid))
            elif run not in targets:
                targets.append(run)
    else:
        targets = [r for r in children.values() if not r.delivered]
    if not targets and not unknown:
        return json.dumps(
            {"results": [], "note": "no uncollected subagents in this turn"}
        )
    pending = [r.task for r in targets if r.task is not None and not r.task.done()]
    if pending:
        await _await_or_cancel(asyncio.wait(pending), ctx.get("cancel"))
    results: list[dict[str, Any]] = []
    for run in targets:
        run.delivered = True
        results.append(run.result())
    for rid in unknown:
        results.append(
            {
                "subagent_id": rid,
                "status": "unknown",
                "text": "error: no subagent with this id was spawned in this turn",
            }
        )
    return json.dumps({"results": results})


def collect_finished(ctx: dict[str, Any]) -> list[dict[str, Any]]:
    """Results of finished children the model has not seen yet; marks them delivered."""
    out: list[dict[str, Any]] = []
    for run in children_of(ctx).values():
        if run.delivered or not run.finished:
            continue
        run.delivered = True
        out.append(run.result())
    return out


def format_subagent_result(payload: dict[str, Any]) -> str:
    """User-side text the model sees for an injected background result."""
    name = str(payload.get("name") or "").strip()
    sid = str(payload.get("subagent_id") or "")
    status = str(payload.get("status") or "")
    text = str(payload.get("text") or "")
    head = f"[Background subagent {sid}"
    if name:
        head += f" ({name})"
    head += f" finished: status={status}]"
    return f"{head}\n{text}" if text else head


async def settle_children(ctx: dict[str, Any], *, cancelled: bool) -> None:
    """Parent turn is ending: cancel or collect background children, record events."""
    children = children_of(ctx)
    if not children:
        return
    running = [r for r in children.values() if r.task is not None and not r.task.done()]
    if running and not cancelled:
        grace = max(0.0, float(settings.orbweaver_subagent_grace_s))
        if grace > 0:
            await asyncio.wait([r.task for r in running if r.task], timeout=grace)
        running = [r for r in running if r.task is not None and not r.task.done()]
    if running:
        for r in running:
            r.cancel.set()
            if r.task is not None:
                r.task.cancel()
        await asyncio.wait(
            [r.task for r in running if r.task], timeout=CANCEL_FINALIZE_S
        )
        reason = "parent_cancelled" if cancelled else "parent_turn_ended"
        for r in running:
            r.status = "cancelled"
            await _parent_event(
                ctx,
                "subagent_cancelled",
                {
                    "subagent_id": r.id,
                    "child_session_id": r.id,
                    "name": r.name,
                    "reason": reason,
                    "elapsed_s": r.elapsed_s,
                },
            )
    for r in children.values():
        if r.delivered or not r.finished:
            continue
        r.delivered = True
        await _parent_event(ctx, "subagent_result", r.result())
