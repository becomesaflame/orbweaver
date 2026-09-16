"""Session router: lets a singleton channel session drive any other session.

A channel such as Telegram (and, later, voice) owns one *operator* session per
identity. That session can be *attached* to another session. While attached,
messages from the channel run ``agent_turn`` on the target instead of the
operator, so the user continues the exact conversation they started in the web
UI or VS Code: same event stream, workspace, model, todos, and session rules.

Two process-wide registries live here so channels never import each other:

* **Sinks** — per-session fan-out for live turn events. The gateway registers
  one global sink (WebSocket subscribers); a channel registers a per-session
  sink on the sessions it is bound to (the operator itself plus the attached
  target). Every channel passes :func:`emitter` to ``agent_turn`` so a turn
  started anywhere is visible everywhere.
* **Attach hooks** — called after :func:`attach` / :func:`detach` so the owning
  channel can move its sink to the new target.

Nothing in this module knows about Telegram; the tool handlers
(:func:`run_operator_tool`) work on the operator session in ``ctx``.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from orbweaver.store import SESSION_TYPE, Entity, Event, Store, session_at_id
from orbweaver.turns import get as get_running_turn

log = logging.getLogger(__name__)

ATTACHED_KEY = "attached_session"
ATTACHED_AT_KEY = "attached_at"
OPERATOR_ROLE = "operator"
SESSION_IRI_PREFIX = "urn:orbweaver:session:"

Sink = Callable[[dict[str, Any]], None]
GlobalSink = Callable[[UUID, dict[str, Any]], None]
AttachHook = Callable[[Entity, UUID | None], None]

_global_sinks: list[GlobalSink] = []
_session_sinks: dict[UUID, dict[str, Sink]] = {}
_attach_hooks: list[AttachHook] = []


class RouterError(ValueError):
    """Bad attach target or unresolvable session reference."""


class AmbiguousSessionRef(RouterError):
    def __init__(self, ref: str, matches: list[Entity]) -> None:
        self.ref = ref
        self.matches = matches
        ids = ", ".join(str(m.id)[:8] for m in matches[:6])
        super().__init__(f"'{ref}' matches several sessions: {ids}")


# --------------------------------------------------------------------------- sinks


def add_global_sink(fn: GlobalSink) -> None:
    if fn not in _global_sinks:
        _global_sinks.append(fn)


def add_sink(session_id: UUID, key: str, fn: Sink) -> None:
    _session_sinks.setdefault(session_id, {})[key] = fn


def remove_sink(session_id: UUID, key: str) -> None:
    bucket = _session_sinks.get(session_id)
    if bucket is None:
        return
    bucket.pop(key, None)
    if not bucket:
        _session_sinks.pop(session_id, None)


def sink_keys(session_id: UUID) -> list[str]:
    return sorted(_session_sinks.get(session_id, {}))


def emit(session_id: UUID, msg: dict[str, Any]) -> None:
    """Fan one live frame out to every sink bound to ``session_id``. Never raises."""
    # Snapshots: a sink may add or remove sinks while we iterate.
    for global_fn in tuple(_global_sinks):
        try:
            global_fn(session_id, msg)
        except Exception:
            log.exception("global sink failed for session %s", session_id)
    for key, session_fn in tuple(_session_sinks.get(session_id, {}).items()):
        try:
            session_fn(msg)
        except Exception:
            log.exception("sink %s failed for session %s", key, session_id)


def emitter(session_id: UUID) -> Sink:
    """``emit`` hook for ``agent_turn`` on ``session_id``."""

    def _emit(msg: dict[str, Any]) -> None:
        emit(session_id, msg)

    return _emit


def add_attach_hook(fn: AttachHook) -> None:
    if fn not in _attach_hooks:
        _attach_hooks.append(fn)


def reset_for_tests() -> None:
    """Clear per-session sinks and attach hooks. Global sinks are process-level
    registrations (the gateway's WebSocket fan-out) and survive."""
    _session_sinks.clear()
    _attach_hooks.clear()


# ------------------------------------------------------------------- attach pointer


def is_operator_session(entity: Entity | None) -> bool:
    if entity is None:
        return False
    jsonld = entity.jsonld or {}
    return jsonld.get("role") == OPERATOR_ROLE or jsonld.get("telegram_user_id") is not None


def _is_subagent(entity: Entity) -> bool:
    jsonld = entity.jsonld or {}
    return jsonld.get("role") == "subagent" or bool(jsonld.get("parent_session"))


def parse_session_ref(raw: Any) -> UUID | None:
    """UUID from a session IRI, a bare UUID string, or None."""
    text = str(raw or "").strip().removeprefix(SESSION_IRI_PREFIX)
    try:
        return UUID(text)
    except (TypeError, ValueError):
        return None


def attached_id(operator: Entity | None) -> UUID | None:
    if operator is None:
        return None
    return parse_session_ref((operator.jsonld or {}).get(ATTACHED_KEY))


def _fire_hooks(operator: Entity, target: UUID | None) -> None:
    for hook in tuple(_attach_hooks):
        try:
            hook(operator, target)
        except Exception:
            log.exception("attach hook failed for operator %s", operator.id)


async def attach(store: Store, operator: Entity, target_id: UUID) -> Entity:
    """Point ``operator`` at ``target_id``. Returns the target session."""
    if target_id == operator.id:
        raise RouterError("cannot attach a session to itself")
    target = await store.get_entity(target_id)
    if target is None or target.at_type != SESSION_TYPE:
        raise RouterError(f"session {target_id} not found")
    if _is_subagent(target):
        raise RouterError("cannot attach to a subagent session; use its parent")
    if is_operator_session(target):
        raise RouterError("cannot attach to another channel's operator session")
    operator.jsonld[ATTACHED_KEY] = session_at_id(target_id)
    operator.jsonld[ATTACHED_AT_KEY] = datetime.now(UTC).isoformat()
    await store.put_entity(operator)
    _fire_hooks(operator, target_id)
    return target


async def detach(store: Store, operator: Entity) -> UUID | None:
    """Clear the pointer. Returns the previous target id (None when not attached)."""
    previous = attached_id(operator)
    operator.jsonld.pop(ATTACHED_KEY, None)
    operator.jsonld.pop(ATTACHED_AT_KEY, None)
    await store.put_entity(operator)
    _fire_hooks(operator, None)
    return previous


async def resolve_target(store: Store, operator: Entity) -> Entity:
    """The session a message from ``operator``'s channel should run on.

    A dangling pointer (target deleted) is cleared so the operator answers again.
    """
    target_id = attached_id(operator)
    if target_id is None:
        return operator
    target = await store.get_entity(target_id)
    if target is None or target.at_type != SESSION_TYPE:
        log.info("router: attached session %s is gone; detaching %s", target_id, operator.id)
        await detach(store, operator)
        return operator
    return target


# ----------------------------------------------------------------------- discovery


def _first_user_text(events: list[Event]) -> str:
    for ev in events:
        if ev.kind == "user":
            return str((ev.payload or {}).get("text") or "").strip()
    return ""


def _last_text(events: list[Event], kind: str) -> str:
    for ev in reversed(events):
        if ev.kind == kind:
            return str((ev.payload or {}).get("text") or "").strip()
    return ""


def _pending_question(events: list[Event]) -> str:
    from orbweaver.agent import pending_ask_user

    pending = pending_ask_user(events)
    if pending is None:
        return ""
    question = str(((pending.payload or {}).get("input") or {}).get("question") or "")
    if not question:
        for ev in reversed(events):
            if ev.kind == "ask_user":
                question = str((ev.payload or {}).get("question") or "")
                break
    return question.strip()


def display_title(jsonld: dict[str, Any], events: list[Event]) -> str:
    title = str(jsonld.get("title") or "").strip()
    if title in {"", "web", "session", "New chat", "vscode"}:
        line = _first_user_text(events).split("\n", 1)[0].strip()
        return line[:80] or "New chat"
    return title[:80]


async def candidate_sessions(store: Store, *, exclude: Iterable[UUID] = ()) -> list[Entity]:
    """Sessions a channel may attach to: no subagents, no operators, none in ``exclude``."""
    skip = set(exclude)
    out: list[Entity] = []
    for ent in await store.list_entities(SESSION_TYPE):
        if ent.id in skip or _is_subagent(ent) or is_operator_session(ent):
            continue
        out.append(ent)
    return out


def _running_channel(session_id: UUID) -> str | None:
    state = get_running_turn(session_id)
    if state is None:
        return None
    return state.channel or "unknown"


async def session_row(store: Store, ent: Entity, events: list[Event] | None = None) -> dict[str, Any]:
    events = await store.list_events(ent.id) if events is None else events
    jsonld = ent.jsonld or {}
    last_at = events[-1].created_at.isoformat() if events else str(jsonld.get("created_at") or "")
    running = _running_channel(ent.id)
    return {
        "id": str(ent.id),
        "short_id": str(ent.id)[:8],
        "title": display_title(jsonld, events),
        "workspace_uri": str(jsonld.get("workspace_uri") or "workspace:default"),
        "channel": str(jsonld.get("channel") or ""),
        "model": str(jsonld.get("model") or ""),
        "last_event_at": last_at,
        "event_count": len(events),
        "running": running is not None,
        "running_channel": running or "",
        "pending_question": _pending_question(events),
    }


async def list_sessions(store: Store, *, exclude: Iterable[UUID] = ()) -> list[dict[str, Any]]:
    rows = [await session_row(store, ent) for ent in await candidate_sessions(store, exclude=exclude)]
    rows.sort(key=lambda r: r["last_event_at"] or "", reverse=True)
    return rows


async def find_session(store: Store, ref: str, *, exclude: Iterable[UUID] = ()) -> Entity | None:
    """Resolve a user-typed reference: full UUID, UUID prefix, or title substring.

    Raises :class:`AmbiguousSessionRef` when more than one session matches.
    """
    text = str(ref or "").strip()
    if not text:
        return None
    exact = parse_session_ref(text)
    candidates = await candidate_sessions(store, exclude=exclude)
    if exact is not None:
        return next((c for c in candidates if c.id == exact), None)
    lowered = text.lower()
    by_prefix = [c for c in candidates if str(c.id).startswith(lowered)]
    if len(by_prefix) == 1:
        return by_prefix[0]
    if len(by_prefix) > 1:
        raise AmbiguousSessionRef(text, by_prefix)
    by_title: list[Entity] = []
    for c in candidates:
        events = await store.list_events(c.id)
        title = display_title(c.jsonld or {}, events).lower()
        if lowered in title or lowered in str((c.jsonld or {}).get("workspace_uri") or "").lower():
            by_title.append(c)
    if len(by_title) == 1:
        return by_title[0]
    if len(by_title) > 1:
        raise AmbiguousSessionRef(text, by_title)
    return None


async def session_digest(store: Store, session_id: UUID) -> dict[str, Any]:
    """What a person needs to pick a session back up, from its event log."""
    from orbweaver.compact.project import BOUNDARY_KINDS
    from orbweaver.todos import normalize_todos

    ent = await store.get_entity(session_id)
    if ent is None or ent.at_type != SESSION_TYPE:
        raise RouterError(f"session {session_id} not found")
    events = await store.list_events(session_id)
    row = await session_row(store, ent, events)
    summary = ""
    for ev in reversed(events):
        if ev.kind in BOUNDARY_KINDS:
            summary = str((ev.payload or {}).get("text") or "").strip()
            break
    todos = normalize_todos((ent.jsonld or {}).get("todos"))
    open_todos = [t for t in todos if t.get("status") not in {"completed", "cancelled"}]
    row.update(
        {
            "last_user_text": _last_text(events, "user")[:600],
            "last_assistant_text": _last_text(events, "assistant")[:1200],
            "todos": todos,
            "open_todo_count": len(open_todos),
            "compact_summary": summary[:800],
            "created_at": str((ent.jsonld or {}).get("created_at") or ""),
        }
    )
    return row


def _age(iso: str) -> str:
    if not iso:
        return ""
    try:
        then = datetime.fromisoformat(iso)
    except ValueError:
        return ""
    if then.tzinfo is None:
        then = then.replace(tzinfo=UTC)
    secs = max(0, int((datetime.now(UTC) - then).total_seconds()))
    if secs < 60:
        return f"{secs}s ago"
    if secs < 3600:
        return f"{secs // 60}m ago"
    if secs < 86400:
        return f"{secs // 3600}h ago"
    return f"{secs // 86400}d ago"


def format_session_list(rows: list[dict[str, Any]], attached: UUID | None = None, limit: int = 15) -> str:
    if not rows:
        return "No other sessions yet."
    lines = []
    for row in rows[:limit]:
        marks = []
        if attached is not None and row["id"] == str(attached):
            marks.append("attached")
        if row["running"]:
            marks.append(f"running via {row['running_channel']}")
        if row["pending_question"]:
            marks.append("waiting for your answer")
        suffix = f"  [{', '.join(marks)}]" if marks else ""
        channel = f" {row['channel']}" if row["channel"] else ""
        age = _age(row["last_event_at"])
        age_s = f" · {age}" if age else ""
        lines.append(f"{row['short_id']}  {row['title']}{channel}{age_s}{suffix}")
    more = len(rows) - limit
    if more > 0:
        lines.append(f"… {more} more")
    return "\n".join(lines)


def format_digest(d: dict[str, Any]) -> str:
    lines = [f"{d['title']}  ({d['short_id']})", f"workspace: {d['workspace_uri']}"]
    if d.get("channel"):
        lines[-1] += f" · home channel: {d['channel']}"
    age = _age(d.get("last_event_at") or "")
    if age:
        lines.append(f"last activity: {age}")
    if d.get("running"):
        lines.append(f"a turn is running now (via {d.get('running_channel') or 'unknown'})")
    if d.get("pending_question"):
        lines.append(f"waiting for your answer: {d['pending_question']}")
    if d.get("open_todo_count"):
        open_items = [
            f"- [{t.get('status')}] {t.get('content')}"
            for t in d.get("todos") or []
            if t.get("status") not in {"completed", "cancelled"}
        ][:8]
        lines.append("open todos:\n" + "\n".join(open_items))
    if d.get("compact_summary"):
        lines.append(f"earlier (compacted): {d['compact_summary']}")
    if d.get("last_user_text"):
        lines.append(f"you last said: {d['last_user_text']}")
    if d.get("last_assistant_text"):
        lines.append(f"last reply: {d['last_assistant_text']}")
    return "\n\n".join(lines)[:3500]


# -------------------------------------------------------------------- operator tools

OPERATOR_TOOL_SPEC: list[dict[str, Any]] = [
    {
        "name": "ListSessions",
        "description": (
            "List the user's other Orbweaver sessions (web, VS Code, cron) with id, title, "
            "workspace, last activity, whether a turn is running, and whether one is waiting "
            "for the user's answer. Use this to find what the user was working on elsewhere."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "SessionDigest",
        "description": (
            "Summary of one session: last user message, last reply, open todos, pending "
            "question, compacted history. Accepts a session id, an id prefix, or a title fragment."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"session": {"type": "string"}},
            "required": ["session"],
        },
    },
    {
        "name": "AttachSession",
        "description": (
            "Hand this chat over to another session. After this call the user's next messages "
            "run on that session (its history, workspace, and model) until they detach. Use when "
            "the user wants to continue work they started in the web UI or VS Code. If several "
            "sessions could match, ask which one before attaching."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"session": {"type": "string"}},
            "required": ["session"],
        },
    },
    {
        "name": "DetachSession",
        "description": "Return this chat to its own session (undo AttachSession).",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
]
OPERATOR_TOOLS = frozenset(t["name"] for t in OPERATOR_TOOL_SPEC)

OPERATOR_SYSTEM_EXTRA = (
    "You are also the operator for this chat: the user's always-on entry point from their "
    "phone. Besides this workspace you can see every other Orbweaver session (web, VS Code, "
    "cron). When the user asks what they were working on, or wants to continue something "
    "they started at their desk, call ListSessions, then SessionDigest on the likely match, "
    "and summarize. To hand the conversation over, call AttachSession: from then on the "
    "user's messages go straight to that session with its full history and workspace, not "
    "to you, until they send /detach. If more than one session could match, ask which. Do "
    "not redo work that belongs to another session; attach instead."
)

ATTACHED_TURN_HINT = (
    "The user is driving this session from Telegram right now (away from their desk). "
    "This session's history, workspace, and plan still apply. Keep replies short and "
    "skimmable; long output belongs in files the user can read later."
)


async def operator_tools(workspace: Any, channel: str) -> list[dict[str, Any]]:
    """Built-in + MCP tools for ``channel`` plus the operator tools."""
    from orbweaver.agent import session_tools

    return [*await session_tools(workspace, channel), *OPERATOR_TOOL_SPEC]


async def run_operator_tool(name: str, inp: dict[str, Any], ctx: dict[str, Any]) -> str:
    store: Store = ctx["store"]
    operator_id: UUID = ctx["session_id"]
    operator = await store.get_entity(operator_id)
    if operator is None:
        return "error: operator session not found"
    exclude = {operator_id}
    if name == "ListSessions":
        rows = await list_sessions(store, exclude=exclude)
        return json.dumps({"attached": _iri_or_none(attached_id(operator)), "sessions": rows})
    if name in {"SessionDigest", "AttachSession"}:
        ref = str(inp.get("session") or "").strip()
        if not ref:
            return "error: session is required (id, id prefix, or title fragment)"
        try:
            target = await find_session(store, ref, exclude=exclude)
        except AmbiguousSessionRef as e:
            return json.dumps(
                {
                    "error": str(e),
                    "matches": [
                        {"id": str(m.id), "title": str((m.jsonld or {}).get("title") or "")}
                        for m in e.matches
                    ],
                }
            )
        if target is None:
            return f"error: no session matches '{ref}'"
        if name == "SessionDigest":
            return json.dumps(await session_digest(store, target.id))
        try:
            await attach(store, operator, target.id)
        except RouterError as e:
            return f"error: {e}"
        digest = await session_digest(store, target.id)
        return json.dumps(
            {
                "attached": str(target.id),
                "title": digest["title"],
                "workspace_uri": digest["workspace_uri"],
                "note": (
                    "The user's next messages run on this session. Tell them what it was "
                    "doing and that they can send /detach to come back."
                ),
                "digest": digest,
            }
        )
    if name == "DetachSession":
        previous = await detach(store, operator)
        return json.dumps({"detached": _iri_or_none(previous)})
    return f"unknown operator tool {name}"


def _iri_or_none(uid: UUID | None) -> str | None:
    return session_at_id(uid) if uid is not None else None
