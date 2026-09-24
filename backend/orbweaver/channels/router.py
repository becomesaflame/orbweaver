"""Session router: a channel operator discovers other sessions and prompts them.

A channel such as Telegram (and, later, voice) owns one *operator* session per
identity. That session stays itself: messages from the channel run on the
operator. To continue desk work the operator calls ``PromptSession``, which
injects or starts a turn on the target (same event stream, workspace, model,
todos, and tools) without moving the channel's cursor into that chat.

Two process-wide registries live here so channels never import each other:

* **Sinks** — per-session fan-out for live turn events. The gateway registers
  one global sink (WebSocket subscribers); a channel may register a global sink
  (Telegram dispatcher) or a per-session sink. Every channel passes
  :func:`emitter` to ``agent_turn`` so a turn started anywhere is visible
  everywhere.
* **Watch set** — sessions the operator recently prompted or explicitly
  watched (``WatchSession``), persisted on the operator JSON-LD so a restart
  still reports that turn's completion.

Nothing in this module is Telegram-specific except that ``PromptSession`` /
``StopSession`` delegate to the channel adapter (lazy import) so the operator
tools can start or stop a turn on another session.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from orbweaver.store import (
    SESSION_TYPE,
    Entity,
    Event,
    Store,
    is_deleted_session,
    new_uuid,
    session_at_id,
)
from orbweaver.turns import get as get_running_turn

log = logging.getLogger(__name__)

ATTACHED_KEY = "attached_session"
ATTACHED_AT_KEY = "attached_at"
PROMPTED_KEY = "prompted_sessions"
LAST_PROMPTED_KEY = "last_prompted_session"
OPERATOR_ROLE = "operator"
SESSION_IRI_PREFIX = "urn:orbweaver:session:"

Sink = Callable[[dict[str, Any]], None]
GlobalSink = Callable[[UUID, dict[str, Any]], None]

_global_sinks: list[GlobalSink] = []
_session_sinks: dict[UUID, dict[str, Sink]] = {}


class RouterError(ValueError):
    """Bad prompt target or unresolvable session reference."""


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


def remove_global_sink(fn: GlobalSink) -> None:
    try:
        _global_sinks.remove(fn)
    except ValueError:
        return


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


def reset_for_tests() -> None:
    """Clear per-session sinks. Global sinks are process-level registrations
    (the gateway's WebSocket fan-out) and survive; Telegram removes its own."""
    _session_sinks.clear()


# ---------------------------------------------------------------- operator / watch set


def is_operator_session(entity: Entity | None) -> bool:
    if entity is None:
        return False
    jsonld = entity.jsonld or {}
    return jsonld.get("role") == OPERATOR_ROLE or jsonld.get("telegram_user_id") is not None


def is_subagent_session(entity: Entity | None) -> bool:
    if entity is None:
        return False
    jsonld = entity.jsonld or {}
    return jsonld.get("role") == "subagent" or bool(jsonld.get("parent_session"))


def parse_session_ref(raw: Any) -> UUID | None:
    """UUID from a session IRI, a bare UUID string, or None."""
    text = str(raw or "").strip().removeprefix(SESSION_IRI_PREFIX)
    try:
        return UUID(text)
    except (TypeError, ValueError):
        return None


def clear_legacy_attach(jsonld: dict[str, Any]) -> bool:
    """Drop leftover ``attached_session`` from the old cursor model. True if changed."""
    changed = False
    if ATTACHED_KEY in jsonld:
        jsonld.pop(ATTACHED_KEY, None)
        changed = True
    if ATTACHED_AT_KEY in jsonld:
        jsonld.pop(ATTACHED_AT_KEY, None)
        changed = True
    return changed


def prompted_ids(operator: Entity | None) -> list[UUID]:
    if operator is None:
        return []
    raw = (operator.jsonld or {}).get(PROMPTED_KEY) or []
    if not isinstance(raw, list):
        raw = [raw]
    out: list[UUID] = []
    seen: set[UUID] = set()
    for item in raw:
        uid = parse_session_ref(item)
        if uid is None or uid in seen:
            continue
        seen.add(uid)
        out.append(uid)
    return out


def last_prompted_id(operator: Entity | None) -> UUID | None:
    if operator is None:
        return None
    return parse_session_ref((operator.jsonld or {}).get(LAST_PROMPTED_KEY))


def _prompted_iris(ids: list[UUID]) -> list[str]:
    return [session_at_id(uid) for uid in ids]


async def mark_prompted(store: Store, operator: Entity, target_id: UUID) -> None:
    """Record that ``operator`` prompted ``target_id`` (watch until terminal turn_done)."""
    ids = [uid for uid in prompted_ids(operator) if uid != target_id]
    ids.append(target_id)
    operator.jsonld[PROMPTED_KEY] = _prompted_iris(ids)
    operator.jsonld[LAST_PROMPTED_KEY] = session_at_id(target_id)
    await store.put_entity(operator)


async def unwatch_prompted(store: Store, operator: Entity, target_id: UUID) -> None:
    ids = [uid for uid in prompted_ids(operator) if uid != target_id]
    if ids:
        operator.jsonld[PROMPTED_KEY] = _prompted_iris(ids)
    else:
        operator.jsonld.pop(PROMPTED_KEY, None)
    await store.put_entity(operator)


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
    """Sessions a channel may prompt: no subagents, operators, deleted, or ``exclude``."""
    skip = set(exclude)
    out: list[Entity] = []
    for ent in await store.list_entities(SESSION_TYPE):
        if ent.id in skip or is_subagent_session(ent) or is_operator_session(ent):
            continue
        if is_deleted_session(ent):
            continue
        out.append(ent)
    return out


def _normalize_workspace_ref(raw: str) -> str:
    text = str(raw or "").strip() or "workspace:default"
    if ":" not in text and not text.startswith(("/", "\\")):
        return f"workspace:{text}"
    return text


async def create_candidate_session(
    store: Store,
    *,
    title: str,
    workspace_uri: str = "workspace:default",
    channel: str = "web",
) -> Entity:
    """A new desk chat the operator can prompt. Raises RouterError on a bad workspace URI."""
    from orbweaver.agent import normalize_channel
    from orbweaver.uris import WorkspaceURIError, validate_workspace_uri
    from orbweaver.workspace import WORKSPACE_KIND_LOCAL

    try:
        uri = validate_workspace_uri(_normalize_workspace_ref(workspace_uri))
    except WorkspaceURIError as e:
        raise RouterError(str(e)) from e
    ch = normalize_channel(channel) or "web"
    heading = (title or "New chat").strip()[:80] or "New chat"
    uid = new_uuid()
    ent = Entity(
        id=uid,
        at_id=session_at_id(uid),
        at_type=SESSION_TYPE,
        jsonld={
            "@id": session_at_id(uid),
            "@type": SESSION_TYPE,
            "workspace_uri": uri,
            "workspace_kind": WORKSPACE_KIND_LOCAL,
            "title": heading,
            "channel": ch,
            "status": "active",
            "created_at": datetime.now(UTC).isoformat(),
        },
    )
    await store.put_entity(ent)
    return ent


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


def format_session_list(
    rows: list[dict[str, Any]], watching: UUID | Iterable[UUID] | None = None, limit: int = 15
) -> str:
    if not rows:
        return "No other sessions yet."
    watched: set[str] = set()
    if watching is not None:
        if isinstance(watching, UUID):
            watched = {str(watching)}
        else:
            watched = {str(uid) for uid in watching}
    lines = []
    for row in rows[:limit]:
        marks = []
        if row["id"] in watched:
            marks.append("watching")
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
        "name": "PromptSession",
        "description": (
            "Send an instruction to another Orbweaver session (web, VS Code). The target keeps "
            "its own history, workspace, model, and tools; this chat stays the operator. Use "
            "when the user wants to continue work they started at their desk. If a turn is "
            "already running the text is injected; if the session is waiting for AskUser the "
            "text is the answer; otherwise a new turn starts. Results are reported back here "
            "(the target is watched until that turn finishes). To be notified when a session "
            "finishes without sending an instruction, call WatchSession instead. "
            "If several sessions could match, ask which one first. If none is the right place, "
            "call CreateSession instead of doing the work here."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "session": {"type": "string"},
                "text": {"type": "string", "description": "Instruction or answer to send"},
            },
            "required": ["session", "text"],
        },
    },
    {
        "name": "CreateSession",
        "description": (
            "Start a new Orbweaver chat when none of the existing sessions is the right place "
            "for this work. It appears in the web sidebar (channel web) with the given title "
            "and workspace. Pass text to run a first instruction immediately (same as "
            "PromptSession). Prefer PromptSession when a matching chat already exists."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Short title for the new chat"},
                "text": {
                    "type": "string",
                    "description": "Optional first instruction to run on the new session",
                },
                "workspace": {
                    "type": "string",
                    "description": (
                        "Workspace URI (workspace:name or file:./rel). A bare name becomes "
                        "workspace:<name>. Defaults to workspace:default."
                    ),
                },
            },
            "required": [],
        },
    },
    {
        "name": "StopSession",
        "description": (
            "Cancel the running turn on another session. Accepts a session id, id prefix, "
            "or title fragment."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"session": {"type": "string"}},
            "required": ["session"],
        },
    },
    {
        "name": "WatchSession",
        "description": (
            "Subscribe this operator chat to another session's next turn completion. "
            "When that turn finishes the user gets a tagged Telegram report (session id, "
            "status, short result) without starting a turn here. Use this when the user "
            "asks to watch sessions or be notified when they finish. Do not ScheduleTask "
            "a recurring poll to check progress. Accepts a session id, id prefix, or "
            "title fragment. Call once per session."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"session": {"type": "string"}},
            "required": ["session"],
        },
    },
    {
        "name": "UnwatchSession",
        "description": (
            "Stop reporting the next turn completion for a session previously passed to "
            "WatchSession or PromptSession. Accepts a session id, id prefix, or title fragment."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"session": {"type": "string"}},
            "required": ["session"],
        },
    },
]
OPERATOR_TOOLS = frozenset(t["name"] for t in OPERATOR_TOOL_SPEC)

OPERATOR_SYSTEM_EXTRA = (
    "You are the operator for this Telegram chat: the user's always-on dispatcher from "
    "their phone, not a cursor inside another conversation. Besides this workspace you "
    "can see every other Orbweaver session (web, VS Code, cron). When the user asks what "
    "they were working on, call ListSessions, then SessionDigest on the likely match, and "
    "summarize. To continue work that belongs in another session, call PromptSession with "
    "a clear instruction; that session keeps its own history, workspace, and tools, and "
    "the result is reported back here. If the user asks you to watch sessions or to be "
    "notified when their turns finish, call WatchSession for each one. Completions arrive "
    "automatically as tagged Telegram reports (no further turn on this chat). Never "
    "ScheduleTask a recurring minute — or any poll — to check another session's progress: "
    "that starts a full turn on this operator session and blocks incoming Telegram "
    "messages. If none of the existing sessions is the right place, call CreateSession "
    "with a title (and text to start work); the new chat shows up in the web sidebar. "
    "Replying to a tagged report on Telegram injects into that session without asking "
    "you. If more than one session could match, ask which. Do not redo long-running work "
    "that belongs to another session; prompt, watch, or create one instead. PromptSession "
    "and CreateSession already ping the user with a short ack when they start a turn."
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
        return json.dumps(
            {
                "watching": [_iri_or_none(uid) for uid in prompted_ids(operator)],
                "sessions": rows,
            }
        )
    if name == "SessionDigest":
        ref = str(inp.get("session") or "").strip()
        if not ref:
            return "error: session is required (id, id prefix, or title fragment)"
        try:
            target = await find_session(store, ref, exclude=exclude)
        except AmbiguousSessionRef as e:
            return _ambiguous_tool_error(e)
        if target is None:
            return f"error: no session matches '{ref}'"
        return json.dumps(await session_digest(store, target.id))
    if name == "PromptSession":
        from orbweaver.channels.telegram import prompt_session_from_operator

        return await prompt_session_from_operator(store, operator, inp)
    if name == "CreateSession":
        title = str(inp.get("title") or "").strip()
        text = str(inp.get("text") or "").strip()
        if not title and not text:
            return "error: title or text is required"
        if not title:
            title = text.split("\n", 1)[0].strip()[:80]
        raw_ws = str(inp.get("workspace") or inp.get("workspace_uri") or "workspace:default")
        try:
            target = await create_candidate_session(
                store, title=title, workspace_uri=raw_ws
            )
        except RouterError as e:
            return f"error: {e}"
        created: dict[str, Any] = {
            "created": str(target.id),
            "title": str((target.jsonld or {}).get("title") or title),
            "workspace_uri": str((target.jsonld or {}).get("workspace_uri") or ""),
            "channel": str((target.jsonld or {}).get("channel") or "web"),
            "short_id": str(target.id)[:8],
        }
        if text:
            from orbweaver.channels.telegram import prompt_session_from_operator

            prompted = await prompt_session_from_operator(
                store, operator, {"session": str(target.id), "text": text}
            )
            try:
                created["prompt"] = json.loads(prompted)
            except json.JSONDecodeError:
                created["prompt"] = prompted
        else:
            created["note"] = (
                "Idle chat created. PromptSession to start work, or the user can open it "
                "in the web UI."
            )
        return json.dumps(created)
    if name == "StopSession":
        from orbweaver.channels.telegram import stop_session_from_operator

        return await stop_session_from_operator(store, operator, inp)
    if name == "WatchSession":
        from orbweaver.channels.telegram import watch_session_from_operator

        return await watch_session_from_operator(store, operator, inp)
    if name == "UnwatchSession":
        from orbweaver.channels.telegram import unwatch_session_from_operator

        return await unwatch_session_from_operator(store, operator, inp)
    return f"unknown operator tool {name}"


def _ambiguous_tool_error(e: AmbiguousSessionRef) -> str:
    return json.dumps(
        {
            "error": str(e),
            "matches": [
                {"id": str(m.id), "title": str((m.jsonld or {}).get("title") or "")}
                for m in e.matches
            ],
        }
    )


def _iri_or_none(uid: UUID | None) -> str | None:
    return session_at_id(uid) if uid is not None else None
