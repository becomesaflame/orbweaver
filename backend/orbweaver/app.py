from __future__ import annotations

import asyncio
import ipaddress
import json
import logging
import os
from collections import defaultdict
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import UUID

from fastapi import (
    Depends,
    FastAPI,
    File,
    HTTPException,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from orbweaver import __version__
from orbweaver.agent import (
    APPROVAL_DECISIONS,
    APPROVAL_SCOPES,
    TurnCancelled,
    agent_turn,
    normalize_channel,
    pending_ask_user,
    resolve_approval,
)
from orbweaver.auth import mint_token, require_user
from orbweaver.config import settings
from orbweaver.memory import expand_chunk_graph, remember, rewrite_search_query
from orbweaver.ratelimit import FileRateLimiter, get_rate_limiter
from orbweaver.store import (
    SESSION_TYPE,
    Entity,
    Job,
    PinBudgetError,
    get_store,
    new_uuid,
    session_at_id,
)
from orbweaver.subagent import is_subagent_session
from orbweaver.uris import (
    WorkspaceURIError,
    list_workspace_dirs,
    mkdir_workspace,
    validate_workspace_uri,
)
from orbweaver.workspace import bind_workspace, normalize_workspace_kind

log = logging.getLogger(__name__)


def _web_dir() -> Path:
    env = os.environ.get("ORBWEAVER_WEB_DIR")
    if env:
        return Path(env)
    here = Path(__file__).resolve()
    for cand in (here.parents[2] / "web", here.parents[1] / "web"):
        if cand.is_dir():
            return cand
    return here.parents[2] / "web"


WEB_DIR = _web_dir()


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    store = get_store()
    connect = getattr(store, "connect", None)
    if connect:
        await connect()
    testing = bool(os.environ.get("PYTEST_VERSION"))
    cron_on = os.environ.get("ORBWEAVER_CRON", "1").lower() not in {"0", "false", "off"}
    if cron_on and not testing:
        from orbweaver.channels.cron import start_cron

        start_cron()
    if settings.telegram_bot_token and not testing:
        from orbweaver.channels.telegram import start_telegram

        asyncio.create_task(start_telegram())
    yield


app = FastAPI(title="Orbweaver", lifespan=_lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


_RATE_LIMIT_WINDOW_S = 60.0
_RATE_LIMIT_MAX = 120


def _parse_ip(value: str) -> str | None:
    host = value.strip()
    if not host:
        return None
    if host.startswith("["):
        end = host.find("]")
        if end == -1:
            return None
        host = host[1:end]
    elif host.count(":") == 1:
        host, _, maybe_port = host.rpartition(":")
        if not maybe_port.isdigit():
            return None
    try:
        return str(ipaddress.ip_address(host))
    except ValueError:
        return None


def _forwarded_client_ip(request: Request) -> str | None:
    xff = request.headers.get("x-forwarded-for")
    if xff:
        for part in xff.split(","):
            parsed = _parse_ip(part)
            if parsed:
                return parsed
    xri = request.headers.get("x-real-ip")
    if xri:
        for part in xri.split(","):
            parsed = _parse_ip(part)
            if parsed:
                return parsed
    return None


def rate_limit_client_ip(request: Request) -> str:
    """Socket IP, or the real client IP when ORBWEAVER_TRUST_PROXY is on."""
    if settings.orbweaver_trust_proxy:
        forwarded = _forwarded_client_ip(request)
        if forwarded:
            return forwarded
    if request.client:
        return request.client.host
    return "anon"


def _rate_limit_key(request: Request) -> str:
    return request.headers.get("authorization") or rate_limit_client_ip(request)


@app.middleware("http")
async def rate_limit(request: Request, call_next):
    if request.url.path in {"/health", "/"}:
        return await call_next(request)
    key = _rate_limit_key(request)
    limiter = get_rate_limiter()
    if isinstance(limiter, FileRateLimiter):
        limiter.max_hits = _RATE_LIMIT_MAX
        limiter.window_seconds = _RATE_LIMIT_WINDOW_S
    if not await limiter.hit(str(key)):
        return JSONResponse({"detail": "rate limited"}, status_code=429)
    return await call_next(request)


class LoginBody(BaseModel):
    sub: str = "local-user"
    password: str = ""


class EntityBody(BaseModel):
    jsonld: dict[str, Any]
    pinned: bool = False


class SearchBody(BaseModel):
    query: str
    session_id: UUID | None = None
    k: int = 8


class RememberBody(BaseModel):
    text: str
    source: str = "api"
    pinned: bool = False


class ReflectBody(BaseModel):
    query: str
    budget: str = "low"


class PinBody(BaseModel):
    kind: str
    id: UUID
    pinned: bool = True


class ForgetBody(BaseModel):
    id: UUID


class SessionBody(BaseModel):
    workspace_uri: str
    workspace_kind: str = "local"
    title: str = "New chat"
    channel: str | None = None


class WorkspaceMkdirBody(BaseModel):
    parent: str = ""
    name: str


class SessionPatch(BaseModel):
    title: str | None = None


class TurnBody(BaseModel):
    text: str


class CancelTurnBody(BaseModel):
    discard: bool = False


class ApproveBody(BaseModel):
    tool_use_id: str
    decision: str  # allow | deny
    scope: str = "once"  # once | session


class RewindBody(BaseModel):
    from_seq: int


class JobBody(BaseModel):
    due_at: datetime
    message: str
    session_id: UUID | None = None
    recurrence: str | None = None


class CorrectionBody(BaseModel):
    text: str
    path: str | None = None
    accepted: bool | None = None


def _user(request: Request) -> dict:
    return require_user(request)


@dataclass
class RunningTurn:
    cancel: asyncio.Event
    inject: asyncio.Event = field(default_factory=asyncio.Event)
    discard: bool = False
    user_seq: int = 0


_running_turns: dict[UUID, RunningTurn] = {}
_ws_subscribers: dict[UUID, set[WebSocket]] = defaultdict(set)
_GENERIC_TITLES = {"", "web", "session", "New chat", "vscode"}


def reset_ws_subscribers_for_tests() -> None:
    _ws_subscribers.clear()


def _stored_channel(jsonld: dict[str, Any]) -> str:
    return normalize_channel(str(jsonld.get("channel") or ""))


def _broadcast(session_id: UUID, msg: dict[str, Any]) -> None:
    sockets = list(_ws_subscribers.get(session_id) or ())
    if not sockets:
        return
    loop = asyncio.get_running_loop()

    async def send_one(ws: WebSocket) -> None:
        try:
            await ws.send_json(msg)
        except Exception:
            _ws_subscribers[session_id].discard(ws)

    for ws in sockets:
        loop.create_task(send_one(ws))


def _event_dict(e) -> dict[str, Any]:
    return {"id": str(e.id), "seq": e.seq, "kind": e.kind, "payload": e.payload}


async def _revert_autotitle_if_needed(store, sess: Entity, discarded_text: str) -> None:
    preview = discarded_text.strip().split("\n", 1)[0][:80]
    title = str(sess.jsonld.get("title") or "").strip()
    if title != preview:
        return
    events = await store.list_events(sess.id)
    first = next((e for e in events if e.kind == "user"), None)
    if first:
        line = str(first.payload.get("text") or "").strip().split("\n", 1)[0][:80]
        sess.jsonld["title"] = line or "New chat"
    else:
        sess.jsonld["title"] = "New chat"
    await store.put_entity(sess)


async def _finish_cancelled_turn(
    store,
    sess: Entity,
    state: RunningTurn,
    produced: list,
    user_text: str,
) -> dict[str, Any]:
    if state.discard:
        if state.user_seq:
            await store.truncate_events(sess.id, state.user_seq)
            await _revert_autotitle_if_needed(store, sess, user_text)
        return {"events": [], "status": "discarded", "user_seq": state.user_seq}
    marker = await store.append_event(sess.id, "turn_interrupted", {"reason": "stop"})
    produced = list(produced) + [marker]
    _broadcast(sess.id, _event_dict(marker))
    return {
        "events": [_event_dict(e) for e in produced],
        "status": "stopped",
        "user_seq": state.user_seq,
    }



def _preview_text(events: list) -> str:
    for ev in events:
        if ev.kind == "user":
            return str(ev.payload.get("text") or "").strip()
    return ""


def _display_title(jsonld: dict[str, Any], preview: str = "") -> str:
    title = str(jsonld.get("title") or "").strip()
    if title in _GENERIC_TITLES:
        line = (preview or "").split("\n", 1)[0].strip()
        return line[:80] or "New chat"
    return title or "New chat"


async def _maybe_autotitle(store, sess: Entity, user_text: str) -> None:
    title = str(sess.jsonld.get("title") or "").strip()
    if title not in _GENERIC_TITLES:
        return
    line = user_text.strip().split("\n", 1)[0][:80]
    if not line:
        return
    sess.jsonld["title"] = line
    await store.put_entity(sess)


@app.get("/health")
async def health() -> dict[str, Any]:
    from orbweaver.hindsight import enabled as hindsight_on

    return {
        "status": "ok",
        "version": __version__,
        "hindsight": hindsight_on(),
    }


@app.post("/v1/auth/token", include_in_schema=settings.orbweaver_allow_http_mint)
async def token(body: LoginBody) -> dict[str, str]:
    if not settings.orbweaver_allow_http_mint:
        raise HTTPException(
            status_code=404,
            detail="HTTP mint is disabled; run `python -m orbweaver.cli mint` on the gateway host",
        )
    return {"token": mint_token(body.sub)}


@app.put("/memory/entities")
async def put_entity(body: EntityBody, _u: dict = Depends(_user)) -> dict[str, str]:
    store = get_store()
    jsonld = dict(body.jsonld)
    at_id = jsonld.get("@id") or jsonld.get("at_id")
    at_type = jsonld.get("@type") or jsonld.get("at_type") or "Document"
    uid = UUID(str(jsonld.get("id") or new_uuid()))
    if not at_id:
        at_id = f"urn:orbweaver:entity:{uid}"
        jsonld["@id"] = at_id
    jsonld["@type"] = at_type
    if at_type == SESSION_TYPE and jsonld.get("workspace_uri"):
        try:
            jsonld["workspace_uri"] = validate_workspace_uri(str(jsonld["workspace_uri"]))
        except WorkspaceURIError as e:
            raise HTTPException(400, str(e)) from e
    ent = Entity(id=uid, at_id=str(at_id), at_type=str(at_type), jsonld=jsonld, pinned=body.pinned)
    try:
        if body.pinned:
            import json as _j

            from orbweaver.store import ensure_pin_budget

            await ensure_pin_budget(store, extra_text=_j.dumps(jsonld))
        await store.put_entity(ent)
    except PinBudgetError as e:
        raise HTTPException(400, str(e)) from e
    except WorkspaceURIError as e:
        raise HTTPException(400, str(e)) from e
    return {"id": str(ent.id), "at_id": ent.at_id}


@app.get("/memory/graph")
async def graph(id: str, depth: int = 1, _u: dict = Depends(_user)) -> dict[str, Any]:
    triples = await get_store().graph(id, depth=depth)
    return {"triples": [{"s": s, "p": p, "o": o} for s, p, o in triples]}


@app.post("/memory/search")
async def search(body: SearchBody, _u: dict = Depends(_user)) -> dict[str, Any]:
    from orbweaver.hindsight import enabled as hindsight_on
    from orbweaver.hindsight import format_recall
    from orbweaver.hindsight import recall as hindsight_recall

    store = get_store()
    q = body.query
    if body.session_id:
        events = await store.list_events(body.session_id)
        q = rewrite_search_query(events, body.query)
    if hindsight_on():
        try:
            payload = format_recall(await hindsight_recall(q))
            return {
                "query_used": q,
                "hits": payload["hits"],
                "graph": payload["graph"],
                "source": "hindsight",
            }
        except Exception as e:
            log.warning("hindsight recall failed, using native: %s", e)
    hits = await store.search_chunks(q, k=body.k)
    graph_hits = await expand_chunk_graph(store, hits)
    return {
        "query_used": q,
        "hits": [{"id": str(c.id), "text": c.text, "score": score} for c, score in hits],
        "graph": graph_hits,
    }


@app.post("/memory/remember")
async def remember_api(body: RememberBody, _u: dict = Depends(_user)) -> dict[str, Any]:
    from orbweaver.hindsight import enabled as hindsight_on
    from orbweaver.hindsight import retain as hindsight_retain

    out: dict[str, Any] = {}
    if hindsight_on() and body.text.strip():
        try:
            out["hindsight"] = await hindsight_retain(
                body.text, context=body.source or "api", retain_async=not body.pinned
            )
        except Exception as e:
            if not body.pinned:
                raise HTTPException(502, f"hindsight retain failed: {e}") from e
    if body.pinned or not hindsight_on():
        try:
            chunk = await remember(
                get_store(), body.text, source=body.source, pinned=body.pinned
            )
        except PinBudgetError as e:
            raise HTTPException(400, str(e)) from e
        out["id"] = str(chunk.id)
    elif "hindsight" in out:
        out["id"] = "hindsight"
    return out


@app.post("/memory/reflect")
async def reflect_api(body: ReflectBody, _u: dict = Depends(_user)) -> dict[str, Any]:
    from orbweaver.hindsight import enabled as hindsight_on
    from orbweaver.hindsight import format_reflect
    from orbweaver.hindsight import reflect as hindsight_reflect

    if not hindsight_on():
        raise HTTPException(503, "Hindsight is not configured (set HINDSIGHT_API_URL)")
    budget = body.budget if body.budget in {"low", "mid", "high"} else "low"
    try:
        return format_reflect(await hindsight_reflect(body.query, budget=budget))
    except Exception as e:
        raise HTTPException(502, str(e)) from e


@app.post("/memory/pin")
async def pin_api(body: PinBody, _u: dict = Depends(_user)) -> dict[str, str]:
    try:
        await get_store().set_pinned(body.kind, body.id, body.pinned)
    except PinBudgetError as e:
        raise HTTPException(400, str(e)) from e
    except KeyError as e:
        raise HTTPException(404, "not found") from e
    return {"status": "ok"}


@app.post("/memory/forget")
async def forget_api(body: ForgetBody, _u: dict = Depends(_user)) -> dict[str, str]:
    await get_store().forget_chunk(body.id)
    return {"status": "forgotten"}


@app.get("/v1/workspaces")
async def browse_workspaces(rel: str = "", _u: dict = Depends(_user)) -> dict[str, Any]:
    try:
        return list_workspace_dirs(settings.workspace_root, rel)
    except FileNotFoundError as e:
        raise HTTPException(404, "folder not found") from e
    except WorkspaceURIError as e:
        raise HTTPException(400, str(e)) from e


@app.post("/v1/workspaces")
async def mkdir_workspace_api(body: WorkspaceMkdirBody, _u: dict = Depends(_user)) -> dict[str, Any]:
    try:
        return mkdir_workspace(settings.workspace_root, body.parent, body.name)
    except WorkspaceURIError as e:
        raise HTTPException(400, str(e)) from e


@app.post("/v1/sessions")
async def create_session(body: SessionBody, _u: dict = Depends(_user)) -> dict[str, Any]:
    try:
        uri = validate_workspace_uri(body.workspace_uri)
    except WorkspaceURIError as e:
        raise HTTPException(400, str(e)) from e
    channel = normalize_channel(body.channel)
    uid = new_uuid()
    ent = Entity(
        id=uid,
        at_id=session_at_id(uid),
        at_type=SESSION_TYPE,
        jsonld={
            "@id": session_at_id(uid),
            "@type": SESSION_TYPE,
            "workspace_uri": uri,
            "workspace_kind": normalize_workspace_kind(body.workspace_kind),
            "title": body.title or "New chat",
            "status": "active",
            "created_at": datetime.now(UTC).isoformat(),
        },
    )
    if channel:
        ent.jsonld["channel"] = channel
    await get_store().put_entity(ent)
    return {
        "id": str(uid),
        "at_id": ent.at_id,
        "workspace_uri": uri,
        "title": ent.jsonld["title"],
        "channel": channel,
    }


@app.get("/v1/sessions")
async def list_sessions(_u: dict = Depends(_user)) -> dict[str, Any]:
    store = get_store()
    out: list[dict[str, Any]] = []
    for ent in await store.list_entities(SESSION_TYPE):
        if is_subagent_session(ent):
            continue
        events = await store.list_events(ent.id)
        preview = _preview_text(events)
        last_at = events[-1].created_at.isoformat() if events else str(ent.jsonld.get("created_at") or "")
        out.append(
            {
                "id": str(ent.id),
                "title": _display_title(ent.jsonld, preview),
                "workspace_uri": str(ent.jsonld.get("workspace_uri") or "workspace:default"),
                "workspace_kind": normalize_workspace_kind(ent.jsonld.get("workspace_kind")),
                "status": str(ent.jsonld.get("status") or "active"),
                "channel": _stored_channel(ent.jsonld),
                "created_at": str(ent.jsonld.get("created_at") or ""),
                "last_event_at": last_at,
                "event_count": len(events),
                "preview": preview[:80],
            }
        )
    out.sort(key=lambda s: s["last_event_at"] or s["created_at"] or "", reverse=True)
    return {"sessions": out}


@app.patch("/v1/sessions/{session_id}")
async def patch_session(session_id: UUID, body: SessionPatch, _u: dict = Depends(_user)) -> dict[str, Any]:
    store = get_store()
    sess = await store.get_entity(session_id)
    if not sess or sess.at_type != SESSION_TYPE:
        raise HTTPException(404, "session not found")
    if body.title is not None:
        title = body.title.strip()[:80] or "New chat"
        sess.jsonld["title"] = title
        await store.put_entity(sess)
    return {"id": str(sess.id), "title": _display_title(sess.jsonld)}


@app.get("/v1/sessions/{session_id}/events")
async def list_events(session_id: UUID, _u: dict = Depends(_user)) -> dict[str, Any]:
    events = await get_store().list_events(session_id)
    return {
        "events": [
            {
                "id": str(e.id),
                "seq": e.seq,
                "kind": e.kind,
                "payload": e.payload,
            }
            for e in events
        ]
    }


async def _run_turn(
    store,
    sess,
    session_id: UUID,
    user_text: str,
    *,
    resume: bool = False,
) -> dict[str, Any]:
    if session_id in _running_turns:
        raise HTTPException(409, "turn already running")
    if not resume:
        await _maybe_autotitle(store, sess, user_text)
    ws, kind, changed = bind_workspace(sess.jsonld, settings.workspace_root)
    if changed:
        await store.put_entity(sess)
    state = RunningTurn(cancel=asyncio.Event())
    _running_turns[session_id] = state
    result: dict[str, Any] | None = None
    try:
        events = await agent_turn(
            store,
            session_id,
            user_text,
            ws,
            workspace_kind=kind,
            emit=lambda msg: _broadcast(session_id, msg),
            cancel=state.cancel,
            turn_state=state,
            resume=resume,
            interactive=True,
        )
        if state.cancel.is_set():
            result = await _finish_cancelled_turn(store, sess, state, events, user_text)
        else:
            stored = await store.list_events(session_id)
            pending = pending_ask_user(stored)
            status = "waiting_ask" if pending else "ok"
            question = ""
            if pending:
                question = str((pending.payload or {}).get("input", {}).get("question") or "")
                if not question:
                    for ev in reversed(stored):
                        if ev.kind == "ask_user":
                            question = str((ev.payload or {}).get("question") or "")
                            break
            result = {
                "events": [_event_dict(e) for e in events],
                "status": status,
                "user_seq": state.user_seq,
            }
            if question:
                result["question"] = question
        return result
    except TurnCancelled as e:
        result = await _finish_cancelled_turn(store, sess, state, e.produced, user_text)
        return result
    except Exception as e:
        import anthropic

        from orbweaver.compact import ContextFullError

        if isinstance(e, ContextFullError):
            raise HTTPException(status_code=502, detail=e.message) from e
        if isinstance(e, anthropic.APIStatusError):
            raise HTTPException(status_code=502, detail=e.message) from e
        raise
    finally:
        current = _running_turns.get(session_id)
        if current is state:
            _running_turns.pop(session_id, None)
        if result is not None:
            _broadcast(
                session_id,
                {
                    "kind": "turn_done",
                    "status": result.get("status"),
                    "user_seq": result.get("user_seq"),
                },
            )


@app.post("/v1/sessions/{session_id}/turns")
async def turn(session_id: UUID, body: TurnBody, _u: dict = Depends(_user)) -> dict[str, Any]:
    store = get_store()
    sess = await store.get_entity(session_id)
    if not sess:
        raise HTTPException(404, "session not found")
    return await _run_turn(store, sess, session_id, body.text)


@app.post("/v1/sessions/{session_id}/turns/inject")
async def inject_turn(
    session_id: UUID, body: TurnBody, _u: dict = Depends(_user)
) -> dict[str, Any]:
    text = body.text.strip()
    if not text:
        raise HTTPException(400, "text is required")
    state = _running_turns.get(session_id)
    if not state:
        raise HTTPException(409, "no turn is running")
    store = get_store()
    ev = await store.append_event(session_id, "user", {"text": text, "injected": True})
    state.inject.set()
    _broadcast(session_id, _event_dict(ev))
    return {"status": "injected", "seq": ev.seq, "event": _event_dict(ev)}


@app.post("/v1/sessions/{session_id}/turns/continue")
async def continue_turn(session_id: UUID, _u: dict = Depends(_user)) -> dict[str, Any]:
    store = get_store()
    sess = await store.get_entity(session_id)
    if not sess:
        raise HTTPException(404, "session not found")
    events = await store.list_events(session_id)
    if not events:
        raise HTTPException(400, "nothing to continue")
    if pending_ask_user(events):
        raise HTTPException(400, "answer the pending AskUser question first")
    return await _run_turn(store, sess, session_id, "", resume=True)


@app.post("/v1/sessions/{session_id}/turns/cancel")
async def cancel_turn(
    session_id: UUID, body: CancelTurnBody, _u: dict = Depends(_user)
) -> dict[str, Any]:
    state = _running_turns.get(session_id)
    if not state:
        return {"status": "idle"}
    state.discard = body.discard
    state.cancel.set()
    return {"status": "cancelling", "discard": body.discard}


def _approve(session_id: UUID, tool_use_id: str, decision: str, scope: str) -> dict[str, Any]:
    decision = (decision or "").strip().lower()
    scope = (scope or "once").strip().lower() or "once"
    if decision not in APPROVAL_DECISIONS:
        raise HTTPException(400, f"decision must be one of {sorted(APPROVAL_DECISIONS)}")
    if scope not in APPROVAL_SCOPES:
        raise HTTPException(400, f"scope must be one of {sorted(APPROVAL_SCOPES)}")
    if not tool_use_id.strip():
        raise HTTPException(400, "tool_use_id is required")
    if not resolve_approval(session_id, tool_use_id, decision, scope):
        raise HTTPException(404, "no pending approval for that tool_use_id")
    return {
        "status": "resolved",
        "tool_use_id": tool_use_id,
        "decision": decision,
        "scope": scope,
    }


@app.post("/v1/sessions/{session_id}/turns/approve")
async def approve_turn(
    session_id: UUID, body: ApproveBody, _u: dict = Depends(_user)
) -> dict[str, Any]:
    """Answer a held `permission_request` for the running turn."""
    return _approve(session_id, body.tool_use_id, body.decision, body.scope)


@app.post("/v1/sessions/{session_id}/rewind")
async def rewind(
    session_id: UUID, body: RewindBody, _u: dict = Depends(_user)
) -> dict[str, Any]:
    if body.from_seq < 1:
        raise HTTPException(400, "from_seq must be >= 1")
    if session_id in _running_turns:
        raise HTTPException(409, "turn already running")
    store = get_store()
    sess = await store.get_entity(session_id)
    if not sess:
        raise HTTPException(404, "session not found")
    events = await store.list_events(session_id)
    discarded = next((e for e in events if e.seq == body.from_seq), None)
    await store.truncate_events(session_id, body.from_seq)
    text = ""
    if discarded and discarded.kind == "user":
        text = str(discarded.payload.get("text") or "")
        await _revert_autotitle_if_needed(store, sess, text)
    return {"status": "ok", "text": text}


@app.post("/v1/sessions/{session_id}/correction")
async def correction(session_id: UUID, body: CorrectionBody, _u: dict = Depends(_user)) -> dict[str, str]:
    await get_store().append_event(
        session_id,
        "UserCorrection",
        {"text": body.text, "path": body.path, "accepted": body.accepted},
    )
    return {"status": "ok"}


@app.websocket("/v1/sessions/{session_id}/ws")
async def session_ws(websocket: WebSocket, session_id: UUID, token: str | None = None) -> None:
    await websocket.accept()
    if not token:
        await websocket.close(code=4401)
        return
    try:
        from orbweaver.auth import decode_token

        decode_token(token)
    except HTTPException:
        await websocket.close(code=4401)
        return
    store = get_store()
    sess = await store.get_entity(session_id)
    if not sess:
        await websocket.close(code=4404)
        return
    _ws_subscribers[session_id].add(websocket)
    try:
        await websocket.send_json({"kind": "subscribed", "session_id": str(session_id)})
        while True:
            raw = await websocket.receive_text()
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if not isinstance(data, dict):
                continue
            typ = str(data.get("type") or "")
            if typ in {"subscribe", "ping"}:
                if typ == "ping":
                    await websocket.send_json({"kind": "pong"})
                continue
            if typ == "cancel":
                state = _running_turns.get(session_id)
                if state:
                    state.discard = bool(data.get("discard"))
                    state.cancel.set()
                    await websocket.send_json(
                        {"kind": "cancelling", "discard": state.discard}
                    )
                else:
                    await websocket.send_json({"kind": "idle"})
                continue
            if typ == "approve":
                # Only reachable from a socket that is not itself awaiting a turn
                # (this loop blocks on _run_turn); the web UI uses the HTTP endpoint.
                try:
                    out = _approve(
                        session_id,
                        str(data.get("tool_use_id") or ""),
                        str(data.get("decision") or ""),
                        str(data.get("scope") or "once"),
                    )
                except HTTPException as e:
                    await websocket.send_json(
                        {"kind": "approval", "status": e.status_code, "detail": e.detail}
                    )
                    continue
                await websocket.send_json({"kind": "approval", **out})
                continue
            resume = bool(data.get("resume"))
            text = str(data.get("text") or "")
            if not resume and not text.strip():
                continue
            if resume:
                existing = await store.list_events(session_id)
                if not existing:
                    await websocket.send_json(
                        {
                            "kind": "error",
                            "detail": "nothing to continue",
                            "status": 400,
                            "status_code": 400,
                        }
                    )
                    continue
            try:
                await _run_turn(store, sess, session_id, text, resume=resume)
            except HTTPException as e:
                await websocket.send_json(
                    {
                        "kind": "error",
                        "status": e.status_code,
                        "status_code": e.status_code,
                        "detail": e.detail,
                    }
                )
    except WebSocketDisconnect:
        return
    finally:
        _ws_subscribers[session_id].discard(websocket)


@app.post("/v1/stt")
async def stt(file: UploadFile = File(...), _u: dict = Depends(_user)) -> dict[str, str]:
    data = await file.read()
    try:
        from orbweaver.stt import transcribe_bytes

        text = transcribe_bytes(data, file.filename or "audio.wav")
    except Exception as e:
        raise HTTPException(501, f"STT unavailable: {e}") from e
    return {"text": text}


@app.post("/v1/jobs")
async def create_job(body: JobBody, _u: dict = Depends(_user)) -> dict[str, str]:
    job = Job(
        id=new_uuid(),
        due_at=body.due_at,
        payload={"message": body.message},
        recurrence=body.recurrence,
        session_id=body.session_id,
    )
    await get_store().put_job(job)
    return {"id": str(job.id)}


@app.get("/v1/jobs")
async def list_jobs(_u: dict = Depends(_user)) -> dict[str, Any]:
    jobs = await get_store().due_jobs(datetime.now(UTC) + timedelta(days=3650))
    return {
        "jobs": [
            {
                "id": str(j.id),
                "due_at": j.due_at.isoformat(),
                "payload": j.payload,
                "session_id": str(j.session_id) if j.session_id else None,
            }
            for j in jobs
        ]
    }


if WEB_DIR.is_dir():
    app.mount("/ui", StaticFiles(directory=str(WEB_DIR), html=True), name="ui")


@app.get("/", response_model=None)
async def root() -> FileResponse | JSONResponse:
    index = WEB_DIR / "index.html"
    if index.is_file():
        return FileResponse(index)
    return JSONResponse({"service": "orbweaver"})
