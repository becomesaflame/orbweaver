from __future__ import annotations

import asyncio
import ipaddress
import json
import logging
import os
from collections import defaultdict, deque
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
from orbweaver.agent import TurnCancelled, agent_turn, normalize_channel, pending_ask_user
from orbweaver.auth import (
    WS_BEARER_SUBPROTOCOL,
    check_jwt_secret,
    decode_token,
    mint_token,
    require_user,
    websocket_subprotocol_token,
)
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
from orbweaver.turns import RunningTurn
from orbweaver.turns import acquire as acquire_turn
from orbweaver.turns import get as get_running_turn
from orbweaver.turns import is_running as turn_is_running
from orbweaver.turns import release as release_turn
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
    # Refuse to serve with a forgeable JWT secret (issue #97). Raising here
    # makes uvicorn log "Application startup failed" and exit non-zero.
    check_jwt_secret(settings)
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


def configure_cors(target: FastAPI, origins: list[str]) -> None:
    """Allow credentialed cross-origin calls only from `origins`.

    The default (empty list) is same-origin only: the bundled web UI is served
    by this process, so browsers never need CORS for it. `*` with credentials
    is rejected by browsers and would echo any Origin, so credentials are only
    enabled for an explicit origin list.
    """
    target.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=bool(origins) and "*" not in origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )


app = FastAPI(title="Orbweaver", lifespan=_lifespan)
configure_cors(app, settings.cors_origins)


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


@dataclass(eq=False)
class _WsSubscriber:
    """One WebSocket client of a session.

    Frames are queued in ``pending`` and written by a single writer task so
    live broadcasts, replies, and replayed history never interleave. Every
    ``seq`` delivered live is recorded in ``sent_seqs`` so a later
    ``subscribe`` with ``after_seq`` does not replay it again.
    """

    ws: WebSocket
    pending: deque[dict[str, Any]] = field(default_factory=deque)
    wake: asyncio.Event = field(default_factory=asyncio.Event)
    sent_seqs: set[int] = field(default_factory=set)

    def send(self, msg: dict[str, Any]) -> None:
        self.pending.append(msg)
        self.wake.set()

    def replay(self, frames: list[dict[str, Any]], top: int) -> None:
        """Queue ``frames`` ahead of live frames still waiting to be written.

        Live frames with ``seq <= top`` are already covered by the replay and
        are dropped; anything else queued (``turn_done``, ``pong``) stays
        behind the replayed history so ordering matches the store.
        """
        kept = [
            m
            for m in self.pending
            if not (isinstance(m.get("seq"), int) and m["seq"] <= top)
        ]
        self.pending.clear()
        self.pending.extend(frames)
        self.pending.extend(kept)
        self.wake.set()


_last_turn_done: dict[UUID, dict[str, Any]] = {}
_ws_subscribers: dict[UUID, set[_WsSubscriber]] = defaultdict(set)
_detached_turns: set[asyncio.Task[Any]] = set()
_GENERIC_TITLES = {"", "web", "session", "New chat", "vscode"}


def reset_ws_subscribers_for_tests() -> None:
    _ws_subscribers.clear()
    _last_turn_done.clear()
    _detached_turns.clear()


def _stored_channel(jsonld: dict[str, Any]) -> str:
    return normalize_channel(str(jsonld.get("channel") or ""))


def _broadcast(session_id: UUID, msg: dict[str, Any]) -> None:
    for sub in list(_ws_subscribers.get(session_id) or ()):
        sub.send(msg)


async def _ws_writer(session_id: UUID, sub: _WsSubscriber) -> None:
    try:
        while True:
            while sub.pending:
                msg = sub.pending.popleft()
                seq = msg.get("seq")
                if isinstance(seq, int):
                    sub.sent_seqs.add(seq)
                await sub.ws.send_json(msg)
            sub.wake.clear()
            await sub.wake.wait()
    except Exception as e:
        # The peer is gone (or mid-close); the receive loop sees the
        # disconnect and unwinds. Nothing to send it anymore.
        log.debug("websocket writer for %s stopped: %s", session_id, e)
    finally:
        _ws_subscribers[session_id].discard(sub)


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
    state = acquire_turn(session_id, channel=_stored_channel(sess.jsonld) or "web")
    if state is None:
        raise HTTPException(409, "turn already running")
    _last_turn_done.pop(session_id, None)
    result: dict[str, Any] | None = None
    failure: str | None = None
    try:
        if not resume:
            await _maybe_autotitle(store, sess, user_text)
        ws, kind, changed = bind_workspace(sess.jsonld, settings.workspace_root)
        if changed:
            await store.put_entity(sess)
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

        failure = str(getattr(e, "message", None) or e) or type(e).__name__
        if isinstance(e, ContextFullError):
            raise HTTPException(status_code=502, detail=e.message) from e
        if isinstance(e, anthropic.APIStatusError):
            raise HTTPException(status_code=502, detail=e.message) from e
        raise
    finally:
        release_turn(session_id, state)
        if result is not None:
            done = {"status": result.get("status"), "user_seq": result.get("user_seq")}
            _last_turn_done[session_id] = done
            _broadcast(session_id, {"kind": "turn_done", **done})
        else:
            _last_turn_done[session_id] = {
                "status": "error",
                "user_seq": state.user_seq,
                "detail": failure or "turn failed",
            }


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
    state = get_running_turn(session_id)
    if not state:
        raise HTTPException(409, "no turn is running")
    ev = await inject_into_turn(get_store(), session_id, state, text)
    return {"status": "injected", "seq": ev.seq, "event": _event_dict(ev)}


async def inject_into_turn(
    store,
    session_id: UUID,
    state: RunningTurn,
    text: str,
    images: list[dict[str, str]] | None = None,
) -> Any:
    """Append a follow-up user event to a running turn and wake its LLM call."""
    payload: dict[str, Any] = {"text": text, "injected": True}
    if images:
        payload["images"] = images
    ev = await store.append_event(session_id, "user", payload)
    state.inject.set()
    _broadcast(session_id, _event_dict(ev))
    return ev


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
    state = get_running_turn(session_id)
    if not state:
        return {"status": "idle"}
    state.discard = body.discard
    state.cancel.set()
    return {"status": "cancelling", "discard": body.discard}


@app.post("/v1/sessions/{session_id}/rewind")
async def rewind(
    session_id: UUID, body: RewindBody, _u: dict = Depends(_user)
) -> dict[str, Any]:
    if body.from_seq < 1:
        raise HTTPException(400, "from_seq must be >= 1")
    if turn_is_running(session_id):
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


_WS_AUTH_TIMEOUT_S = 10.0


async def _ws_first_frame_token(websocket: WebSocket) -> str | None:
    """Token from a first-message auth frame: {"type": "auth", "token": "<jwt>"}."""
    try:
        raw = await asyncio.wait_for(websocket.receive_text(), timeout=_WS_AUTH_TIMEOUT_S)
    except TimeoutError:
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict) or str(data.get("type") or "") != "auth":
        return None
    tok = data.get("token")
    return str(tok) if tok else None


async def _ws_authenticate(websocket: WebSocket, query_token: str | None) -> bool:
    """Accept the socket and resolve the bearer token.

    Preferred: `Sec-WebSocket-Protocol: bearer, <jwt>` (echoed back as
    `bearer`). Also accepted: a first frame `{"type": "auth", "token": ...}`.
    Deprecated, removed in a future release: `?token=<jwt>` in the URL, which
    ends up in access logs and proxies.
    """
    bearer_offered, sub_token = websocket_subprotocol_token(websocket)
    await websocket.accept(subprotocol=WS_BEARER_SUBPROTOCOL if bearer_offered else None)
    token: str | None
    if bearer_offered:
        token = sub_token
    elif query_token:
        log.warning(
            "websocket auth via ?token= query parameter is deprecated and will be removed; "
            "send `Sec-WebSocket-Protocol: bearer, <jwt>` or a first frame "
            '{"type": "auth", "token": "<jwt>"} instead'
        )
        token = query_token
    else:
        token = await _ws_first_frame_token(websocket)
    if not token:
        return False
    try:
        decode_token(token)
    except HTTPException:
        return False
    return True


@app.websocket("/v1/sessions/{session_id}/ws")
async def session_ws(websocket: WebSocket, session_id: UUID, token: str | None = None) -> None:
    try:
        authed = await _ws_authenticate(websocket, token)
    except WebSocketDisconnect:
        return
    if not authed:
        await websocket.close(code=4401)
        return
    store = get_store()
    sess = await store.get_entity(session_id)
    if not sess:
        await websocket.close(code=4404)
        return
    sub = _WsSubscriber(ws=websocket)
    _ws_subscribers[session_id].add(sub)
    writer = asyncio.create_task(_ws_writer(session_id, sub))
    turn_task: asyncio.Task[Any] | None = None

    def error(status: int, detail: Any) -> None:
        sub.send({"kind": "error", "status": status, "status_code": status, "detail": detail})

    try:
        sub.send({"kind": "subscribed", "session_id": str(session_id)})
        while True:
            raw = await websocket.receive_text()
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if not isinstance(data, dict):
                continue
            typ = str(data.get("type") or "")
            if typ == "auth":
                # Token already resolved by _ws_authenticate; a late auth
                # frame (e.g. after subprotocol auth) is harmless.
                continue
            if typ == "ping":
                sub.send({"kind": "pong"})
                continue
            if typ == "subscribe":
                after_seq = data.get("after_seq")
                if isinstance(after_seq, int) and not isinstance(after_seq, bool):
                    await _ws_resubscribe(store, session_id, sub, after_seq)
                continue
            if typ == "cancel":
                state = get_running_turn(session_id)
                if state:
                    state.discard = bool(data.get("discard"))
                    state.cancel.set()
                    sub.send({"kind": "cancelling", "discard": state.discard})
                else:
                    sub.send({"kind": "idle"})
                continue
            if typ == "inject":
                text = str(data.get("text") or "").strip()
                state = get_running_turn(session_id)
                if not text:
                    error(400, "text is required")
                elif not state:
                    error(409, "no turn is running")
                else:
                    ev = await inject_into_turn(store, session_id, state, text)
                    sub.send({"kind": "injected", "seq": ev.seq})
                continue
            resume = bool(data.get("resume"))
            text = str(data.get("text") or "")
            if not resume and not text.strip():
                continue
            if resume:
                existing = await store.list_events(session_id)
                if not existing:
                    error(400, "nothing to continue")
                    continue
            if turn_task is not None and not turn_task.done():
                error(409, "turn already running")
                continue
            # Run the turn as its own task so this loop keeps reading frames:
            # cancel / ping / inject / subscribe work while the agent is busy,
            # and the turn survives this socket closing.
            turn_task = asyncio.create_task(
                _ws_turn(sub, store, sess, session_id, text, resume=resume)
            )
            _detached_turns.add(turn_task)
            turn_task.add_done_callback(_detached_turns.discard)
    except WebSocketDisconnect:
        return
    finally:
        _ws_subscribers[session_id].discard(sub)
        writer.cancel()


async def _ws_turn(
    sub: _WsSubscriber,
    store,
    sess,
    session_id: UUID,
    text: str,
    *,
    resume: bool,
) -> None:
    try:
        await _run_turn(store, sess, session_id, text, resume=resume)
    except HTTPException as e:
        sub.send(
            {
                "kind": "error",
                "status": e.status_code,
                "status_code": e.status_code,
                "detail": e.detail,
            }
        )
    except Exception as e:
        log.exception("websocket turn failed for %s", session_id)
        sub.send({"kind": "error", "status": 500, "status_code": 500, "detail": str(e)})


async def _ws_resubscribe(store, session_id: UUID, sub: _WsSubscriber, after_seq: int) -> None:
    """Replay stored events with ``seq > after_seq`` then report turn status.

    ``sub`` has been a live subscriber since the socket was accepted, so every
    event is either already sent (``sent_seqs``), waiting in ``sub.pending``
    (dropped by ``replay`` because the stored copy is newer), or in the store
    snapshot read here. Nothing after the snapshot can carry ``seq <= top``,
    so the replay attaches to the live stream with no gap and no duplicate.
    """
    events = await store.list_events(session_id)
    top = max((e.seq for e in events), default=after_seq)
    frames = [
        _event_dict(e) for e in events if e.seq > after_seq and e.seq not in sub.sent_seqs
    ]
    status: dict[str, Any] = {
        "kind": "turn_status",
        "running": turn_is_running(session_id),
        "seq": top,
        "last_turn_done": _last_turn_done.get(session_id),
    }
    frames.append(status)
    sub.replay(frames, top)


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
