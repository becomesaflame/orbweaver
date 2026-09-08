from __future__ import annotations

import asyncio
import json
import os
from collections import defaultdict
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from time import time
from typing import Any
from uuid import UUID

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from orbweaver import __version__
from orbweaver.agent import TurnCancelled, agent_turn, pending_ask_user
from orbweaver.auth import mint_token, require_user
from orbweaver.config import settings
from orbweaver.memory import remember, rewrite_search_query
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

_hits: dict[str, list[float]] = defaultdict(list)


@app.middleware("http")
async def rate_limit(request: Request, call_next):
    if request.url.path in {"/health", "/"}:
        return await call_next(request)
    key = request.headers.get("authorization") or request.client.host if request.client else "anon"
    now = time()
    window = [t for t in _hits[key] if now - t < 60]
    if len(window) >= 120:
        return JSONResponse({"detail": "rate limited"}, status_code=429)
    window.append(now)
    _hits[key] = window
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


@dataclass
class RunningTurn:
    cancel: asyncio.Event
    inject: asyncio.Event = field(default_factory=asyncio.Event)
    discard: bool = False
    user_seq: int = 0


_running_turns: dict[UUID, RunningTurn] = {}
_GENERIC_TITLES = {"", "web", "session", "New chat"}


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
    store, sess: Entity, state: RunningTurn, produced: list, user_text: str
) -> dict[str, Any]:
    if state.discard:
        if state.user_seq:
            await store.truncate_events(sess.id, state.user_seq)
            await _revert_autotitle_if_needed(store, sess, user_text)
        return {"events": [], "status": "discarded", "user_seq": state.user_seq}
    marker = await store.append_event(sess.id, "turn_interrupted", {"reason": "stop"})
    produced = list(produced) + [marker]
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
async def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


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
            from orbweaver.store import ensure_pin_budget
            import json as _j

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
    store = get_store()
    q = body.query
    if body.session_id:
        events = await store.list_events(body.session_id)
        q = rewrite_search_query(events, body.query)
    hits = await store.search_chunks(q, k=body.k)
    graph_hits: list[dict[str, str]] = []
    seen_edges: set[tuple[str, str, str]] = set()
    for chunk, _score in hits:
        for eid in chunk.entity_ids:
            ent = await store.get_entity(eid)
            if not ent:
                continue
            for s, p, o in await store.graph(ent.at_id, depth=1):
                edge = (s, p, o)
                if edge in seen_edges:
                    continue
                seen_edges.add(edge)
                graph_hits.append({"s": s, "p": p, "o": o})
    return {
        "query_used": q,
        "hits": [{"id": str(c.id), "text": c.text, "score": score} for c, score in hits],
        "graph": graph_hits,
    }


@app.post("/memory/remember")
async def remember_api(body: RememberBody, _u: dict = Depends(_user)) -> dict[str, str]:
    try:
        chunk = await remember(get_store(), body.text, source=body.source, pinned=body.pinned)
    except PinBudgetError as e:
        raise HTTPException(400, str(e)) from e
    return {"id": str(chunk.id)}


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
    await get_store().put_entity(ent)
    return {"id": str(uid), "at_id": ent.at_id, "workspace_uri": uri, "title": ent.jsonld["title"]}


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
    store, sess, session_id: UUID, user_text: str, *, resume: bool = False
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
    try:
        events = await agent_turn(
            store,
            session_id,
            user_text,
            ws,
            workspace_kind=kind,
            cancel=state.cancel,
            turn_state=state,
            resume=resume,
            interactive=True,
        )
        if state.cancel.is_set():
            return await _finish_cancelled_turn(store, sess, state, events, user_text)
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
        out: dict[str, Any] = {
            "events": [_event_dict(e) for e in events],
            "status": status,
            "user_seq": state.user_seq,
        }
        if question:
            out["question"] = question
        return out
    except TurnCancelled as e:
        return await _finish_cancelled_turn(store, sess, state, e.produced, user_text)
    except Exception as e:
        import anthropic

        if isinstance(e, anthropic.APIStatusError):
            raise HTTPException(status_code=502, detail=e.message) from e
        raise
    finally:
        current = _running_turns.get(session_id)
        if current is state:
            _running_turns.pop(session_id, None)


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
    try:
        while True:
            raw = await websocket.receive_text()
            data = json.loads(raw)
            text = data.get("text") or ""
            if not text:
                continue
            ws, kind, changed = bind_workspace(sess.jsonld, settings.workspace_root)
            if changed:
                await store.put_entity(sess)

            def emit(msg: dict[str, Any]) -> None:
                asyncio.get_event_loop().create_task(websocket.send_json(msg))

            await agent_turn(store, session_id, text, ws, workspace_kind=kind, emit=emit)
            await websocket.send_json({"kind": "turn_done"})
    except WebSocketDisconnect:
        return


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
