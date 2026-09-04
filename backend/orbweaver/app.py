from __future__ import annotations

import asyncio
import json
import os
from collections import defaultdict
from contextlib import asynccontextmanager
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

from orbweaver.agent import agent_turn
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
from orbweaver.uris import WorkspaceURIError, validate_workspace_uri
from orbweaver.workspace import make_workspace

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
    title: str = "session"


class TurnBody(BaseModel):
    text: str


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


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/auth/token")
async def token(body: LoginBody) -> dict[str, str]:
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
            "workspace_kind": body.workspace_kind,
            "title": body.title,
            "status": "active",
        },
    )
    await get_store().put_entity(ent)
    return {"id": str(uid), "at_id": ent.at_id, "workspace_uri": uri}


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


@app.post("/v1/sessions/{session_id}/turns")
async def turn(session_id: UUID, body: TurnBody, _u: dict = Depends(_user)) -> dict[str, Any]:
    store = get_store()
    sess = await store.get_entity(session_id)
    if not sess:
        raise HTTPException(404, "session not found")
    kind = str(sess.jsonld.get("workspace_kind") or "local")
    uri = str(sess.jsonld.get("workspace_uri"))
    ws = make_workspace(kind, uri, settings.workspace_root)
    events = await agent_turn(store, session_id, body.text, ws, workspace_kind=kind)
    return {
        "events": [
            {"id": str(e.id), "kind": e.kind, "payload": e.payload} for e in events
        ]
    }


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
            kind = str(sess.jsonld.get("workspace_kind") or "local")
            uri = str(sess.jsonld.get("workspace_uri"))
            ws = make_workspace(kind, uri, settings.workspace_root)

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
