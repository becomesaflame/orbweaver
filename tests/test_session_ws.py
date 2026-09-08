import asyncio
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import quote
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from orbweaver.agent import TurnCancelled
from orbweaver.app import app, reset_ws_subscribers_for_tests
from orbweaver.auth import mint_token
from orbweaver.store import reset_store_for_tests


@pytest.fixture(autouse=True)
def _store():
    reset_store_for_tests()
    reset_ws_subscribers_for_tests()


def _token() -> str:
    return mint_token("t")


def _auth() -> dict[str, str]:
    return {"authorization": f"Bearer {_token()}"}


def _open_session(client: TestClient, **extra) -> str:
    body = {"workspace_uri": "workspace:default", "workspace_kind": "local", **extra}
    sess = client.post("/v1/sessions", json=body, headers=_auth())
    assert sess.status_code == 200, sess.text
    return sess.json()["id"]


def _ws_url(sid: str, token: str | None = None) -> str:
    url = f"/v1/sessions/{sid}/ws"
    if token is not None:
        url += f"?token={quote(token)}"
    return url


def _collect_until_done(ws) -> list[dict]:
    msgs = []
    while True:
        msg = ws.receive_json()
        if msg.get("kind") in {"subscribed", "pong", "cancelling", "idle"}:
            continue
        msgs.append(msg)
        if msg.get("kind") in {"turn_done", "error"}:
            return msgs


async def _streaming_slow_turn(
    store,
    session_id,
    user_text,
    ws,
    workspace_kind="local",
    emit=None,
    cancel=None,
    turn_state=None,
    resume=False,
    **_kwargs,
):
    produced = []
    if not resume:
        ev = await store.append_event(session_id, "user", {"text": user_text})
        if turn_state is not None:
            turn_state.user_seq = ev.seq
        if emit:
            emit({"kind": ev.kind, "payload": ev.payload, "id": str(ev.id), "seq": ev.seq})
    mid = await store.append_event(session_id, "assistant", {"text": "partial"})
    produced.append(mid)
    if emit:
        emit({"kind": mid.kind, "payload": mid.payload, "id": str(mid.id), "seq": mid.seq})
    while True:
        if cancel is not None and cancel.is_set():
            raise TurnCancelled(produced)
        await asyncio.sleep(0.02)


def test_ws_rejects_missing_token():
    with (
        TestClient(app) as client,
        pytest.raises(WebSocketDisconnect) as exc,
        client.websocket_connect(_ws_url(str(uuid4()))) as ws,
    ):
        ws.receive_text()
    assert exc.value.code == 4401


def test_ws_rejects_invalid_token():
    with (
        TestClient(app) as client,
        pytest.raises(WebSocketDisconnect) as exc,
        client.websocket_connect(_ws_url(str(uuid4()), "not-a-jwt")) as ws,
    ):
        ws.receive_text()
    assert exc.value.code == 4401


def test_ws_rejects_unknown_session():
    with (
        TestClient(app) as client,
        pytest.raises(WebSocketDisconnect) as exc,
        client.websocket_connect(_ws_url(str(uuid4()), _token())) as ws,
    ):
        ws.receive_text()
    assert exc.value.code == 4404


def test_ws_event_order(tmp_path, monkeypatch):
    monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
    from orbweaver.config import settings

    monkeypatch.setattr(settings, "workspace_root", str(tmp_path))
    monkeypatch.setattr(settings, "anthropic_api_key", "")
    token = _token()
    with TestClient(app) as client:
        sid = _open_session(client)
        with client.websocket_connect(_ws_url(sid, token)) as ws:
            assert ws.receive_json()["kind"] == "subscribed"
            ws.send_json({"text": "hello stream"})
            msgs = _collect_until_done(ws)
    kinds = [m["kind"] for m in msgs]
    assert kinds[0] == "user"
    assert kinds[-1] == "turn_done"
    assert "assistant" in kinds
    assert kinds.index("assistant") < kinds.index("turn_done")
    assert msgs[-1]["status"] == "ok"
    assert msgs[-1]["user_seq"] == msgs[0]["seq"]
    assert msgs[0]["payload"]["text"] == "hello stream"


def test_ws_cancel_stops_streaming(tmp_path, monkeypatch):
    monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
    from orbweaver.config import settings

    monkeypatch.setattr(settings, "workspace_root", str(tmp_path))
    monkeypatch.setattr("orbweaver.app.agent_turn", _streaming_slow_turn)
    token = _token()
    headers = _auth()
    with TestClient(app) as client:
        sid = _open_session(client)
        with client.websocket_connect(_ws_url(sid, token)) as ws:
            assert ws.receive_json()["kind"] == "subscribed"
            ws.send_json({"text": "keep this"})
            first = ws.receive_json()
            second = ws.receive_json()
            assert first["kind"] == "user"
            assert second["kind"] == "assistant"
            cancelled = client.post(
                f"/v1/sessions/{sid}/turns/cancel",
                json={"discard": False},
                headers=headers,
            )
            assert cancelled.status_code == 200, cancelled.text
            rest = _collect_until_done(ws)
    kinds = [m["kind"] for m in [first, second, *rest]]
    assert kinds[-1] == "turn_done"
    assert rest[-1]["status"] == "stopped"
    assert "turn_interrupted" in kinds
    assert kinds.index("turn_interrupted") < kinds.index("turn_done")
    assert kinds.count("turn_done") == 1
    assert "assistant" not in [m["kind"] for m in rest]


def test_ws_inject_during_stream(tmp_path, monkeypatch):
    monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
    from orbweaver.config import settings

    monkeypatch.setattr(settings, "workspace_root", str(tmp_path))
    monkeypatch.setattr("orbweaver.app.agent_turn", _streaming_slow_turn)
    token = _token()
    headers = _auth()
    with TestClient(app) as client:
        sid = _open_session(client)
        with client.websocket_connect(_ws_url(sid, token)) as ws:
            assert ws.receive_json()["kind"] == "subscribed"
            ws.send_json({"text": "first prompt"})
            assert ws.receive_json()["kind"] == "user"
            assert ws.receive_json()["kind"] == "assistant"
            injected = client.post(
                f"/v1/sessions/{sid}/turns/inject",
                json={"text": "also handle empty input"},
                headers=headers,
            )
            assert injected.status_code == 200, injected.text
            events = client.get(f"/v1/sessions/{sid}/events", headers=headers)
            users = [e for e in events.json()["events"] if e["kind"] == "user"]
            assert [u["payload"]["text"] for u in users] == [
                "first prompt",
                "also handle empty input",
            ]
            client.post(
                f"/v1/sessions/{sid}/turns/cancel",
                json={"discard": False},
                headers=headers,
            )
            rest = _collect_until_done(ws)
    assert rest[-1]["kind"] == "turn_done"
    assert rest[-1]["status"] == "stopped"


@pytest.fixture
def ws_session(tmp_path, monkeypatch):
    monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
    from orbweaver.config import settings

    monkeypatch.setattr(settings, "workspace_root", str(tmp_path))
    monkeypatch.setattr(settings, "anthropic_api_key", "")
    token = mint_token("t")
    headers = {"authorization": f"Bearer {token}"}
    with TestClient(app) as client:
        sess = client.post(
            "/v1/sessions",
            json={"workspace_uri": "workspace:default", "channel": "vscode", "title": "vscode"},
            headers=headers,
        )
        assert sess.status_code == 200, sess.text
        yield client, token, headers, sess.json()["id"]


def test_ws_streams_turn_events(ws_session):
    client, token, _headers, sid = ws_session
    with client.websocket_connect(f"/v1/sessions/{sid}/ws?token={token}") as ws:
        hello = ws.receive_json()
        assert hello["kind"] == "subscribed"
        assert hello["session_id"] == sid
        ws.send_json({"text": "hello from vscode"})
        kinds = []
        last = None
        while True:
            msg = ws.receive_json()
            kinds.append(msg["kind"])
            last = msg
            if msg["kind"] == "turn_done":
                break
        assert "assistant" in kinds
        assert last["status"] == "ok"


def test_ws_broadcasts_http_turn(ws_session):
    client, token, headers, sid = ws_session
    with client.websocket_connect(f"/v1/sessions/{sid}/ws?token={token}") as ws:
        assert ws.receive_json()["kind"] == "subscribed"

        def post_turn():
            return client.post(
                f"/v1/sessions/{sid}/turns",
                json={"text": "stream me"},
                headers=headers,
            )

        with ThreadPoolExecutor(max_workers=1) as pool:
            fut = pool.submit(post_turn)
            kinds = []
            while True:
                msg = ws.receive_json()
                kinds.append(msg["kind"])
                if msg["kind"] == "turn_done":
                    break
            posted = fut.result()
        assert posted.status_code == 200, posted.text
        assert "assistant" in kinds
        assert posted.json()["status"] == "ok"
