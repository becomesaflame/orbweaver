from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from orbweaver.app import app, reset_ws_subscribers_for_tests
from orbweaver.auth import mint_token
from orbweaver.store import reset_store_for_tests


@pytest.fixture(autouse=True)
def _store():
    reset_store_for_tests()
    reset_ws_subscribers_for_tests()


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
