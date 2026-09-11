"""Issue #97: refuse forgeable JWT secrets, same-origin CORS, WS token off the URL."""

from __future__ import annotations

import logging
import sys
from urllib.parse import quote
from uuid import uuid4

import pytest
import uvicorn
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from orbweaver.app import app, configure_cors, reset_ws_subscribers_for_tests
from orbweaver.auth import (
    INSECURE_DEFAULT_SECRET,
    InsecureJwtSecretError,
    check_jwt_secret,
    jwt_secret_problem,
    mint_token,
)
from orbweaver.cli import main
from orbweaver.config import Settings, settings
from orbweaver.store import reset_store_for_tests

STRONG_SECRET = "s" * 32


@pytest.fixture(autouse=True)
def _store():
    reset_store_for_tests()
    reset_ws_subscribers_for_tests()


# --- JWT secret -------------------------------------------------------------


@pytest.mark.parametrize(
    "secret",
    ["", INSECURE_DEFAULT_SECRET, "short", "x" * 31],
    ids=["empty", "default", "short", "31-bytes"],
)
def test_weak_secrets_are_rejected(secret):
    assert jwt_secret_problem(secret) is not None
    cfg = Settings(orbweaver_jwt_secret=secret, orbweaver_dev_insecure=False)
    with pytest.raises(InsecureJwtSecretError, match="ORBWEAVER_JWT_SECRET"):
        check_jwt_secret(cfg)


def test_strong_secret_passes():
    assert jwt_secret_problem(STRONG_SECRET) is None
    check_jwt_secret(Settings(orbweaver_jwt_secret=STRONG_SECRET, orbweaver_dev_insecure=False))


def test_dev_insecure_flag_allows_default_secret():
    cfg = Settings(orbweaver_jwt_secret=INSECURE_DEFAULT_SECRET, orbweaver_dev_insecure=True)
    check_jwt_secret(cfg)


def test_app_startup_fails_with_default_secret(monkeypatch):
    monkeypatch.setattr(settings, "orbweaver_jwt_secret", INSECURE_DEFAULT_SECRET)
    monkeypatch.setattr(settings, "orbweaver_dev_insecure", False)
    with pytest.raises(InsecureJwtSecretError), TestClient(app):
        pass


def test_app_startup_ok_with_dev_override(monkeypatch):
    monkeypatch.setattr(settings, "orbweaver_jwt_secret", INSECURE_DEFAULT_SECRET)
    monkeypatch.setattr(settings, "orbweaver_dev_insecure", True)
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200


def test_app_startup_ok_with_strong_secret(monkeypatch):
    monkeypatch.setattr(settings, "orbweaver_jwt_secret", STRONG_SECRET)
    monkeypatch.setattr(settings, "orbweaver_dev_insecure", False)
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200


def test_health_stays_unauthenticated():
    with TestClient(app) as client:
        r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_serve_cli_exits_nonzero_with_default_secret(monkeypatch, caplog):
    calls: list[dict] = []
    monkeypatch.setattr(uvicorn, "run", lambda *a, **kw: calls.append(kw))
    monkeypatch.setattr(settings, "orbweaver_jwt_secret", INSECURE_DEFAULT_SECRET)
    monkeypatch.setattr(settings, "orbweaver_dev_insecure", False)
    monkeypatch.setattr(sys, "argv", ["orbweaver", "serve"])
    with caplog.at_level(logging.ERROR, logger="orbweaver"), pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 1
    assert "refusing to start" in caplog.text
    assert "ORBWEAVER_JWT_SECRET" in caplog.text
    assert "ORBWEAVER_DEV_INSECURE=1" in caplog.text
    assert calls == []


def test_serve_cli_runs_with_dev_override(monkeypatch):
    calls: list[dict] = []
    monkeypatch.setattr(uvicorn, "run", lambda *a, **kw: calls.append(kw))
    monkeypatch.setattr(settings, "orbweaver_jwt_secret", INSECURE_DEFAULT_SECRET)
    monkeypatch.setattr(settings, "orbweaver_dev_insecure", True)
    monkeypatch.setattr(sys, "argv", ["orbweaver", "serve"])
    main()
    assert len(calls) == 1


def test_serve_cli_runs_with_strong_secret(monkeypatch):
    calls: list[dict] = []
    monkeypatch.setattr(uvicorn, "run", lambda *a, **kw: calls.append(kw))
    monkeypatch.setattr(settings, "orbweaver_jwt_secret", STRONG_SECRET)
    monkeypatch.setattr(settings, "orbweaver_dev_insecure", False)
    monkeypatch.setattr(sys, "argv", ["orbweaver", "serve"])
    main()
    assert len(calls) == 1


# --- CORS -------------------------------------------------------------------


def test_cors_origins_setting_parses_list():
    cfg = Settings(orbweaver_cors_origins=" https://a.example, https://b.example ,")
    assert cfg.cors_origins == ["https://a.example", "https://b.example"]
    assert Settings(orbweaver_cors_origins="").cors_origins == []


def _preflight(client: TestClient, origin: str):
    return client.options(
        "/v1/sessions",
        headers={
            "origin": origin,
            "access-control-request-method": "POST",
            "access-control-request-headers": "authorization,content-type",
        },
    )


def test_default_cors_is_same_origin_only():
    assert settings.cors_origins == []
    with TestClient(app) as client:
        pre = _preflight(client, "https://evil.example")
        assert "access-control-allow-origin" not in pre.headers
        assert "access-control-allow-credentials" not in pre.headers
        plain = client.get("/health", headers={"origin": "https://evil.example"})
    assert plain.status_code == 200
    assert "access-control-allow-origin" not in plain.headers


def test_configured_cors_origin_is_allowed_with_credentials():
    other = FastAPI()

    @other.post("/v1/sessions")
    async def _ep() -> dict[str, str]:
        return {"ok": "yes"}

    configure_cors(other, ["https://ui.example"])
    client = TestClient(other)
    good = _preflight(client, "https://ui.example")
    assert good.headers.get("access-control-allow-origin") == "https://ui.example"
    assert good.headers.get("access-control-allow-credentials") == "true"
    bad = _preflight(client, "https://evil.example")
    assert "access-control-allow-origin" not in bad.headers


def test_wildcard_cors_never_sends_credentials():
    other = FastAPI()

    @other.get("/x")
    async def _ep() -> dict[str, str]:
        return {"ok": "yes"}

    configure_cors(other, ["*"])
    client = TestClient(other)
    r = client.get("/x", headers={"origin": "https://any.example"})
    assert r.headers.get("access-control-allow-origin") == "*"
    assert "access-control-allow-credentials" not in r.headers


# --- WebSocket token transport ---------------------------------------------


def _open_session(client: TestClient) -> str:
    headers = {"authorization": f"Bearer {mint_token('t')}"}
    r = client.post(
        "/v1/sessions",
        json={"workspace_uri": "workspace:default", "workspace_kind": "local"},
        headers=headers,
    )
    assert r.status_code == 200, r.text
    return r.json()["id"]


def test_ws_subprotocol_token_is_accepted_and_echoed():
    token = mint_token("t")
    with TestClient(app) as client:
        sid = _open_session(client)
        with client.websocket_connect(
            f"/v1/sessions/{sid}/ws", subprotocols=["bearer", token]
        ) as ws:
            assert ws.accepted_subprotocol == "bearer"
            assert ws.receive_json()["kind"] == "subscribed"


def test_ws_subprotocol_bad_token_is_rejected():
    with (
        TestClient(app) as client,
        pytest.raises(WebSocketDisconnect) as exc,
        client.websocket_connect(
            f"/v1/sessions/{uuid4()}/ws", subprotocols=["bearer", "not-a-jwt"]
        ) as ws,
    ):
        ws.receive_text()
    assert exc.value.code == 4401


def test_ws_subprotocol_bearer_without_token_is_rejected():
    with (
        TestClient(app) as client,
        pytest.raises(WebSocketDisconnect) as exc,
        client.websocket_connect(f"/v1/sessions/{uuid4()}/ws", subprotocols=["bearer"]) as ws,
    ):
        ws.receive_text()
    assert exc.value.code == 4401


def test_ws_first_frame_auth_is_accepted():
    token = mint_token("t")
    with TestClient(app) as client:
        sid = _open_session(client)
        with client.websocket_connect(f"/v1/sessions/{sid}/ws") as ws:
            ws.send_json({"type": "auth", "token": token})
            assert ws.receive_json()["kind"] == "subscribed"
            ws.send_json({"type": "ping"})
            assert ws.receive_json()["kind"] == "pong"


def test_ws_first_frame_bad_token_is_rejected():
    with (
        TestClient(app) as client,
        pytest.raises(WebSocketDisconnect) as exc,
        client.websocket_connect(f"/v1/sessions/{uuid4()}/ws") as ws,
    ):
        ws.send_json({"type": "auth", "token": "not-a-jwt"})
        ws.receive_text()
    assert exc.value.code == 4401


def test_ws_first_frame_must_be_auth():
    with (
        TestClient(app) as client,
        pytest.raises(WebSocketDisconnect) as exc,
        client.websocket_connect(f"/v1/sessions/{uuid4()}/ws") as ws,
    ):
        ws.send_json({"text": "not an auth frame"})
        ws.receive_text()
    assert exc.value.code == 4401


def test_ws_query_token_still_works_with_deprecation_warning(caplog):
    token = mint_token("t")
    with TestClient(app) as client:
        sid = _open_session(client)
        with (
            caplog.at_level(logging.WARNING, logger="orbweaver.app"),
            client.websocket_connect(f"/v1/sessions/{sid}/ws?token={quote(token)}") as ws,
        ):
            assert ws.accepted_subprotocol is None
            assert ws.receive_json()["kind"] == "subscribed"
    assert any(
        "?token=" in rec.getMessage() and "deprecated" in rec.getMessage()
        for rec in caplog.records
    )


def test_ws_query_bad_token_is_rejected():
    with (
        TestClient(app) as client,
        pytest.raises(WebSocketDisconnect) as exc,
        client.websocket_connect(f"/v1/sessions/{uuid4()}/ws?token=not-a-jwt") as ws,
    ):
        ws.receive_text()
    assert exc.value.code == 4401
