"""Deploy must wait for in-flight turns instead of SIGTERM-killing them."""

from __future__ import annotations

import logging
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import httpx
import pytest
from httpx import ASGITransport, AsyncClient

from orbweaver import turns
from orbweaver.app import app
from orbweaver.channels import cron
from orbweaver.channels import telegram as tg
from orbweaver.cli import main
from orbweaver.config import settings
from orbweaver.drain import is_local_admin, wait_for_idle
from orbweaver.store import SESSION_TYPE, Entity, Job, reset_store_for_tests, session_at_id
from orbweaver.workspace import LocalWorkspace


def test_acquire_raises_when_draining_new_session():
    running = uuid4()
    holder = turns.acquire(running, channel="telegram")
    assert holder is not None
    turns.begin_drain()
    assert turns.get(running) is holder
    assert turns.acquire(running, channel="web") is None
    with pytest.raises(turns.GatewayDraining, match="draining"):
        turns.acquire(uuid4(), channel="telegram")
    with pytest.raises(turns.GatewayDraining):
        turns.acquire(uuid4(), channel="web")
    with pytest.raises(turns.GatewayDraining):
        turns.acquire(uuid4(), channel="cron")
    child = turns.acquire(uuid4(), channel="subagent")
    assert child is not None


def test_wait_for_idle_holds_until_running_turn_finishes():
    polls = {"n": 0}
    turn = {"session_id": str(uuid4()), "channel": "telegram"}

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if request.method == "POST" and path.endswith("/v1/admin/drain"):
            return httpx.Response(200, json={"draining": True, "turns": [turn], "count": 1})
        polls["n"] += 1
        if polls["n"] < 3:
            return httpx.Response(200, json={"draining": True, "turns": [turn], "count": 1})
        return httpx.Response(200, json={"draining": True, "turns": [], "count": 0})

    slept: list[float] = []
    transport = httpx.MockTransport(handler)
    with httpx.Client(transport=transport, base_url="http://127.0.0.1:8080") as client:
        assert wait_for_idle(client, timeout_s=10, interval_s=0.01, sleep=slept.append) == "idle"
    assert polls["n"] >= 3
    assert slept


def test_wait_for_idle_skips_missing_endpoint():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"detail": "not found"})

    with httpx.Client(
        transport=httpx.MockTransport(handler), base_url="http://127.0.0.1:8080"
    ) as client:
        assert wait_for_idle(client, timeout_s=1, interval_s=0.01, sleep=lambda _s: None) == "unavailable"


def test_wait_for_idle_times_out_while_turn_still_running():
    turn = {"session_id": str(uuid4()), "channel": "telegram"}

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"draining": True, "turns": [turn], "count": 1})

    with httpx.Client(
        transport=httpx.MockTransport(handler), base_url="http://127.0.0.1:8080"
    ) as client:
        assert wait_for_idle(client, timeout_s=0, interval_s=0.01, sleep=lambda _s: None) == "timeout"


def test_wait_for_idle_when_gateway_is_down():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    with httpx.Client(
        transport=httpx.MockTransport(handler), base_url="http://127.0.0.1:8080"
    ) as client:
        assert wait_for_idle(client, timeout_s=1, sleep=lambda _s: None) == "unavailable"


def test_local_admin_rejects_forwarded_clients():
    loopback = SimpleNamespace(client=SimpleNamespace(host="127.0.0.1"), headers={})
    proxied = SimpleNamespace(
        client=SimpleNamespace(host="127.0.0.1"),
        headers={"x-forwarded-for": "203.0.113.9"},
    )
    assert is_local_admin(loopback) is True
    assert is_local_admin(proxied) is False


def test_drain_cli_uses_waiter(monkeypatch):
    seen: dict[str, object] = {}

    def fake_wait(client, *, timeout_s, interval_s):
        seen["timeout"] = timeout_s
        seen["url"] = str(client.base_url)
        seen["interval"] = interval_s
        return "idle"

    monkeypatch.setattr("orbweaver.drain.wait_for_idle", fake_wait)
    monkeypatch.setattr(
        sys, "argv", ["orbweaver", "drain", "--url", "http://127.0.0.1:8080", "--timeout", "12"]
    )
    main()
    assert seen["timeout"] == 12.0
    assert "127.0.0.1:8080" in str(seen["url"])


def test_update_live_waits_before_restart():
    script = Path(__file__).resolve().parents[1] / "deploy" / "update-live.sh"
    text = script.read_text(encoding="utf-8")
    drain_at = text.index("orbweaver.cli drain")
    restart_at = text.index("systemctl --user restart orbweaver")
    assert drain_at < restart_at


@pytest.mark.asyncio
async def test_admin_drain_is_loopback_only():
    sid = uuid4()
    holder = turns.acquire(sid, channel="telegram")
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        denied = await client.post("/v1/admin/drain", headers={"x-forwarded-for": "203.0.113.9"})
        assert denied.status_code == 403
        assert turns.get(sid) is holder
        assert turns.is_draining() is False
        ok = await client.post("/v1/admin/drain")
        assert ok.status_code == 200, ok.text
        body = ok.json()
        assert body["draining"] is True
        assert body["count"] == 1
        assert body["turns"][0]["channel"] == "telegram"
        status = await client.get("/v1/admin/drain")
        assert status.json()["count"] == 1


@pytest.mark.asyncio
async def test_web_turn_503_while_draining(tmp_path, monkeypatch, auth_header):
    monkeypatch.setattr(settings, "workspace_root", str(tmp_path))
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        sess = await client.post(
            "/v1/sessions",
            json={"workspace_uri": "workspace:default", "workspace_kind": "local"},
            headers=auth_header,
        )
        assert sess.status_code == 200, sess.text
        sid = sess.json()["id"]
        drain = await client.post("/v1/admin/drain")
        assert drain.status_code == 200
        blocked = await client.post(
            f"/v1/sessions/{sid}/turns", json={"text": "hello"}, headers=auth_header
        )
        assert blocked.status_code == 503
        assert "draining" in blocked.json()["detail"]


@pytest.mark.asyncio
async def test_telegram_replies_when_gateway_is_draining(tmp_path, monkeypatch):
    store = reset_store_for_tests()
    sid = uuid4()
    sess = Entity(
        id=sid,
        at_id=session_at_id(sid),
        at_type=SESSION_TYPE,
        jsonld={
            "@id": session_at_id(sid),
            "@type": SESSION_TYPE,
            "workspace_uri": "workspace:default",
            "workspace_kind": "local",
            "channel": "telegram",
            "telegram_user_id": 1,
            "telegram_chat_id": 1,
        },
    )
    await store.put_entity(sess)
    turns.begin_drain()
    bound = tg.Bound(
        store=store,
        operator=sess,
        target=sess,
        ws=LocalWorkspace("workspace:default", str(tmp_path)),
        kind="local",
        chat_id=1,
    )

    async def fake_workspace(_update, _context, **_kw):
        return bound

    monkeypatch.setattr(tg, "_session_workspace", fake_workspace)
    update = MagicMock()
    update.message.reply_text = AsyncMock()
    await tg._run_turn(update, MagicMock(), "are you there?")
    assert "updating" in update.message.reply_text.await_args.args[0].lower()
    assert not turns.is_running(sid)


@pytest.mark.asyncio
async def test_cron_skips_when_gateway_is_draining(tmp_path, monkeypatch, caplog):
    store = reset_store_for_tests()
    monkeypatch.setattr(settings, "workspace_root", str(tmp_path))
    fake_turn = AsyncMock(return_value=[])
    monkeypatch.setattr(cron, "agent_turn", fake_turn)
    sid = uuid4()
    await store.put_entity(
        Entity(
            id=sid,
            at_id=session_at_id(sid),
            at_type=SESSION_TYPE,
            jsonld={
                "@id": session_at_id(sid),
                "@type": SESSION_TYPE,
                "workspace_uri": "workspace:default",
                "workspace_kind": "local",
            },
        )
    )
    job = Job(
        id=uuid4(),
        due_at=datetime.now(UTC) - timedelta(seconds=1),
        payload={"message": "nightly"},
        recurrence="day",
        session_id=sid,
    )
    await store.put_job(job)
    turns.begin_drain()
    with caplog.at_level(logging.INFO, logger="orbweaver.channels.cron"):
        await cron.sweep_and_wait()
    fake_turn.assert_not_awaited()
    assert any("draining" in r.getMessage() for r in caplog.records)
    assert not turns.is_running(sid)
