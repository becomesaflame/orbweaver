from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

import pytest
from httpx import ASGITransport, AsyncClient

from orbweaver.agent import (
    TOOL_SPEC,
    resolve_channel,
    static_system,
    tools_for_channel,
)
from orbweaver.app import app
from orbweaver.channels.cron import sweep
from orbweaver.channels.telegram import session_for_telegram_user
from orbweaver.config import settings
from orbweaver.permissions.pipeline import can_use_tool
from orbweaver.store import (
    SESSION_TYPE,
    Entity,
    Event,
    Job,
    get_store,
    reset_store_for_tests,
    session_at_id,
)
from orbweaver.workspace import LocalWorkspace


def test_global_tool_spec_still_defines_proposepatch():
    assert any(t["name"] == "ProposePatch" for t in TOOL_SPEC)


@pytest.mark.parametrize("channel", ["telegram", "web", "cron", ""])
def test_non_vscode_tool_specs_omit_proposepatch(channel):
    names = {t["name"] for t in tools_for_channel(channel)}
    assert "ProposePatch" not in names
    assert "Write" in names
    assert "StrReplace" in names


def test_vscode_tool_spec_includes_proposepatch():
    names = {t["name"] for t in tools_for_channel("vscode")}
    assert "ProposePatch" in names
    assert "Write" in names
    assert "StrReplace" in names


def test_system_prompt_mentions_proposepatch_only_for_vscode():
    assert "ProposePatch" not in static_system()
    assert "ProposePatch" not in static_system("telegram")
    assert "ProposePatch" not in static_system("web")
    assert "ProposePatch" not in static_system("cron")
    assert "ProposePatch" in static_system("vscode")


def test_resolve_channel_from_jsonld_title_and_extra():
    assert resolve_channel({"channel": "vscode"}) == "vscode"
    assert resolve_channel({"client": "vscode"}) == "vscode"
    assert resolve_channel({"title": "vscode"}) == "vscode"
    assert resolve_channel({"telegram_user_id": 1}) == "telegram"
    assert resolve_channel({"title": "New chat"}) == ""
    assert resolve_channel(channel="cron") == "cron"
    assert resolve_channel({"channel": "vscode"}, channel="cron") == "cron"
    assert resolve_channel(extra={"channel": "telegram"}) == "telegram"


@pytest.mark.asyncio
async def test_in_project_write_still_auto_applies(tmp_path, monkeypatch):
    async def boom(*_a, **_k):
        raise AssertionError("classifier should not run")

    monkeypatch.setattr("orbweaver.permissions.pipeline.classify_action", boom)
    sid = uuid4()
    ctx = {
        "workspace": LocalWorkspace("workspace:default", str(tmp_path)),
        "workspace_kind": "local",
        "headless": True,
        "session_id": sid,
        "events": [Event(id=uuid4(), session_id=sid, seq=1, kind="user", payload={"text": "edit"})],
    }
    decision = await can_use_tool("Write", {"path": "src/a.py", "content": "x"}, ctx)
    assert decision.behavior == "allow"
    assert decision.fast_path == "acceptEdits"


@pytest.mark.asyncio
async def test_telegram_session_sets_channel():
    store = reset_store_for_tests()
    ent = await session_for_telegram_user(store, 42)
    assert ent.jsonld["channel"] == "telegram"
    names = {t["name"] for t in tools_for_channel(resolve_channel(ent.jsonld))}
    assert "ProposePatch" not in names
    assert "Write" in names


@pytest.mark.asyncio
async def test_telegram_session_migrates_channel():
    store = reset_store_for_tests()
    ent = await session_for_telegram_user(store, 42)
    ent.jsonld.pop("channel", None)
    await store.put_entity(ent)
    migrated = await session_for_telegram_user(store, 42)
    assert migrated.jsonld["channel"] == "telegram"


@pytest.mark.asyncio
async def test_web_session_channel_omits_proposepatch(auth_header):
    reset_store_for_tests()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        sess = await client.post(
            "/v1/sessions",
            json={
                "workspace_uri": "workspace:default",
                "workspace_kind": "local",
                "title": "New chat",
                "channel": "web",
            },
            headers=auth_header,
        )
        assert sess.status_code == 200, sess.text
        sid = sess.json()["id"]
    stored = await get_store().get_entity(UUID(sid))
    assert stored is not None
    assert stored.jsonld["channel"] == "web"
    names = {t["name"] for t in tools_for_channel(resolve_channel(stored.jsonld))}
    assert "ProposePatch" not in names
    assert "Write" in names


@pytest.mark.asyncio
async def test_vscode_session_channel_keeps_proposepatch(auth_header):
    reset_store_for_tests()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        sess = await client.post(
            "/v1/sessions",
            json={
                "workspace_uri": "workspace:default",
                "workspace_kind": "local",
                "title": "orbweaver",
                "channel": "vscode",
            },
            headers=auth_header,
        )
        assert sess.status_code == 200, sess.text
        sid = sess.json()["id"]
    stored = await get_store().get_entity(UUID(sid))
    assert stored is not None
    assert stored.jsonld["channel"] == "vscode"
    names = {t["name"] for t in tools_for_channel(resolve_channel(stored.jsonld))}
    assert "ProposePatch" in names


@pytest.mark.asyncio
async def test_cron_sweep_passes_cron_channel(tmp_path, monkeypatch):
    store = reset_store_for_tests()
    monkeypatch.setattr(settings, "workspace_root", str(tmp_path))
    captured = {}

    async def fake_turn(*_a, **kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr("orbweaver.channels.cron.agent_turn", fake_turn)
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
                "channel": "vscode",
            },
        )
    )
    await store.put_job(
        Job(
            id=uuid4(),
            due_at=datetime.now(UTC) - timedelta(seconds=1),
            payload={"message": "do it"},
            session_id=sid,
        )
    )
    await sweep()
    assert captured["channel"] == "cron"
    names = {t["name"] for t in tools_for_channel(captured["channel"])}
    assert "ProposePatch" not in names
    assert "Write" in names
