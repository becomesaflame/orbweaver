from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from orbweaver.app import app
from orbweaver.store import reset_store_for_tests
from orbweaver.workspace import LocalWorkspace


@pytest.fixture(autouse=True)
def _store():
    reset_store_for_tests()


@pytest.mark.asyncio
async def test_session_and_turn_without_anthropic(tmp_path: Path, monkeypatch, auth_header):
    monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
    from orbweaver.config import settings

    monkeypatch.setattr(settings, "workspace_root", str(tmp_path))
    monkeypatch.setattr(settings, "anthropic_api_key", "")
    (tmp_path / "hello.txt").write_text("hi", encoding="utf-8")
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        headers = auth_header
        sess = await client.post(
            "/v1/sessions",
            json={"workspace_uri": "workspace:default", "workspace_kind": "local"},
            headers=headers,
        )
        assert sess.status_code == 200, sess.text
        sid = sess.json()["id"]
        turned = await client.post(
            f"/v1/sessions/{sid}/turns",
            json={"text": "hello"},
            headers=headers,
        )
        assert turned.status_code == 200, turned.text
        kinds = [e["kind"] for e in turned.json()["events"]]
        assert "assistant" in kinds


@pytest.mark.asyncio
async def test_rejects_absolute_workspace_uri(auth_header):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.post(
            "/v1/sessions",
            json={"workspace_uri": "/Users/me/proj", "workspace_kind": "local"},
            headers=auth_header,
        )
        assert r.status_code == 400


def test_local_workspace_read_write(tmp_path: Path):
    ws = LocalWorkspace("workspace:default", str(tmp_path))
    ws.write("a.txt", "hello")
    assert ws.read("a.txt") == "hello"
    patch = ws.propose_patch("a.txt", "hello", "hello world")
    assert patch["ok"] is True

@pytest.mark.asyncio
async def test_list_sessions_autotitle_and_rename(tmp_path, monkeypatch, auth_header):
    monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
    from orbweaver.config import settings

    monkeypatch.setattr(settings, "workspace_root", str(tmp_path))
    monkeypatch.setattr(settings, "anthropic_api_key", "")
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        headers = auth_header
        sess = await client.post(
            "/v1/sessions",
            json={"workspace_uri": "workspace:default", "workspace_kind": "local"},
            headers=headers,
        )
        assert sess.status_code == 200, sess.text
        sid = sess.json()["id"]
        listed = await client.get("/v1/sessions", headers=headers)
        assert listed.status_code == 200, listed.text
        rows = listed.json()["sessions"]
        assert rows[0]["id"] == sid
        assert rows[0]["title"] == "New chat"
        turned = await client.post(
            f"/v1/sessions/{sid}/turns",
            json={"text": "wire up the sidebar"},
            headers=headers,
        )
        assert turned.status_code == 200, turned.text
        listed = await client.get("/v1/sessions", headers=headers)
        assert listed.json()["sessions"][0]["title"] == "wire up the sidebar"
        patched = await client.patch(
            f"/v1/sessions/{sid}",
            json={"title": "Sidebar"},
            headers=headers,
        )
        assert patched.status_code == 200, patched.text
        assert patched.json()["title"] == "Sidebar"
        listed = await client.get("/v1/sessions", headers=headers)
        assert listed.json()["sessions"][0]["title"] == "Sidebar"
