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


@pytest.mark.asyncio
async def test_docker_workspace_kind_coerced_to_local(auth_header):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.post(
            "/v1/sessions",
            json={"workspace_uri": "workspace:default", "workspace_kind": "docker"},
            headers=auth_header,
        )
        assert r.status_code == 200, r.text
        sid = r.json()["id"]
        listed = await client.get("/v1/sessions", headers=auth_header)
        row = next(s for s in listed.json()["sessions"] if s["id"] == sid)
        assert row["workspace_kind"] == "local"


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



@pytest.mark.asyncio
async def test_browse_workspaces(tmp_path: Path, monkeypatch, auth_header):
    (tmp_path / "orbweaver").mkdir()
    monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
    from orbweaver.config import settings

    monkeypatch.setattr(settings, "workspace_root", str(tmp_path))
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        denied = await client.get("/v1/workspaces")
        assert denied.status_code == 401
        listed = await client.get("/v1/workspaces", headers=auth_header)
        assert listed.status_code == 200, listed.text
        body = listed.json()
        assert body["uri"] == "workspace:default"
        assert [d["name"] for d in body["dirs"]] == ["orbweaver"]
        child = await client.get("/v1/workspaces", params={"rel": "orbweaver"}, headers=auth_header)
        assert child.status_code == 200, child.text
        assert child.json()["uri"] == "workspace:orbweaver"
        made = await client.post(
            "/v1/workspaces",
            json={"parent": "orbweaver", "name": "notes"},
            headers=auth_header,
        )
        assert made.status_code == 200, made.text
        assert made.json()["uri"] == "file:./orbweaver/notes"
        escaped = await client.get("/v1/workspaces", params={"rel": ".."}, headers=auth_header)
        assert escaped.status_code == 400


@pytest.mark.asyncio
async def test_session_channel_vscode(auth_header):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        bad = await client.post(
            "/v1/sessions",
            json={"workspace_uri": "workspace:default", "channel": "VS CODE"},
            headers=auth_header,
        )
        assert bad.status_code == 400
        created = await client.post(
            "/v1/sessions",
            json={
                "workspace_uri": "workspace:default",
                "workspace_kind": "local",
                "title": "vscode",
                "channel": "vscode",
            },
            headers=auth_header,
        )
        assert created.status_code == 200, created.text
        assert created.json()["channel"] == "vscode"
        listed = await client.get("/v1/sessions", headers=auth_header)
        row = next(s for s in listed.json()["sessions"] if s["id"] == created.json()["id"])
        assert row["channel"] == "vscode"


@pytest.mark.asyncio
async def test_health_includes_version():
    from orbweaver import __version__

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"
        assert r.json()["version"] == __version__
