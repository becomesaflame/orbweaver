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
async def test_session_and_turn_without_anthropic(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
    from orbweaver.config import settings

    monkeypatch.setattr(settings, "workspace_root", str(tmp_path))
    monkeypatch.setattr(settings, "anthropic_api_key", "")
    (tmp_path / "hello.txt").write_text("hi", encoding="utf-8")
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        tok = (await client.post("/v1/auth/token", json={"sub": "t"})).json()["token"]
        headers = {"authorization": f"Bearer {tok}"}
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
async def test_rejects_absolute_workspace_uri():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        tok = (await client.post("/v1/auth/token", json={"sub": "t"})).json()["token"]
        r = await client.post(
            "/v1/sessions",
            json={"workspace_uri": "/Users/me/proj", "workspace_kind": "local"},
            headers={"authorization": f"Bearer {tok}"},
        )
        assert r.status_code == 400


def test_local_workspace_read_write(tmp_path: Path):
    ws = LocalWorkspace("workspace:default", str(tmp_path))
    ws.write("a.txt", "hello")
    assert ws.read("a.txt") == "hello"
    patch = ws.propose_patch("a.txt", "hello", "hello world")
    assert patch["ok"] is True
