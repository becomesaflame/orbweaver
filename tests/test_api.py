from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

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
async def test_delete_job_route_rejects_bad_id(auth_header):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        bad = await client.delete("/v1/jobs/not-a-uuid", headers=auth_header)
        assert bad.status_code == 422, bad.text


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
async def test_list_sessions_does_not_load_full_event_logs(auth_header, monkeypatch):
    """The rail polls GET /v1/sessions every few seconds.

    Listing used to ``list_events`` every chat. Decoding those payloads on the
    asyncio loop froze the web UI for a minute or two until the listing
    finished, then the next poll started the stall again.
    """
    from uuid import UUID

    store = reset_store_for_tests()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        sess = await client.post(
            "/v1/sessions",
            json={"workspace_uri": "workspace:default", "title": "fat log"},
            headers=auth_header,
        )
        assert sess.status_code == 200, sess.text
        sid = UUID(sess.json()["id"])
        await store.append_event(sid, "user", {"text": "first prompt here"})
        blob = "x" * 50_000
        for i in range(40):
            await store.append_event(sid, "tool_result", {"content": blob, "i": i})

        async def boom(_session_id):
            raise AssertionError("list_events must not run for GET /v1/sessions")

        monkeypatch.setattr(store, "list_events", boom)
        listed = await client.get("/v1/sessions", headers=auth_header)
        assert listed.status_code == 200, listed.text
        row = next(s for s in listed.json()["sessions"] if s["id"] == str(sid))
        assert row["event_count"] == 41
        assert row["preview"].startswith("first prompt")
        assert row["title"] == "fat log"


@pytest.mark.asyncio
async def test_delete_session_hides_it_but_keeps_events(tmp_path: Path, monkeypatch, auth_header):
    """Soft delete: gone from the listing, entity and events still in the store."""
    monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
    from orbweaver.config import settings

    monkeypatch.setattr(settings, "workspace_root", str(tmp_path))
    monkeypatch.setattr(settings, "anthropic_api_key", "")
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        headers = auth_header
        keep = await client.post(
            "/v1/sessions",
            json={"workspace_uri": "workspace:default", "workspace_kind": "local"},
            headers=headers,
        )
        drop = await client.post(
            "/v1/sessions",
            json={"workspace_uri": "workspace:default", "workspace_kind": "local"},
            headers=headers,
        )
        keep_id, drop_id = keep.json()["id"], drop.json()["id"]
        turned = await client.post(
            f"/v1/sessions/{drop_id}/turns", json={"text": "some history"}, headers=headers
        )
        assert turned.status_code == 200, turned.text

        listed = await client.get("/v1/sessions", headers=headers)
        assert {s["id"] for s in listed.json()["sessions"]} == {keep_id, drop_id}

        deleted = await client.delete(f"/v1/sessions/{drop_id}", headers=headers)
        assert deleted.status_code == 200, deleted.text
        assert deleted.json()["status"] == "deleted"

        listed = await client.get("/v1/sessions", headers=headers)
        assert [s["id"] for s in listed.json()["sessions"]] == [keep_id]

        # Soft delete, so the transcript survives and the delete is recoverable.
        from uuid import UUID

        from orbweaver.store import get_store, is_deleted_session

        store = get_store()
        ent = await store.get_entity(UUID(drop_id))
        assert ent is not None and is_deleted_session(ent)
        assert ent.jsonld.get("deleted_at")
        assert await store.list_events(UUID(drop_id)), "events must be kept"


@pytest.mark.asyncio
async def test_delete_session_is_idempotent_and_404s_unknown(auth_header):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        made = await client.post(
            "/v1/sessions", json={"workspace_uri": "workspace:default"}, headers=auth_header
        )
        sid = made.json()["id"]
        first = await client.delete(f"/v1/sessions/{sid}", headers=auth_header)
        second = await client.delete(f"/v1/sessions/{sid}", headers=auth_header)
        assert first.status_code == 200, first.text
        assert second.status_code == 200, second.text
        assert second.json()["status"] == "deleted"

        missing = await client.delete(
            "/v1/sessions/00000000-0000-4000-8000-000000000000", headers=auth_header
        )
        assert missing.status_code == 404

        denied = await client.delete(f"/v1/sessions/{sid}")
        assert denied.status_code == 401


@pytest.mark.asyncio
async def test_delete_session_refuses_while_a_turn_runs(auth_header):
    """Deleting mid-turn would hide a session the running turn still writes to."""
    from uuid import UUID

    from orbweaver.turns import acquire as acquire_turn
    from orbweaver.turns import release as release_turn

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        made = await client.post(
            "/v1/sessions", json={"workspace_uri": "workspace:default"}, headers=auth_header
        )
        sid = made.json()["id"]
        state = acquire_turn(UUID(sid), "web")
        assert state is not None
        try:
            busy = await client.delete(f"/v1/sessions/{sid}", headers=auth_header)
            assert busy.status_code == 409, busy.text
        finally:
            release_turn(UUID(sid), state)
        # Once the turn is done the same delete succeeds.
        ok = await client.delete(f"/v1/sessions/{sid}", headers=auth_header)
        assert ok.status_code == 200, ok.text


@pytest.mark.asyncio
async def test_turn_on_deleted_session_is_refused(tmp_path: Path, monkeypatch, auth_header):
    """A deleted chat must not be revivable by posting a turn to its id."""
    monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
    from orbweaver.config import settings

    monkeypatch.setattr(settings, "workspace_root", str(tmp_path))
    monkeypatch.setattr(settings, "anthropic_api_key", "")
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        made = await client.post(
            "/v1/sessions",
            json={"workspace_uri": "workspace:default", "workspace_kind": "local"},
            headers=auth_header,
        )
        sid = made.json()["id"]
        assert (await client.delete(f"/v1/sessions/{sid}", headers=auth_header)).status_code == 200
        turned = await client.post(
            f"/v1/sessions/{sid}/turns", json={"text": "are you there"}, headers=auth_header
        )
        assert turned.status_code == 404, turned.text
        listed = await client.get("/v1/sessions", headers=auth_header)
        assert sid not in {s["id"] for s in listed.json()["sessions"]}



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
        alias = await client.post(
            "/v1/sessions",
            json={"workspace_uri": "workspace:default", "channel": "VS CODE"},
            headers=auth_header,
        )
        assert alias.status_code == 200, alias.text
        assert alias.json()["channel"] == "vscode"
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
async def test_delete_job_cancels_scheduled_job(auth_header):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        due = (datetime.now(UTC) + timedelta(hours=1)).isoformat()
        created = await client.post(
            "/v1/jobs",
            json={"due_at": due, "message": "ping later"},
            headers=auth_header,
        )
        assert created.status_code == 200, created.text
        job_id = created.json()["id"]

        listed = await client.get("/v1/jobs", headers=auth_header)
        assert listed.status_code == 200, listed.text
        assert any(j["id"] == job_id for j in listed.json()["jobs"])

        deleted = await client.delete(f"/v1/jobs/{job_id}", headers=auth_header)
        assert deleted.status_code == 200, deleted.text
        assert deleted.json()["id"] == job_id

        listed = await client.get("/v1/jobs", headers=auth_header)
        assert job_id not in {j["id"] for j in listed.json()["jobs"]}


@pytest.mark.asyncio
async def test_delete_job_is_404_unknown_and_requires_auth(auth_header):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        missing = await client.delete(f"/v1/jobs/{uuid4()}", headers=auth_header)
        assert missing.status_code == 404
        assert missing.json()["detail"] == "job not found"

        denied = await client.delete(f"/v1/jobs/{uuid4()}")
        assert denied.status_code == 401


@pytest.mark.asyncio
async def test_health_includes_version():
    from orbweaver import __version__

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"
        assert r.json()["version"] == __version__
        assert "provider" in r.json()["llm"]
        assert "model" in r.json()["llm"]
