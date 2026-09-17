"""LCD context snapshot: workspace label, git branch, prompt tokens."""

import subprocess
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from httpx import ASGITransport, AsyncClient

from orbweaver.app import app
from orbweaver.compact.usage import record_usage, reset_compact_state
from orbweaver.config import settings
from orbweaver.git_ritual import current_branch
from orbweaver.session_status import context_snapshot, workspace_label
from orbweaver.store import reset_store_for_tests


@pytest.fixture(autouse=True)
def _store():
    reset_store_for_tests()
    reset_compact_state()


def test_workspace_label_uses_home_tilde(tmp_path, monkeypatch):
    home = tmp_path / "home"
    root = home / "workspaces"
    checkout = root / "orbweaver"
    checkout.mkdir(parents=True)
    monkeypatch.setattr("orbweaver.session_status.Path.home", lambda: home)
    label = workspace_label("workspace:orbweaver", root, path=checkout)
    assert label == "~/workspaces/orbweaver"


def _init_branch(path: Path, name: str) -> None:
    subprocess.run(["git", "init", "-b", name], cwd=path, check=True, capture_output=True)


def test_current_branch_reads_git_head(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_branch(repo, "ui-facelift")
    assert current_branch(repo) == "ui-facelift"
    assert current_branch(tmp_path / "not-a-repo") == ""


@pytest.fixture
def proj_root(tmp_path):
    (tmp_path / "proj").mkdir()
    _init_branch(tmp_path / "proj", "ui-facelift")
    return tmp_path


@pytest.mark.asyncio
async def test_events_payload_includes_context(proj_root, monkeypatch, auth_header):
    monkeypatch.setattr(settings, "workspace_root", str(proj_root))
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        sess = await client.post(
            "/v1/sessions",
            json={"workspace_uri": "workspace:proj", "workspace_kind": "local", "channel": "web"},
            headers=auth_header,
        )
        assert sess.status_code == 200, sess.text
        sid = sess.json()["id"]
        record_usage(uuid4(), 1, at_seq=0)  # noise: other session
        record_usage(UUID(sid), 58140, at_seq=0)
        r = await client.get(f"/v1/sessions/{sid}/events", headers=auth_header)
        assert r.status_code == 200, r.text
        ctx = r.json()["context"]
        assert ctx["tokens"] == 58140
        assert ctx["window"] == settings.context_window
        assert ctx["branch"] == "ui-facelift"
        assert "proj" in ctx["workspace"]


def test_context_snapshot_zero_tokens_without_usage():
    sid = uuid4()
    snap = context_snapshot(sid, [])
    assert snap["tokens"] == 0
    assert snap["window"] == settings.context_window
    assert "branch" in snap
    assert "workspace" in snap
