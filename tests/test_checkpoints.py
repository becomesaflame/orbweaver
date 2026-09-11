"""Per-turn workspace checkpoints: rewind restores files, not just the transcript."""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
from uuid import uuid4

import pytest
from httpx import ASGITransport, AsyncClient

from orbweaver.app import app
from orbweaver.checkpoints import (
    CheckpointHeadMismatch,
    checkpoint_ref,
    prune_checkpoints,
    restore_checkpoint,
    take_checkpoint,
    turn_edits_files,
)
from orbweaver.config import settings
from orbweaver.store import Event, get_store, reset_store_for_tests

_GIT_ENV = {
    **os.environ,
    "GIT_CONFIG_GLOBAL": "/dev/null",
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_AUTHOR_NAME": "Orbweaver Test",
    "GIT_AUTHOR_EMAIL": "test@orbweaver.local",
    "GIT_COMMITTER_NAME": "Orbweaver Test",
    "GIT_COMMITTER_EMAIL": "test@orbweaver.local",
    "GIT_TERMINAL_PROMPT": "0",
}


def _git(cwd: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(cwd), *args],
        capture_output=True,
        text=True,
        check=True,
        env=_GIT_ENV,
    )
    return proc.stdout.strip()


def _init_repo(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    _git(path, "init", "-q", "-b", "main")
    (path / "keep.txt").write_text("keep\n", encoding="utf-8")
    (path / "tracked.txt").write_text("original\n", encoding="utf-8")
    (path / "gone.txt").write_text("about to be deleted\n", encoding="utf-8")
    (path / ".gitignore").write_text("*.log\n.orbweaver/\n", encoding="utf-8")
    _git(path, "add", "-A")
    _git(path, "commit", "-q", "-m", "base")
    (path / "ignored.log").write_text("junk\n", encoding="utf-8")
    return path


def _tree_listing(root: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root).as_posix()
        if rel.split("/")[0] in {".git", ".orbweaver"}:
            continue
        if p.is_file():
            out[rel] = p.read_text(encoding="utf-8")
    return out


def _fake_turn(repo: Path) -> None:
    """What a misbehaving turn does: writes two files, edits one, deletes one."""
    (repo / "new_a.txt").write_text("a\n", encoding="utf-8")
    (repo / "pkg").mkdir()
    (repo / "pkg" / "new_b.txt").write_text("b\n", encoding="utf-8")
    (repo / "tracked.txt").write_text("changed by the agent\n", encoding="utf-8")
    (repo / "gone.txt").unlink()


@pytest.fixture(autouse=True)
def _store():
    reset_store_for_tests()


@pytest.fixture
def checkpoints_on(monkeypatch):
    monkeypatch.setattr(settings, "orbweaver_checkpoints", True)
    monkeypatch.setattr(settings, "anthropic_api_key", "")


@pytest.fixture
def client_for(monkeypatch, auth_header):
    def _make(root: Path):
        monkeypatch.setenv("WORKSPACE_ROOT", str(root))
        monkeypatch.setattr(settings, "workspace_root", str(root))
        return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")

    return _make


async def _open(client, headers, uri="workspace:ws") -> str:
    r = await client.post(
        "/v1/sessions",
        json={"workspace_uri": uri, "workspace_kind": "local"},
        headers=headers,
    )
    assert r.status_code == 200, r.text
    return r.json()["id"]


async def _turn(client, headers, sid: str, text: str) -> dict:
    turned = await client.post(f"/v1/sessions/{sid}/turns", json={"text": text}, headers=headers)
    assert turned.status_code == 200, turned.text
    listed = await client.get(f"/v1/sessions/{sid}/events", headers=headers)
    return next(e for e in listed.json()["events"] if e["kind"] == "user")


async def _record_edit_tool(sid: str) -> None:
    from uuid import UUID

    store = get_store()
    await store.append_event(
        UUID(sid),
        "tool_call",
        {"id": "toolu_1", "name": "Write", "input": {"path": "new_a.txt", "content": "a\n"}},
    )
    await store.append_event(
        UUID(sid), "tool_result", {"tool_use_id": "toolu_1", "name": "Write", "content": "ok"}
    )


async def test_rewind_restores_files_and_leaves_real_index_alone(
    tmp_path, checkpoints_on, client_for, auth_header
):
    repo = _init_repo(tmp_path / "ws")
    (repo / "staged.txt").write_text("staged, not committed\n", encoding="utf-8")
    _git(repo, "add", "staged.txt")
    before_tree = _tree_listing(repo)
    before_index = _git(repo, "ls-files", "-s")
    before_status = _git(repo, "status", "--porcelain")

    async with client_for(tmp_path) as client:
        sid = await _open(client, auth_header)
        user = await _turn(client, auth_header, sid, "rewrite everything")
        cp = user["payload"]["checkpoint"]
        assert cp["tree"] and cp["head"] == _git(repo, "rev-parse", "HEAD")
        ref = checkpoint_ref(sid, user["seq"])
        assert _git(repo, "rev-parse", f"{ref}^{{tree}}") == cp["tree"]
        listed = _git(repo, "ls-tree", "-r", "--name-only", cp["tree"]).splitlines()
        assert "staged.txt" in listed and "ignored.log" not in listed
        assert not any(name.startswith(".orbweaver") for name in listed)

        _fake_turn(repo)
        await _record_edit_tool(sid)
        assert _tree_listing(repo) != before_tree

        # restore_files omitted: a Write tool call in the discarded turn turns it on.
        rewound = await client.post(
            f"/v1/sessions/{sid}/rewind", json={"from_seq": user["seq"]}, headers=auth_header
        )
        assert rewound.status_code == 200, rewound.text
        body = rewound.json()
        assert body["restore_files"] is True
        summary = body["checkpoint_restore"]
        assert summary["status"] == "restored"
        assert sorted(summary["restored"]) == ["gone.txt", "tracked.txt"]
        assert sorted(summary["deleted"]) == ["new_a.txt", "pkg/new_b.txt"]

        assert _tree_listing(repo) == before_tree
        assert not (repo / "pkg").exists()
        assert (repo / "ignored.log").read_text(encoding="utf-8") == "junk\n"
        assert _git(repo, "ls-files", "-s") == before_index
        assert _git(repo, "status", "--porcelain") == before_status

        events = (await client.get(f"/v1/sessions/{sid}/events", headers=auth_header)).json()[
            "events"
        ]
        assert [e["kind"] for e in events] == ["checkpoint_restore"]
        assert events[0]["payload"]["deleted"] == summary["deleted"]
        assert summary["seq"] == events[0]["seq"]


async def test_rewind_default_off_without_edit_tools(
    tmp_path, checkpoints_on, client_for, auth_header
):
    repo = _init_repo(tmp_path / "ws")
    async with client_for(tmp_path) as client:
        sid = await _open(client, auth_header)
        user = await _turn(client, auth_header, sid, "just chatting")
        (repo / "untouched_by_rewind.txt").write_text("x\n", encoding="utf-8")
        rewound = await client.post(
            f"/v1/sessions/{sid}/rewind", json={"from_seq": user["seq"]}, headers=auth_header
        )
        assert rewound.status_code == 200, rewound.text
        assert rewound.json()["restore_files"] is False
        assert "checkpoint_restore" not in rewound.json()
        assert (repo / "untouched_by_rewind.txt").exists()


async def test_rewind_refuses_when_head_moved_unless_forced(
    tmp_path, checkpoints_on, client_for, auth_header
):
    repo = _init_repo(tmp_path / "ws")
    before_tree = _tree_listing(repo)
    async with client_for(tmp_path) as client:
        sid = await _open(client, auth_header)
        user = await _turn(client, auth_header, sid, "change and commit")
        _fake_turn(repo)
        _git(repo, "add", "-A")
        _git(repo, "commit", "-q", "-m", "agent committed")
        await _record_edit_tool(sid)

        refused = await client.post(
            f"/v1/sessions/{sid}/rewind",
            json={"from_seq": user["seq"], "restore_files": True},
            headers=auth_header,
        )
        assert refused.status_code == 409, refused.text
        assert "force" in refused.json()["detail"]
        kinds = [
            e["kind"]
            for e in (await client.get(f"/v1/sessions/{sid}/events", headers=auth_header)).json()[
                "events"
            ]
        ]
        assert kinds[0] == "user", "a refused restore must not truncate the transcript"
        assert (repo / "new_a.txt").exists()

        forced = await client.post(
            f"/v1/sessions/{sid}/rewind",
            json={"from_seq": user["seq"], "restore_files": True, "force": True},
            headers=auth_header,
        )
        assert forced.status_code == 200, forced.text
        assert forced.json()["checkpoint_restore"]["status"] == "restored"
        assert _tree_listing(repo) == before_tree


async def test_rewind_outside_git_notes_missing_checkpoint(
    tmp_path, checkpoints_on, client_for, auth_header
):
    plain = tmp_path / "ws"
    plain.mkdir()
    async with client_for(tmp_path) as client:
        sid = await _open(client, auth_header)
        user = await _turn(client, auth_header, sid, "no git here")
        assert "not a git repository" in user["payload"]["checkpoint"]["unavailable"]
        assert not (plain / ".orbweaver").exists()
        rewound = await client.post(
            f"/v1/sessions/{sid}/rewind",
            json={"from_seq": user["seq"], "restore_files": True},
            headers=auth_header,
        )
        assert rewound.status_code == 200, rewound.text
        note = rewound.json()["checkpoint_restore"]
        assert note["status"] == "skipped"
        assert "not a git repository" in note["note"]
        assert "no copy-based fallback" in note["note"]


def test_take_checkpoint_skips_when_disabled(tmp_path, monkeypatch):
    repo = _init_repo(tmp_path / "ws")
    monkeypatch.setattr(settings, "orbweaver_checkpoints", False)
    assert take_checkpoint(repo) is None
    monkeypatch.setattr(settings, "orbweaver_checkpoints", True)
    cp = take_checkpoint(repo)
    assert cp is not None and cp.tree
    assert take_checkpoint(tmp_path / "nope") is None


def test_restore_never_deletes_unknown_files(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "orbweaver_checkpoints", True)
    repo = _init_repo(tmp_path / "ws")
    cp = take_checkpoint(repo)
    assert cp is not None
    (repo / "build.log").write_text("ignored output\n", encoding="utf-8")
    (repo / ".orbweaver-tmp").mkdir()
    (repo / ".orbweaver-tmp" / "scratch").write_text("scratch\n", encoding="utf-8")
    (repo / "new_a.txt").write_text("a\n", encoding="utf-8")
    summary = restore_checkpoint(repo, cp.payload())
    assert summary.deleted == ["new_a.txt"]
    assert (repo / "build.log").exists()
    assert (repo / ".orbweaver-tmp" / "scratch").exists()
    with pytest.raises(CheckpointHeadMismatch):
        restore_checkpoint(repo, {"tree": cp.tree, "head": "0" * 40})


def test_prune_checkpoints_by_age_and_session(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "orbweaver_checkpoints", True)
    from orbweaver.checkpoints import pin_checkpoint

    repo = _init_repo(tmp_path / "ws")
    cp = take_checkpoint(repo)
    assert cp is not None
    sid_a, sid_b = uuid4(), uuid4()
    pin_checkpoint(cp, sid_a, 1)
    pin_checkpoint(cp, sid_b, 1)
    assert prune_checkpoints(repo, max_age_days=30) == []
    assert prune_checkpoints(repo, session_id=sid_b) == [checkpoint_ref(sid_b, 1)]
    far_future = time.time() + 40 * 86400
    assert prune_checkpoints(repo, max_age_days=30, now=far_future) == [checkpoint_ref(sid_a, 1)]
    assert _git(repo, "for-each-ref", "refs/orbweaver/") == ""


def test_turn_edits_files_rule():
    sid = uuid4()

    def ev(kind: str, **payload) -> Event:
        return Event(id=uuid4(), session_id=sid, seq=1, kind=kind, payload=payload)

    assert not turn_edits_files([ev("user", text="hi"), ev("tool_call", name="Read")])
    assert turn_edits_files([ev("tool_call", name="StrReplace")])
    assert turn_edits_files([ev("tool_call", name="Bash", input={"command": "rm x"})])
    assert not turn_edits_files([ev("tool_call", name="Grep"), ev("assistant", text="done")])
