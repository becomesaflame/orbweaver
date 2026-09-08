from __future__ import annotations

import os
import subprocess
from pathlib import Path
from uuid import uuid4

import pytest
from orbweaver.agent import run_tools, static_system
from orbweaver.git_ritual import (
    ADD_ALL_WARNING,
    DETACHED_WARNING,
    REBASE_WARNING,
    RITUAL_MARK,
    annotate_bash_output,
    command_uses_git,
    repo_for_command,
)
from orbweaver.store import reset_store_for_tests
from orbweaver.workspace import LocalWorkspace

_GIT_IDENT = {
    **os.environ,
    "GIT_AUTHOR_NAME": "Orbweaver Test",
    "GIT_AUTHOR_EMAIL": "test@orbweaver.local",
    "GIT_COMMITTER_NAME": "Orbweaver Test",
    "GIT_COMMITTER_EMAIL": "test@orbweaver.local",
    "GIT_TERMINAL_PROMPT": "0",
}


def _git(cwd: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(cwd), *args],
        capture_output=True,
        text=True,
        check=check,
        env=_GIT_IDENT,
    )


def _init_repo(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    _git(path, "init", "-b", "main")
    _git(path, "config", "user.name", "Orbweaver Test")
    _git(path, "config", "user.email", "test@orbweaver.local")
    (path / "f.txt").write_text("a\n", encoding="utf-8")
    _git(path, "add", "f.txt")
    _git(path, "commit", "-m", "base")
    return path


def test_command_uses_git():
    assert command_uses_git("git status")
    assert command_uses_git("cd repo && git rebase origin/main")
    assert command_uses_git("true; git push origin feat")
    assert not command_uses_git("echo hi")
    assert not command_uses_git("echo git")
    assert not command_uses_git("")


def test_static_system_teaches_rebase_continue():
    text = static_system()
    assert "rebase --continue" in text
    assert "Everything up-to-date" in text
    assert "git add -A" in text


def test_bash_tool_mentions_git_status_footer():
    from orbweaver.agent import TOOL_SPEC

    bash = next(t for t in TOOL_SPEC if t["name"] == "Bash")
    assert "rebase-in-progress" in bash["description"].lower()


def test_annotate_skips_non_git(tmp_path: Path):
    _init_repo(tmp_path)
    assert annotate_bash_output(tmp_path, "echo hi", "hi") == "hi"


def test_annotate_appends_status(tmp_path: Path):
    _init_repo(tmp_path)
    out = annotate_bash_output(tmp_path, "git push origin main", "Everything up-to-date")
    assert "Everything up-to-date" in out
    assert RITUAL_MARK in out
    assert "branch=" in out
    assert "HEAD=" in out


def test_annotate_follows_cd_into_nested_repo(tmp_path: Path):
    nested = _init_repo(tmp_path / "orbweaver")
    out = annotate_bash_output(
        tmp_path,
        f"cd {nested} && git rebase origin/main",
        "CONFLICT (content)",
    )
    assert RITUAL_MARK in out
    assert str(nested.resolve()) in out
    assert repo_for_command(tmp_path, f"cd {nested} && git status") == nested.resolve()


def test_rebase_in_progress_and_detached_head(tmp_path: Path):
    repo = _init_repo(tmp_path)
    _git(repo, "checkout", "-b", "feat")
    (repo / "f.txt").write_text("b\n", encoding="utf-8")
    _git(repo, "commit", "-am", "feat")
    _git(repo, "checkout", "main")
    (repo / "f.txt").write_text("c\n", encoding="utf-8")
    _git(repo, "commit", "-am", "main-side")
    _git(repo, "checkout", "feat")
    conflict = _git(repo, "rebase", "main", check=False)
    assert conflict.returncode != 0

    out = annotate_bash_output(
        repo,
        "git add -A && git commit -m done && git push origin feat",
        "[detached HEAD abc123] done\nEverything up-to-date",
    )
    assert "Everything up-to-date" in out
    assert REBASE_WARNING.split(":")[0] in out
    assert DETACHED_WARNING.split(":")[0] in out
    assert ADD_ALL_WARNING[:20] in out


@pytest.mark.asyncio
async def test_run_tools_bash_git_gets_ritual(tmp_path: Path, monkeypatch):
    from orbweaver.config import settings

    monkeypatch.setattr(settings, "orbweaver_sandbox", False)
    _init_repo(tmp_path)
    ws = LocalWorkspace("workspace:default", str(tmp_path))
    ctx = {"workspace": ws, "store": reset_store_for_tests(), "session_id": uuid4()}
    result = await run_tools("Bash", {"command": "git status --short --branch"}, ctx)
    assert RITUAL_MARK in result
    assert "branch=" in result


@pytest.mark.asyncio
async def test_run_tools_bash_echo_has_no_ritual(tmp_path: Path, monkeypatch):
    from orbweaver.config import settings

    monkeypatch.setattr(settings, "orbweaver_sandbox", False)
    ws = LocalWorkspace("workspace:default", str(tmp_path))
    ctx = {"workspace": ws, "store": reset_store_for_tests(), "session_id": uuid4()}
    result = await run_tools("Bash", {"command": "echo hi"}, ctx)
    assert result.strip() == "hi"
    assert RITUAL_MARK not in result
