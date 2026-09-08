from __future__ import annotations

import os
import subprocess
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import anthropic
import pytest

from orbweaver.agent import GIT_NOT_DONE_NUDGE, agent_turn, run_tools, static_system
from orbweaver.git_ritual import (
    ADD_ALL_WARNING,
    DETACHED_WARNING,
    NOT_DONE_WARNING,
    REBASE_WARNING,
    RITUAL_MARK,
    STAGED_TMP_WARNING,
    annotate_bash_output,
    command_uses_git,
    escalate_lines,
    repo_for_command,
    ritual_says_not_done,
)
from orbweaver.permissions.pipeline import PermissionDecision
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


def _start_conflicted_rebase(path: Path) -> Path:
    repo = _init_repo(path)
    _git(repo, "checkout", "-b", "feat")
    (repo / "f.txt").write_text("b\n", encoding="utf-8")
    _git(repo, "commit", "-am", "feat")
    _git(repo, "checkout", "main")
    (repo / "f.txt").write_text("c\n", encoding="utf-8")
    _git(repo, "commit", "-am", "main-side")
    _git(repo, "checkout", "feat")
    conflict = _git(repo, "rebase", "main", check=False)
    assert conflict.returncode != 0
    return repo


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
    assert "NOT_DONE" in text
    assert "StrReplace" in text


def test_escalate_lines_are_mechanical():
    assert NOT_DONE_WARNING.split(":")[0] in escalate_lines(
        "git commit -m done", rebase=True, detached=True, status=""
    )[0]
    assert escalate_lines(
        "git rebase --continue", rebase=True, detached=True, status=""
    ) == []
    assert escalate_lines("git status", rebase=True, detached=True, status="") == []
    extra = escalate_lines(
        "git add -A", rebase=False, detached=False, status="?? .orbweaver-tmp/"
    )
    assert ADD_ALL_WARNING in extra
    assert STAGED_TMP_WARNING in extra
    assert ritual_says_not_done(NOT_DONE_WARNING)
    assert not ritual_says_not_done("Everything up-to-date")


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
    assert not ritual_says_not_done(out)


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
    repo = _start_conflicted_rebase(tmp_path)
    out = annotate_bash_output(
        repo,
        "git add -A && git commit -m done && git push origin feat",
        "[detached HEAD abc123] done\nEverything up-to-date",
    )
    assert "Everything up-to-date" in out
    assert REBASE_WARNING.split(":")[0] in out
    assert DETACHED_WARNING.split(":")[0] in out
    assert ADD_ALL_WARNING[:20] in out
    assert NOT_DONE_WARNING.split(":")[0] in out
    assert ritual_says_not_done(out)


def test_git_status_during_rebase_does_not_escalate(tmp_path: Path):
    repo = _start_conflicted_rebase(tmp_path)
    out = annotate_bash_output(repo, "git status --short --branch", "ok")
    assert REBASE_WARNING.split(":")[0] in out
    assert not ritual_says_not_done(out)

    cont = annotate_bash_output(repo, "git rebase --continue", "error")
    assert not ritual_says_not_done(cont)


def test_push_on_detached_head_escalates(tmp_path: Path):
    repo = _init_repo(tmp_path)
    _git(repo, "checkout", "--detach")
    out = annotate_bash_output(repo, "git push origin main", "Everything up-to-date")
    assert DETACHED_WARNING.split(":")[0] in out
    assert ritual_says_not_done(out)


def test_add_all_mentions_orbweaver_tmp(tmp_path: Path):
    repo = _init_repo(tmp_path)
    junk = repo / ".orbweaver-tmp" / "ow-ssh"
    junk.mkdir(parents=True)
    (junk / "key").write_text("secret\n", encoding="utf-8")
    out = annotate_bash_output(repo, "git add -A", "ok")
    assert ADD_ALL_WARNING[:20] in out
    assert ".orbweaver-tmp" in out
    assert STAGED_TMP_WARNING[:20] in out


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


class _ToolUse:
    def __init__(self, name, inp, uid="tu-git"):
        self.type = "tool_use"
        self.id = uid
        self.name = name
        self.input = inp


class _RecordingAnthropic:
    def __init__(self, responses, *a, **k):
        self._responses = list(responses)
        self.calls = []
        self.messages = self

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if not self._responses:
            return SimpleNamespace(
                content=[SimpleNamespace(type="text", text="rebase still unfinished")]
            )
        return self._responses.pop(0)


@pytest.mark.asyncio
async def test_commit_during_rebase_refuses_success_conclude(tmp_path: Path, monkeypatch):
    from orbweaver.config import settings

    monkeypatch.setattr(settings, "orbweaver_sandbox", False)
    monkeypatch.setattr(settings, "anthropic_api_key", "sk-test")

    async def allow(*_a, **_k):
        return PermissionDecision("allow", "test", "test")

    monkeypatch.setattr("orbweaver.agent.can_use_tool", allow)

    async def no_compact(*_a, **_k):
        return None

    async def no_probe(_name, output, **_k):
        return {"flagged": False, "output": output}

    monkeypatch.setattr("orbweaver.agent.maybe_compact", no_compact)
    monkeypatch.setattr("orbweaver.agent.probe_tool_output", no_probe)
    _start_conflicted_rebase(tmp_path)
    commit = SimpleNamespace(
        content=[_ToolUse("Bash", {"command": "git commit -m done --allow-empty"})]
    )
    premature = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="PR updated, Everything up-to-date")]
    )
    client = _RecordingAnthropic([commit, premature])

    def factory(*a, **k):
        return client

    monkeypatch.setattr(anthropic, "AsyncAnthropic", factory)
    store = reset_store_for_tests()
    sid = uuid4()
    ws = LocalWorkspace("workspace:default", str(tmp_path))
    events = await agent_turn(store, sid, "rebase then push", ws, max_rounds=4)
    texts = [e.payload.get("text") for e in events if e.kind == "assistant"]
    assert any("PR updated" in (t or "") for t in texts)
    assert any("rebase still unfinished" in (t or "") for t in texts)
    assert len(client.calls) >= 3
    nudged = str(client.calls[2]["messages"][-1]["content"])
    assert "NOT_DONE" in nudged
    assert "rebase --continue" in GIT_NOT_DONE_NUDGE
    tool_out = next(
        e.payload.get("content") or ""
        for e in events
        if e.kind == "tool_result"
    )
    assert ritual_says_not_done(tool_out)
    assert any("rebase still unfinished" in (t or "") for t in texts)
    assert len(client.calls) >= 3
    nudged = str(client.calls[2]["messages"][-1]["content"])
    assert "NOT_DONE" in nudged
    assert "rebase --continue" in GIT_NOT_DONE_NUDGE
    tool_out = next(
        e.payload.get("content") or ""
        for e in events
        if e.kind == "tool_result"
    )
    assert ritual_says_not_done(tool_out)
    assert any("rebase still unfinished" in (t or "") for t in texts)
    assert len(client.calls) >= 3
    nudged = str(client.calls[2]["messages"][-1]["content"])
    assert "NOT_DONE" in nudged
    assert "rebase --continue" in GIT_NOT_DONE_NUDGE
    tool_out = next(
        e.payload.get("content") or ""
        for e in events
        if e.kind == "tool_result"
    )
    assert ritual_says_not_done(tool_out)
