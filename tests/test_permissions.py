from pathlib import Path
from uuid import uuid4

import pytest

from orbweaver.config import settings
from orbweaver.permissions.denial import DenialTrackingState, reset_denial_states
from orbweaver.permissions.pipeline import TurnAborted, can_use_tool
from orbweaver.permissions.rules import (
    in_working_set,
    is_critical_rm,
    is_protected_git_push,
    path_is_always_denied,
)
from orbweaver.store import Event
from orbweaver.workspace import LocalWorkspace


def _ctx(tmp_path: Path, *, headless=False, kind="local"):
    reset_denial_states()
    sid = uuid4()
    ws = LocalWorkspace("workspace:default", str(tmp_path))
    return {
        "workspace": ws,
        "workspace_kind": kind,
        "headless": headless,
        "session_id": sid,
        "denial_state": DenialTrackingState(),
        "events": [
            Event(id=uuid4(), session_id=sid, seq=1, kind="user", payload={"text": "please help"})
        ],
    }


@pytest.mark.asyncio
async def test_allowlist_skips_classifier(tmp_path, monkeypatch):
    async def boom(*_a, **_k):
        raise AssertionError("classifier should not run")

    monkeypatch.setattr("orbweaver.permissions.pipeline.classify_action", boom)
    decision = await can_use_tool("Read", {"path": "src/a.py"}, _ctx(tmp_path))
    assert decision.behavior == "allow"
    assert decision.fast_path == "allowlist"


@pytest.mark.asyncio
async def test_websearch_allowlisted(tmp_path, monkeypatch):
    async def boom(*_a, **_k):
        raise AssertionError("classifier should not run")

    monkeypatch.setattr("orbweaver.permissions.pipeline.classify_action", boom)
    decision = await can_use_tool("WebSearch", {"query": "agent max_turns"}, _ctx(tmp_path))
    assert decision.behavior == "allow"
    assert decision.fast_path == "allowlist"


@pytest.mark.asyncio
async def test_send_photo_allowlisted(tmp_path, monkeypatch):
    async def boom(*_a, **_k):
        raise AssertionError("classifier should not run")

    monkeypatch.setattr("orbweaver.permissions.pipeline.classify_action", boom)
    decision = await can_use_tool("SendPhoto", {"path": "attachments/a.jpg"}, _ctx(tmp_path))
    assert decision.behavior == "allow"
    assert decision.fast_path == "allowlist"


@pytest.mark.asyncio
async def test_send_photo_denies_env(tmp_path, monkeypatch):
    async def boom(*_a, **_k):
        raise AssertionError("classifier should not run")

    monkeypatch.setattr("orbweaver.permissions.pipeline.classify_action", boom)
    decision = await can_use_tool("SendPhoto", {"path": ".env"}, _ctx(tmp_path))
    assert decision.behavior == "deny"


@pytest.mark.asyncio
async def test_in_project_delete_skips_classifier(tmp_path, monkeypatch):
    async def boom(*_a, **_k):
        raise AssertionError("classifier should not run")

    monkeypatch.setattr("orbweaver.permissions.pipeline.classify_action", boom)
    decision = await can_use_tool("Delete", {"path": "src/a.py"}, _ctx(tmp_path))
    assert decision.behavior == "allow"
    assert decision.fast_path == "acceptEdits"


@pytest.mark.asyncio
async def test_delete_env_denied(tmp_path, monkeypatch):
    async def boom(*_a, **_k):
        raise AssertionError("classifier should not run")

    monkeypatch.setattr("orbweaver.permissions.pipeline.classify_action", boom)
    decision = await can_use_tool("Delete", {"path": ".env"}, _ctx(tmp_path))
    assert decision.behavior == "deny"


@pytest.mark.asyncio
async def test_delete_outside_working_set_denied(tmp_path, monkeypatch):
    async def boom(*_a, **_k):
        raise AssertionError("classifier should not run")

    monkeypatch.setattr("orbweaver.permissions.pipeline.classify_action", boom)
    decision = await can_use_tool("Delete", {"path": "/etc/passwd"}, _ctx(tmp_path))
    assert decision.behavior == "deny"


@pytest.mark.asyncio
async def test_todowrite_allowlisted(tmp_path, monkeypatch):
    async def boom(*_a, **_k):
        raise AssertionError("classifier should not run")

    monkeypatch.setattr("orbweaver.permissions.pipeline.classify_action", boom)
    decision = await can_use_tool(
        "TodoWrite",
        {"todos": [{"id": "1", "content": "ship", "status": "pending"}]},
        _ctx(tmp_path),
    )
    assert decision.behavior == "allow"
    assert decision.fast_path == "allowlist"


@pytest.mark.asyncio
async def test_readlints_allowlisted(tmp_path, monkeypatch):
    async def boom(*_a, **_k):
        raise AssertionError("classifier should not run")

    monkeypatch.setattr("orbweaver.permissions.pipeline.classify_action", boom)
    decision = await can_use_tool("ReadLints", {"paths": ["src/a.py"]}, _ctx(tmp_path))
    assert decision.behavior == "allow"
    assert decision.fast_path == "allowlist"


@pytest.mark.asyncio
async def test_in_project_write_skips_classifier(tmp_path, monkeypatch):
    async def boom(*_a, **_k):
        raise AssertionError("classifier should not run")

    monkeypatch.setattr("orbweaver.permissions.pipeline.classify_action", boom)
    decision = await can_use_tool("Write", {"path": "src/a.py", "content": "x"}, _ctx(tmp_path))
    assert decision.behavior == "allow"
    assert decision.fast_path == "acceptEdits"


@pytest.mark.asyncio
async def test_git_push_ask_rule_never_skips(tmp_path, monkeypatch):
    async def boom(*_a, **_k):
        raise AssertionError("classifier should not auto-approve ask rules")

    monkeypatch.setattr("orbweaver.permissions.pipeline.classify_action", boom)
    monkeypatch.setattr("orbweaver.permissions.pipeline.sandbox_available", lambda: True)
    decision = await can_use_tool("Bash", {"command": "git push origin main"}, _ctx(tmp_path))
    assert decision.behavior == "ask"
    assert decision.fast_path == "ask_rule"


@pytest.mark.asyncio
async def test_git_push_ask_headless_aborts(tmp_path, monkeypatch):
    monkeypatch.setattr("orbweaver.permissions.pipeline.sandbox_available", lambda: True)
    with pytest.raises(TurnAborted) as ei:
        await can_use_tool(
            "Bash",
            {"command": "git push origin main"},
            _ctx(tmp_path, headless=True),
        )
    assert "Stopped this turn" in ei.value.message
    assert ei.value.payload["reason"] == "ask_required_headless"


@pytest.mark.asyncio
async def test_sandboxed_bash_auto_allow(tmp_path, monkeypatch):
    async def boom(*_a, **_k):
        raise AssertionError("classifier should not run for sandboxed bash")

    monkeypatch.setattr("orbweaver.permissions.pipeline.classify_action", boom)
    monkeypatch.setattr("orbweaver.permissions.pipeline.sandbox_available", lambda: True)
    decision = await can_use_tool("Bash", {"command": "pytest -q"}, _ctx(tmp_path))
    assert decision.behavior == "allow"
    assert decision.fast_path == "sandbox"


@pytest.mark.asyncio
async def test_sandbox_unavailable_fail_closed(tmp_path, monkeypatch):
    monkeypatch.setattr("orbweaver.permissions.pipeline.sandbox_available", lambda: False)
    monkeypatch.setattr(settings, "orbweaver_sandbox_fail_if_unavailable", True)

    async def boom(*_a, **_k):
        raise AssertionError("should deny before classifier")

    monkeypatch.setattr("orbweaver.permissions.pipeline.classify_action", boom)
    decision = await can_use_tool("Bash", {"command": "echo hi"}, _ctx(tmp_path))
    assert decision.behavior == "deny"
    assert "sandbox_unavailable" in decision.reason


@pytest.mark.asyncio
async def test_env_path_denied(tmp_path, monkeypatch):
    async def boom(*_a, **_k):
        raise AssertionError("classifier should not run")

    monkeypatch.setattr("orbweaver.permissions.pipeline.classify_action", boom)
    decision = await can_use_tool("Write", {"path": ".env", "content": "x"}, _ctx(tmp_path))
    assert decision.behavior == "deny"


@pytest.mark.asyncio
async def test_feature_branch_push_is_sandboxed(tmp_path, monkeypatch):
    async def boom(*_a, **_k):
        raise AssertionError("classifier should not run for feature-branch push")

    monkeypatch.setattr("orbweaver.permissions.pipeline.classify_action", boom)
    monkeypatch.setattr("orbweaver.permissions.pipeline.sandbox_available", lambda: True)
    decision = await can_use_tool(
        "Bash",
        {"command": "git push -u origin feature/telegram-local-bash"},
        _ctx(tmp_path, headless=True),
    )
    assert decision.behavior == "allow"
    assert decision.fast_path == "sandbox"


@pytest.mark.asyncio
async def test_legacy_git_push_glob_skips_feature_branch(tmp_path, monkeypatch):
    async def boom(*_a, **_k):
        raise AssertionError("legacy git push glob must not reach classifier")

    monkeypatch.setattr("orbweaver.permissions.pipeline.classify_action", boom)
    monkeypatch.setattr("orbweaver.permissions.pipeline.sandbox_available", lambda: True)
    monkeypatch.setattr(settings, "orbweaver_permission_ask", "Bash(git push *)")
    decision = await can_use_tool(
        "Bash",
        {"command": "git push origin HEAD"},
        _ctx(tmp_path, headless=True),
    )
    assert decision.behavior == "allow"
    assert decision.fast_path == "sandbox"


@pytest.mark.asyncio
async def test_spawn_subagent_not_allowlisted(tmp_path, monkeypatch):
    async def deny(*_a, **_k):
        return {"should_block": True, "reason": "unauthorized spawn", "stage": "thinking"}

    monkeypatch.setattr("orbweaver.permissions.pipeline.classify_action", deny)
    decision = await can_use_tool("SpawnSubagent", {"task": "wipe remotes"}, _ctx(tmp_path))
    assert decision.behavior == "deny"
    assert decision.fast_path == "handoff"


@pytest.mark.asyncio
async def test_denial_limit_headless_aborts(tmp_path, monkeypatch):
    async def deny(*_a, **_k):
        return {"should_block": True, "reason": "nope", "stage": "fast"}

    monkeypatch.setattr("orbweaver.permissions.pipeline.classify_action", deny)
    monkeypatch.setattr("orbweaver.permissions.pipeline.sandbox_available", lambda: False)
    monkeypatch.setattr(settings, "orbweaver_sandbox_fail_if_unavailable", False)
    ctx = _ctx(tmp_path, headless=True)
    for _ in range(2):
        d = await can_use_tool("WebFetch", {"url": "https://example.com"}, ctx)
        assert d.behavior == "deny"
    with pytest.raises(TurnAborted) as ei:
        await can_use_tool("WebFetch", {"url": "https://example.com"}, ctx)
    assert ei.value.payload["reason"] == "classifier_denial_limit"
    assert "Stopped this turn" in ei.value.message


@pytest.mark.asyncio
async def test_denial_success_resets_consecutive(tmp_path, monkeypatch):
    n = {"i": 0}

    async def flip(*_a, **_k):
        n["i"] += 1
        if n["i"] <= 2:
            return {"should_block": True, "reason": "nope", "stage": "fast"}
        return {"should_block": False, "reason": "ok", "stage": "fast"}

    monkeypatch.setattr("orbweaver.permissions.pipeline.classify_action", flip)
    monkeypatch.setattr("orbweaver.permissions.pipeline.sandbox_available", lambda: False)
    monkeypatch.setattr(settings, "orbweaver_sandbox_fail_if_unavailable", False)
    ctx = _ctx(tmp_path, headless=True)
    await can_use_tool("WebFetch", {"url": "https://a.example"}, ctx)
    await can_use_tool("WebFetch", {"url": "https://b.example"}, ctx)
    assert ctx["denial_state"].consecutive_denials == 2
    d = await can_use_tool("WebFetch", {"url": "https://c.example"}, ctx)
    assert d.behavior == "allow"
    assert ctx["denial_state"].consecutive_denials == 0


def test_critical_rm_and_deny_names():
    assert is_critical_rm("rm -rf /")
    assert is_critical_rm("rm -rf $HOME")
    assert not is_critical_rm("rm -rf ./build")
    assert path_is_always_denied(".env")
    assert path_is_always_denied(".ssh/id_rsa")
    assert is_protected_git_push("git push origin main")
    assert is_protected_git_push("git push -u origin master")
    assert is_protected_git_push("git push origin HEAD:main")
    assert is_protected_git_push("git push origin feature:main")
    assert is_protected_git_push("git push --force origin feature")
    assert is_protected_git_push("git push -f origin foo")
    assert is_protected_git_push("git push origin +main")
    assert not is_protected_git_push("git push origin feature/telegram-local-bash")
    assert not is_protected_git_push("git push -u origin HEAD")
    assert not is_protected_git_push("pytest -q")


@pytest.mark.asyncio
async def test_unsandboxed_and_permissions_all_use_classifier(tmp_path, monkeypatch):
    seen = {}

    async def classify(*_a, **kwargs):
        seen["ran"] = True
        return {"should_block": False, "reason": "ok", "stage": "fast"}

    monkeypatch.setattr("orbweaver.permissions.pipeline.classify_action", classify)
    monkeypatch.setattr("orbweaver.permissions.pipeline.sandbox_available", lambda: True)
    d1 = await can_use_tool("Bash", {"command": "journalctl --user -n 1"}, _ctx(tmp_path))
    assert d1.fast_path == "sandbox"
    assert "ran" not in seen
    d2 = await can_use_tool(
        "Bash",
        {"command": "docker ps", "permissions": ["all"]},
        _ctx(tmp_path),
    )
    assert d2.behavior == "allow"
    assert d2.fast_path == "classifier"
    d3 = await can_use_tool(
        "Bash",
        {"command": "curl https://example.com", "permissions": ["full_network"]},
        _ctx(tmp_path),
    )
    assert d3.fast_path == "classifier"
    d4 = await can_use_tool("Bash", {"command": "id", "unsandboxed": True}, _ctx(tmp_path))
    assert d4.fast_path == "classifier"


@pytest.mark.asyncio
async def test_read_outside_working_set_is_classified(tmp_path, monkeypatch):
    async def classify(*_a, **_k):
        return {"should_block": False, "reason": "host read ok", "stage": "fast"}

    monkeypatch.setattr("orbweaver.permissions.pipeline.classify_action", classify)
    decision = await can_use_tool("Read", {"path": "/var/log/syslog"}, _ctx(tmp_path))
    assert decision.behavior == "allow"
    assert decision.fast_path == "classifier"


@pytest.mark.asyncio
async def test_read_in_workspace_still_allowlisted(tmp_path, monkeypatch):
    async def boom(*_a, **_k):
        raise AssertionError("classifier should not run")

    monkeypatch.setattr("orbweaver.permissions.pipeline.classify_action", boom)
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("x", encoding="utf-8")
    decision = await can_use_tool("Read", {"path": "src/a.py"}, _ctx(tmp_path))
    assert decision.fast_path == "allowlist"


def test_in_working_set_relative(tmp_path):
    ws = LocalWorkspace("workspace:default", str(tmp_path))
    assert in_working_set("src/a.py", ws)
    assert in_working_set("src/a.py", ws, write=True)
    assert not in_working_set("/etc/passwd", ws, write=True)


@pytest.mark.asyncio
async def test_classifier_ask_is_not_a_deny(tmp_path, monkeypatch):
    async def ask(*_a, **_k):
        return {"verdict": "ask", "should_block": False, "should_ask": True, "reason": "confirm override", "stage": "thinking"}

    monkeypatch.setattr("orbweaver.permissions.pipeline.classify_action", ask)
    monkeypatch.setattr("orbweaver.permissions.pipeline.sandbox_available", lambda: True)
    decision = await can_use_tool(
        "Bash",
        {"command": "git clone git@github.com:becomesaflame/orbweaver.git .", "permissions": ["all"]},
        _ctx(tmp_path),
    )
    assert decision.behavior == "ask"
    assert "confirm override" in decision.reason


@pytest.mark.asyncio
async def test_classifier_ask_headless_aborts(tmp_path, monkeypatch):
    async def ask(*_a, **_k):
        return {"verdict": "ask", "should_block": False, "should_ask": True, "reason": "confirm override", "stage": "thinking"}

    monkeypatch.setattr("orbweaver.permissions.pipeline.classify_action", ask)
    monkeypatch.setattr("orbweaver.permissions.pipeline.sandbox_available", lambda: False)
    monkeypatch.setattr(settings, "orbweaver_sandbox_fail_if_unavailable", False)
    with pytest.raises(TurnAborted) as ei:
        await can_use_tool("WebFetch", {"url": "https://example.com"}, _ctx(tmp_path, headless=True))
    assert ei.value.payload["reason"] == "ask_required_headless"
