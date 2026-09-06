from pathlib import Path
from uuid import uuid4

import pytest
from orbweaver.config import settings
from orbweaver.permissions.denial import DenialTrackingState, reset_denial_states
from orbweaver.permissions.pipeline import TurnAborted, can_use_tool
from orbweaver.permissions.rules import is_critical_rm, path_is_always_denied
from orbweaver.store import Event
from orbweaver.workspace import DockerWorkspace, LocalWorkspace


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
            {"command": "git push origin feature"},
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


@pytest.mark.asyncio
async def test_in_project_write_docker_skips_classifier(tmp_path, monkeypatch):
    async def boom(*_a, **_k):
        raise AssertionError("classifier should not run")

    monkeypatch.setattr("orbweaver.permissions.pipeline.classify_action", boom)
    ctx = _ctx(tmp_path, kind="docker")
    ctx["workspace"] = DockerWorkspace("workspace:default", str(tmp_path))
    decision = await can_use_tool("Write", {"path": "src/a.py", "content": "x"}, ctx)
    assert decision.behavior == "allow"
    assert decision.fast_path == "acceptEdits"
