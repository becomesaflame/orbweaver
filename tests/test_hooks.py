"""Workspace PreToolUse / PostToolUse hooks."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import anthropic
import pytest

from orbweaver.agent import agent_turn, preflight_tool, run_tools
from orbweaver.hooks import (
    apply_post_tool_use,
    load_hooks_config,
    matcher_hits,
    run_pre_tool_use,
)
from orbweaver.permissions.pipeline import TurnAborted
from orbweaver.permissions.rules import path_is_always_denied
from orbweaver.sandbox.policy import PROTECTED_WRITE_REL, load_sandbox_policy
from orbweaver.store import reset_store_for_tests
from orbweaver.workspace import LocalWorkspace


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _cmd(script: Path) -> str:
    return f"{sys.executable} {script}"


def _ctx(tmp_path: Path, *, headless=False, interactive=False, home: Path | None = None):
    ws = LocalWorkspace("workspace:default", str(tmp_path))
    root = Path(ws.root)
    cfg = load_hooks_config(root, environ={}, home=home or tmp_path / "nohome")
    store = reset_store_for_tests()
    sid = uuid4()
    return {
        "workspace": ws,
        "store": store,
        "session_id": sid,
        "workspace_kind": "local",
        "headless": headless,
        "interactive": interactive,
        "hooks_config": cfg,
        "events": [],
        "channel": "cron" if headless and not interactive else "web",
    }


class _ToolUse:
    def __init__(self, name, inp, uid="tu1"):
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
            return SimpleNamespace(content=[SimpleNamespace(type="text", text="done")])
        return self._responses.pop(0)


def test_hooks_json_is_secret_path(tmp_path: Path, monkeypatch):
    assert path_is_always_denied(".orbweaver/hooks.json")
    assert path_is_always_denied("/home/user/.orbweaver/hooks.json")
    assert not path_is_always_denied("docs/hooks.json")
    assert ".orbweaver/hooks.json" in PROTECTED_WRITE_REL
    monkeypatch.setattr("orbweaver.sandbox.policy.Path.home", lambda: tmp_path / "home")
    policy = load_sandbox_policy(tmp_path, environ={}, home=tmp_path / "home")
    denied = {p.resolve() for p in policy.deny_read}
    assert (tmp_path / "home" / ".orbweaver" / "hooks.json").resolve() in denied
    assert (tmp_path / ".orbweaver" / "hooks.json").resolve() in denied


def test_load_appends_host_then_workspace_then_extra(tmp_path: Path):
    home = tmp_path / "home"
    root = tmp_path / "ws"
    extra = tmp_path / "extra-hooks.json"
    _write_json(
        home / ".orbweaver" / "hooks.json",
        {"hooks": {"PreToolUse": [{"matcher": "Bash", "command": "echo host"}]}},
    )
    _write_json(
        root / ".orbweaver" / "hooks.json",
        {"hooks": {"PostToolUse": [{"matcher": "Read", "command": "echo workspace"}]}},
    )
    extra.write_text(
        json.dumps({"PostToolUseFailure": [{"matcher": "*", "command": "echo extra"}]}),
        encoding="utf-8",
    )
    cfg = load_hooks_config(root, environ={"ORBWEAVER_HOOKS_CONFIG": str(extra)}, home=home)
    assert [h.event for h in cfg.hooks] == ["PreToolUse", "PostToolUse", "PostToolUseFailure"]
    assert cfg.hooks[0].command == "echo host"
    assert cfg.hooks[1].command == "echo workspace"
    assert cfg.hooks[2].command == "echo extra"


def test_matcher_tool_name_and_permission_rule():
    assert matcher_hits("Bash", "Bash", {"command": "ls"})
    assert not matcher_hits("Read", "Bash", {})
    assert matcher_hits("*", "Read", {})
    assert matcher_hits("Read|Bash", "Read", {})
    assert matcher_hits("Bash(echo *)", "Bash", {"command": "echo hi"})
    assert not matcher_hits("Bash(git *)", "Bash", {"command": "echo hi"})


@pytest.mark.asyncio
async def test_pre_tool_use_script_denies_bash(tmp_path: Path):
    script = tmp_path / "deny_bash.py"
    script.write_text(
        """import json, sys
data = json.load(sys.stdin)
assert data['tool_name'] == 'Bash'
print(json.dumps({
  'hookSpecificOutput': {
    'hookEventName': 'PreToolUse',
    'permissionDecision': 'deny',
    'permissionDecisionReason': 'Bash denied by workspace hook',
  }
}))
sys.exit(2)
""",
        encoding="utf-8",
    )
    _write_json(
        tmp_path / ".orbweaver" / "hooks.json",
        {"hooks": {"PreToolUse": [{"matcher": "Bash", "command": _cmd(script)}]}},
    )
    ctx = _ctx(tmp_path)
    marker = tmp_path / "was_run"
    inp, decision = await preflight_tool("Bash", {"command": f"touch {marker}"}, ctx)
    assert decision.behavior == "deny"
    assert decision.fast_path == "hook"
    assert "denied by workspace hook" in decision.reason
    assert inp["command"].startswith("touch ")
    assert not marker.exists()


@pytest.mark.asyncio
async def test_post_tool_use_script_appends_read_line(tmp_path: Path):
    script = tmp_path / "post_read.py"
    script.write_text("print('hook-annotated-read')\n", encoding="utf-8")
    _write_json(
        tmp_path / ".orbweaver" / "hooks.json",
        {"hooks": {"PostToolUse": [{"matcher": "Read", "command": _cmd(script)}]}},
    )
    ctx = _ctx(tmp_path)
    (tmp_path / "note.txt").write_text("hello hooks\n", encoding="utf-8")
    raw = await run_tools("Read", {"path": "note.txt"}, ctx)
    out = await apply_post_tool_use("Read", {"path": "note.txt"}, raw, ctx)
    assert "hello hooks" in out
    assert "hook-annotated-read" in out
    assert out.index("hello hooks") < out.index("hook-annotated-read")


@pytest.mark.asyncio
async def test_post_tool_use_failure_on_read_error(tmp_path: Path):
    script = tmp_path / "on_fail.py"
    script.write_text("print('hook-saw-failure')\n", encoding="utf-8")
    _write_json(
        tmp_path / ".orbweaver" / "hooks.json",
        {"hooks": {"PostToolUseFailure": [{"matcher": "Read", "command": _cmd(script)}]}},
    )
    ctx = _ctx(tmp_path)
    raw = await run_tools("Read", {"path": "missing.txt"}, ctx)
    assert raw.lower().startswith("error")
    out = await apply_post_tool_use("Read", {"path": "missing.txt"}, raw, ctx)
    assert "hook-saw-failure" in out


@pytest.mark.asyncio
async def test_pre_tool_use_rewrites_input_before_permissions(tmp_path: Path, monkeypatch):
    script = tmp_path / "rewrite.py"
    script.write_text(
        """import json, sys
data = json.load(sys.stdin)
inp = dict(data['tool_input'])
inp['path'] = 'safe.txt'
print(json.dumps({'hookSpecificOutput': {'updatedInput': inp}}))
""",
        encoding="utf-8",
    )
    _write_json(
        tmp_path / ".orbweaver" / "hooks.json",
        {"hooks": {"PreToolUse": [{"matcher": "Read", "command": _cmd(script)}]}},
    )
    seen: dict = {}

    async def capture(name, inp, ctx):
        seen["name"] = name
        seen["path"] = inp.get("path")
        from orbweaver.permissions.pipeline import PermissionDecision

        return PermissionDecision("allow", "ok", "allowlist")

    monkeypatch.setattr("orbweaver.agent.can_use_tool", capture)
    ctx = _ctx(tmp_path)
    inp, decision = await preflight_tool("Read", {"path": ".env"}, ctx)
    assert inp["path"] == "safe.txt"
    assert seen["path"] == "safe.txt"
    assert decision.behavior == "allow"


@pytest.mark.asyncio
async def test_pre_tool_use_cancel_aborts_turn(tmp_path: Path):
    script = tmp_path / "cancel.py"
    script.write_text(
        'print(\'{"continue": false, "stopReason": "hook cancelled the turn"}\')\n',
        encoding="utf-8",
    )
    _write_json(
        tmp_path / ".orbweaver" / "hooks.json",
        {"hooks": {"PreToolUse": [{"matcher": "Bash", "command": _cmd(script)}]}},
    )
    ctx = _ctx(tmp_path)
    with pytest.raises(TurnAborted) as ei:
        await preflight_tool("Bash", {"command": "echo hi"}, ctx)
    assert ei.value.payload["reason"] == "hook_cancel"
    assert "cancelled" in ei.value.message.lower()


@pytest.mark.asyncio
async def test_prompt_hook_fails_closed_headless(tmp_path: Path):
    _write_json(
        tmp_path / ".orbweaver" / "hooks.json",
        {
            "hooks": {
                "PreToolUse": [
                    {"matcher": "Bash", "type": "prompt", "prompt": "approve this bash?"}
                ]
            }
        },
    )
    ctx = _ctx(tmp_path, headless=True)
    pre = await run_pre_tool_use("Bash", {"command": "echo hi"}, ctx)
    assert pre.action == "deny"
    assert "fail closed" in pre.reason


@pytest.mark.asyncio
async def test_tty_read_hook_does_not_hang_headless(tmp_path: Path):
    script = tmp_path / "tty.py"
    script.write_text(
        """import json, sys
try:
    open('/dev/tty').readline()
    print(json.dumps({'permissionDecision': 'allow'}))
except OSError as e:
    print(json.dumps({
      'hookSpecificOutput': {
        'permissionDecision': 'deny',
        'permissionDecisionReason': f'no tty: {e}',
      }
    }))
    sys.exit(2)
""",
        encoding="utf-8",
    )
    _write_json(
        tmp_path / ".orbweaver" / "hooks.json",
        {
            "hooks": {
                "PreToolUse": [{"matcher": "Bash", "command": _cmd(script), "timeout": 2}]
            }
        },
    )
    ctx = _ctx(tmp_path, headless=True)
    _inp, decision = await preflight_tool("Bash", {"command": "echo hang"}, ctx)
    assert decision.behavior == "deny"
    assert decision.fast_path == "hook"


@pytest.mark.asyncio
async def test_agent_turn_pre_hook_blocks_bash(tmp_path: Path, monkeypatch):
    script = tmp_path / "deny_bash.py"
    script.write_text(
        "import json, sys\n"
        "print(json.dumps({'hookSpecificOutput': {"
        "'permissionDecision': 'deny',"
        "'permissionDecisionReason': 'Bash denied by workspace hook'}}))\n"
        "sys.exit(2)\n",
        encoding="utf-8",
    )
    _write_json(
        tmp_path / ".orbweaver" / "hooks.json",
        {"hooks": {"PreToolUse": [{"matcher": "Bash", "command": _cmd(script)}]}},
    )
    from orbweaver.config import settings

    monkeypatch.setattr("orbweaver.hooks.Path.home", lambda: tmp_path / "nohome")
    monkeypatch.setattr(settings, "anthropic_api_key", "sk-test")
    resp = SimpleNamespace(content=[_ToolUse("Bash", {"command": "touch was_run"})])

    def factory(*a, **k):
        return _RecordingAnthropic([resp], *a, **k)

    monkeypatch.setattr(anthropic, "AsyncAnthropic", factory)
    store = reset_store_for_tests()
    sid = uuid4()
    ws = LocalWorkspace("workspace:default", str(tmp_path))
    events = await agent_turn(store, sid, "run it", ws, headless=False)
    kinds = [e.kind for e in events]
    assert "permission_decision" in kinds
    decision = next(e for e in events if e.kind == "permission_decision")
    assert decision.payload["behavior"] == "deny"
    assert decision.payload["fast_path"] == "hook"
    result = next(e for e in events if e.kind == "tool_result")
    assert "workspace hook" in result.payload["content"]
    assert not (tmp_path / "was_run").exists()
