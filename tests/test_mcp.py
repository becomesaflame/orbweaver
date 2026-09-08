"""MCP client: config merge, fake stdio server, permission pipeline."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from uuid import uuid4

import pytest
from orbweaver.agent import TOOL_SPEC, run_tools, session_tools
from orbweaver.mcp import call_mcp_tool, load_mcp_config, mcp_tool_specs, reset_mcp_sessions
from orbweaver.mcp.tools import exposed_tool_name
from orbweaver.permissions.pipeline import can_use_tool
from orbweaver.permissions.rules import SAFE_ALLOWLIST, path_is_always_denied
from orbweaver.sandbox.policy import PROTECTED_WRITE_REL
from orbweaver.workspace import LocalWorkspace

FAKE_SERVER = Path(__file__).resolve().parent / "mcp_fake_server.py"


def _server_spec(env: dict[str, str] | None = None) -> dict:
    spec = {
        "command": sys.executable,
        "args": [str(FAKE_SERVER)],
    }
    if env:
        spec["env"] = env
    return spec


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


@pytest.fixture
async def mcp_clean():
    await reset_mcp_sessions()
    yield
    await reset_mcp_sessions()


def test_mcp_json_is_secret_path(tmp_path: Path, monkeypatch):
    assert path_is_always_denied(".orbweaver/mcp.json")
    assert path_is_always_denied("/home/user/.orbweaver/mcp.json")
    assert not path_is_always_denied("docs/mcp.json")
    assert ".orbweaver/mcp.json" in PROTECTED_WRITE_REL
    from orbweaver.sandbox.policy import load_sandbox_policy

    monkeypatch.setattr("orbweaver.sandbox.policy.Path.home", lambda: tmp_path / "home")
    policy = load_sandbox_policy(tmp_path, environ={}, home=tmp_path / "home")
    denied = {p.resolve() for p in policy.deny_read}
    assert (tmp_path / "home" / ".orbweaver" / "mcp.json").resolve() in denied
    assert (tmp_path / ".orbweaver" / "mcp.json").resolve() in denied


def test_load_merges_host_env_over_workspace_command(tmp_path: Path):
    home = tmp_path / "home"
    root = tmp_path / "ws"
    _write_json(
        home / ".orbweaver" / "mcp.json",
        {
            "mcpServers": {
                "fake": {
                    "command": "should-be-replaced",
                    "env": {"FAKE_MCP_TOKEN": "from-host", "KEEP": "host"},
                }
            }
        },
    )
    _write_json(
        root / ".orbweaver" / "mcp.json",
        {
            "mcpServers": {
                "fake": {
                    "command": sys.executable,
                    "args": [str(FAKE_SERVER)],
                    "env": {"KEEP": "workspace"},
                }
            }
        },
    )
    cfg = load_mcp_config(root, environ={}, home=home)
    assert len(cfg.enabled()) == 1
    spec = cfg.enabled()[0]
    assert spec.command == sys.executable
    assert spec.args == (str(FAKE_SERVER),)
    assert spec.env["FAKE_MCP_TOKEN"] == "from-host"
    assert spec.env["KEEP"] == "workspace"


def test_extra_config_file_and_disabled(tmp_path: Path, monkeypatch):
    root = tmp_path / "ws"
    root.mkdir()
    extra = tmp_path / "extra-mcp.json"
    extra.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "one": {"command": "echo", "disabled": True},
                    "two": {"command": "npx", "args": ["-y", "demo"]},
                }
            }
        ),
        encoding="utf-8",
    )
    cfg = load_mcp_config(
        root,
        environ={"ORBWEAVER_MCP_CONFIG": str(extra)},
        home=tmp_path / "nohome",
    )
    names = {s.name for s in cfg.enabled()}
    assert names == {"two"}


def test_skips_url_only_servers(tmp_path: Path):
    root = tmp_path / "ws"
    _write_json(
        root / ".orbweaver" / "mcp.json",
        {"mcpServers": {"remote": {"url": "https://example.invalid/mcp"}}},
    )
    cfg = load_mcp_config(root, environ={}, home=tmp_path / "nohome")
    assert cfg.enabled() == ()


def test_mcp_tools_are_not_allowlisted():
    assert "mcp_fake_echo" not in SAFE_ALLOWLIST
    assert not any(name.startswith("mcp_") for name in SAFE_ALLOWLIST)


@pytest.mark.asyncio
async def test_fake_stdio_server_list_and_call(tmp_path: Path, mcp_clean, monkeypatch):
    home = tmp_path / "home"
    root = tmp_path / "ws"
    monkeypatch.setattr("orbweaver.mcp.config.Path.home", lambda: home)
    _write_json(
        root / ".orbweaver" / "mcp.json",
        {"mcpServers": {"fake": _server_spec({"FAKE_MCP_TOKEN": "tok-test"})}},
    )
    ws = LocalWorkspace("workspace:default", str(root))
    cfg = load_mcp_config(root, environ={}, home=home)
    specs = await mcp_tool_specs(ws, config=cfg)
    names = {t["name"] for t in specs}
    assert "mcp_fake_echo" in names
    assert "mcp_fake_token_probe" in names
    echo = next(t for t in specs if t["name"] == "mcp_fake_echo")
    assert echo["input_schema"]["properties"]["text"]["type"] == "string"

    out = await call_mcp_tool("mcp_fake_echo", {"text": "hello-mcp"}, ws)
    assert out == "hello-mcp"
    token = await call_mcp_tool("mcp_fake_token_probe", {}, ws)
    assert token == "tok-test"


@pytest.mark.asyncio
async def test_session_tools_appends_mcp_without_editing_builtin_spec(
    tmp_path: Path, mcp_clean, monkeypatch
):
    root = tmp_path / "ws"
    _write_json(root / ".orbweaver" / "mcp.json", {"mcpServers": {"fake": _server_spec()}})
    monkeypatch.setattr("orbweaver.mcp.config.Path.home", lambda: tmp_path / "nohome")
    ws = LocalWorkspace("workspace:default", str(root))
    combined = await session_tools(ws)
    builtin = {t["name"] for t in TOOL_SPEC}
    names = {t["name"] for t in combined}
    assert builtin <= names
    assert "mcp_fake_echo" in names
    assert all(t in TOOL_SPEC for t in combined if t["name"] in builtin)


@pytest.mark.asyncio
async def test_run_tools_dispatches_mcp(tmp_path: Path, mcp_clean, monkeypatch):
    root = tmp_path / "ws"
    _write_json(root / ".orbweaver" / "mcp.json", {"mcpServers": {"fake": _server_spec()}})
    monkeypatch.setattr("orbweaver.mcp.config.Path.home", lambda: tmp_path / "nohome")
    ws = LocalWorkspace("workspace:default", str(root))
    await mcp_tool_specs(ws)
    out = await run_tools(
        "mcp_fake_echo",
        {"text": "via-run-tools"},
        {"workspace": ws, "store": None, "session_id": uuid4(), "workspace_kind": "local"},
    )
    assert out == "via-run-tools"


@pytest.mark.asyncio
async def test_mcp_tool_uses_classifier_not_allowlist(tmp_path, monkeypatch, mcp_clean):
    seen = {}

    async def classify(events, name, inp, **_k):
        seen["name"] = name
        seen["inp"] = inp
        return {"verdict": "allow", "reason": "mcp ok", "stage": "fast"}

    monkeypatch.setattr("orbweaver.permissions.pipeline.classify_action", classify)
    from orbweaver.permissions.denial import DenialTrackingState, reset_denial_states

    reset_denial_states()
    sid = uuid4()
    ws = LocalWorkspace("workspace:default", str(tmp_path))
    ctx = {
        "workspace": ws,
        "workspace_kind": "local",
        "headless": False,
        "session_id": sid,
        "denial_state": DenialTrackingState(),
        "events": [],
    }
    decision = await can_use_tool("mcp_fake_echo", {"text": "hi"}, ctx)
    assert decision.behavior == "allow"
    assert decision.fast_path == "classifier"
    assert seen["name"] == "mcp_fake_echo"
    assert seen["inp"]["text"] == "hi"


@pytest.mark.asyncio
async def test_mcp_tool_deny_rule(tmp_path, monkeypatch, mcp_clean):
    async def boom(*_a, **_k):
        raise AssertionError("deny rule should not reach classifier")

    monkeypatch.setattr("orbweaver.permissions.pipeline.classify_action", boom)
    from orbweaver.config import settings
    from orbweaver.permissions.denial import DenialTrackingState, reset_denial_states

    monkeypatch.setattr(settings, "orbweaver_permission_deny", "mcp_fake_echo")
    reset_denial_states()
    ws = LocalWorkspace("workspace:default", str(tmp_path))
    ctx = {
        "workspace": ws,
        "workspace_kind": "local",
        "headless": False,
        "session_id": uuid4(),
        "denial_state": DenialTrackingState(),
        "events": [],
    }
    decision = await can_use_tool("mcp_fake_echo", {"text": "nope"}, ctx)
    assert decision.behavior == "deny"
    assert decision.fast_path == "deny_rule"


def test_exposed_name_sanitizes():
    assert exposed_tool_name("notion", "API-post-search") == "mcp_notion_API-post-search"
    assert exposed_tool_name("my server", "foo/bar") == "mcp_my_server_foo_bar"
