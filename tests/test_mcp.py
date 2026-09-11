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


def test_url_servers_use_http_transport(tmp_path: Path):
    root = tmp_path / "ws"
    _write_json(
        root / ".orbweaver" / "mcp.json",
        {
            "mcpServers": {
                "remote": {
                    "url": "https://example.invalid/mcp",
                    "headers": {"Authorization": "Bearer ${REMOTE_TOKEN}"},
                    "envPassthrough": ["REMOTE_TOKEN"],
                    "timeout_s": 12,
                },
                "bogus": {"url": "ftp://example.invalid/mcp"},
            }
        },
    )
    cfg = load_mcp_config(root, environ={"REMOTE_TOKEN": "r-tok"}, home=tmp_path / "nohome")
    assert [s.name for s in cfg.enabled()] == ["remote"]
    remote = cfg.enabled()[0]
    assert remote.transport == "http"
    assert remote.headers["Authorization"] == "Bearer r-tok"
    assert remote.timeout_s == 12
    assert remote.startup_timeout_s == 8.0


def test_env_expansion_only_from_allowlist(tmp_path: Path):
    root = tmp_path / "ws"
    _write_json(
        root / ".orbweaver" / "mcp.json",
        {
            "envPassthrough": ["GLOBAL_OK"],
            "mcpServers": {
                "fake": {
                    "command": "x",
                    "env": {
                        "A": "${GLOBAL_OK}",
                        "B": "${LOCAL_OK}",
                        "C": "${ANTHROPIC_API_KEY}",
                        "D": "${NOT_LISTED}",
                        "E": "pre-${GLOBAL_OK}-post",
                    },
                    "envPassthrough": ["LOCAL_OK", "ANTHROPIC_API_KEY"],
                }
            },
        },
    )
    environ = {
        "GLOBAL_OK": "g",
        "LOCAL_OK": "l",
        "ANTHROPIC_API_KEY": "sk-secret",
        "NOT_LISTED": "n",
    }
    spec = load_mcp_config(root, environ=environ, home=tmp_path / "nohome").enabled()[0]
    assert spec.env == {"A": "g", "B": "l", "C": "", "D": "", "E": "pre-g-post"}


def test_build_mcp_env_minimal():
    from orbweaver.mcp.environment import build_mcp_env

    environ = {
        "PATH": "/bin",
        "HOME": "/h",
        "LANG": "C.UTF-8",
        "LC_ALL": "C",
        "TERM": "xterm",
        "TMPDIR": "/tmp",
        "USER": "u",
        "ANTHROPIC_API_KEY": "sk",
        "OPENAI_API_KEY": "sk2",
        "ORBWEAVER_JWT_SECRET": "j",
        "TELEGRAM_BOT_TOKEN": "t",
        "DATABASE_URL": "postgres://",
        "MY_SERVICE_TOKEN": "svc",
        "RANDOM_VAR": "r",
    }
    env = build_mcp_env(environ, explicit={"EXPLICIT": "1"}, passthrough=["MY_SERVICE_TOKEN", "ORBWEAVER_JWT_SECRET", "*"])
    assert env == {
        "PATH": "/bin",
        "HOME": "/h",
        "LANG": "C.UTF-8",
        "LC_ALL": "C",
        "TERM": "xterm",
        "TMPDIR": "/tmp",
        "USER": "u",
        "MY_SERVICE_TOKEN": "svc",
        "EXPLICIT": "1",
    }
    assert build_mcp_env({})["PATH"]


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


def _permission_ctx(tmp_path: Path) -> dict:
    from orbweaver.permissions.denial import DenialTrackingState, reset_denial_states

    reset_denial_states()
    return {
        "workspace": LocalWorkspace("workspace:default", str(tmp_path)),
        "workspace_kind": "local",
        "headless": False,
        "session_id": uuid4(),
        "denial_state": DenialTrackingState(),
        "events": [],
    }


@pytest.mark.asyncio
async def test_stdio_server_env_is_minimal(tmp_path: Path, mcp_clean, monkeypatch):
    home = tmp_path / "home"
    root = tmp_path / "ws"
    monkeypatch.setattr("orbweaver.mcp.config.Path.home", lambda: home)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-leak")
    monkeypatch.setenv("ORBWEAVER_JWT_SECRET", "jwt-leak")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "tg-leak")
    monkeypatch.setenv("DATABASE_URL", "postgres://leak")
    monkeypatch.setenv("FAKE_ALLOWED", "allowed-value")
    spec = _server_spec({"FAKE_EXPLICIT": "explicit-value", "FAKE_MCP_TOKEN": "${FAKE_ALLOWED}"})
    spec["envPassthrough"] = ["FAKE_ALLOWED"]
    _write_json(root / ".orbweaver" / "mcp.json", {"mcpServers": {"fake": spec}})
    ws = LocalWorkspace("workspace:default", str(root))
    await mcp_tool_specs(ws)

    seen = json.loads(await call_mcp_tool("mcp_fake_env_probe", {}, ws))
    assert "ANTHROPIC_API_KEY" not in seen
    assert "ORBWEAVER_JWT_SECRET" not in seen
    assert "TELEGRAM_BOT_TOKEN" not in seen
    assert "DATABASE_URL" not in seen
    assert seen["FAKE_ALLOWED"] == "allowed-value"  # envPassthrough forwards the variable
    assert seen["FAKE_MCP_TOKEN"] == "allowed-value"  # ${VAR} expanded from the allowlist
    assert seen["FAKE_EXPLICIT"] == "explicit-value"
    assert seen["PATH"]


@pytest.mark.asyncio
async def test_stdio_negotiates_2025_06_18(tmp_path: Path, mcp_clean, monkeypatch):
    from orbweaver.mcp.client import StdioMcpSession
    from orbweaver.mcp.config import McpServerSpec

    session = StdioMcpSession(
        McpServerSpec(name="fake", command=sys.executable, args=(str(FAKE_SERVER),))
    )
    try:
        tools = await session.list_tools()
        assert session.protocol_version == "2025-06-18"
        assert session.server_capabilities()["tools"]["listChanged"] is True
        assert {t["name"] for t in tools} >= {"echo", "env_probe", "sleep"}
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_per_server_timeout_returns_error_not_hang(tmp_path: Path, mcp_clean, monkeypatch):
    import time

    root = tmp_path / "ws"
    monkeypatch.setattr("orbweaver.mcp.config.Path.home", lambda: tmp_path / "nohome")
    spec = _server_spec()
    spec["timeout_s"] = 0.2
    _write_json(root / ".orbweaver" / "mcp.json", {"mcpServers": {"fake": spec}})
    ws = LocalWorkspace("workspace:default", str(root))
    await mcp_tool_specs(ws)

    started = time.monotonic()
    out = await call_mcp_tool("mcp_fake_sleep", {"seconds": 5}, ws)
    assert time.monotonic() - started < 3
    assert out.startswith("MCP error (fake/sleep)")
    assert "timed out after 0.2s" in out
    # The session restarts cleanly after the timeout.
    assert await call_mcp_tool("mcp_fake_echo", {"text": "alive"}, ws) == "alive"


@pytest.mark.asyncio
async def test_readonly_annotation_auto_allows(tmp_path: Path, mcp_clean, monkeypatch):
    async def boom(*_a, **_k):
        raise AssertionError("readOnlyHint tool should not reach the classifier")

    monkeypatch.setattr("orbweaver.permissions.pipeline.classify_action", boom)
    root = tmp_path / "ws"
    monkeypatch.setattr("orbweaver.mcp.config.Path.home", lambda: tmp_path / "nohome")
    _write_json(root / ".orbweaver" / "mcp.json", {"mcpServers": {"fake": _server_spec()}})
    ws = LocalWorkspace("workspace:default", str(root))
    specs = await mcp_tool_specs(ws)
    probe = next(t for t in specs if t["name"] == "mcp_fake_env_probe")
    assert "(Environment probe)" in probe["description"]
    assert "[read-only]" in probe["description"]
    assert set(probe) == {"name", "description", "input_schema"}

    ctx = _permission_ctx(root)
    decision = await can_use_tool("mcp_fake_env_probe", {}, ctx)
    assert decision.behavior == "allow"
    assert decision.fast_path == "mcp_readonly"


@pytest.mark.asyncio
async def test_readonly_auto_allow_can_be_disabled(tmp_path: Path, mcp_clean, monkeypatch):
    from orbweaver.config import settings

    seen = {}

    async def classify(events, name, inp, **_k):
        seen["name"] = name
        return {"verdict": "allow", "reason": "ok", "stage": "fast"}

    monkeypatch.setattr("orbweaver.permissions.pipeline.classify_action", classify)
    monkeypatch.setattr(settings, "orbweaver_mcp_auto_allow_readonly", False)
    root = tmp_path / "ws"
    monkeypatch.setattr("orbweaver.mcp.config.Path.home", lambda: tmp_path / "nohome")
    _write_json(root / ".orbweaver" / "mcp.json", {"mcpServers": {"fake": _server_spec()}})
    ws = LocalWorkspace("workspace:default", str(root))
    await mcp_tool_specs(ws)
    decision = await can_use_tool("mcp_fake_env_probe", {}, _permission_ctx(root))
    assert decision.fast_path == "classifier"
    assert seen["name"] == "mcp_fake_env_probe"


@pytest.mark.asyncio
async def test_unannotated_and_destructive_go_to_classifier(tmp_path: Path, mcp_clean, monkeypatch):
    seen: list[tuple[str, str]] = []

    async def classify(events, name, inp, *, extra_framing="", **_k):
        seen.append((name, extra_framing))
        return {"verdict": "ask", "reason": "check", "stage": "fast"}

    monkeypatch.setattr("orbweaver.permissions.pipeline.classify_action", classify)
    root = tmp_path / "ws"
    monkeypatch.setattr("orbweaver.mcp.config.Path.home", lambda: tmp_path / "nohome")
    _write_json(root / ".orbweaver" / "mcp.json", {"mcpServers": {"fake": _server_spec()}})
    ws = LocalWorkspace("workspace:default", str(root))
    await mcp_tool_specs(ws)

    plain = await can_use_tool("mcp_fake_echo", {"text": "x"}, _permission_ctx(root))
    assert plain.behavior == "ask"
    destructive = await can_use_tool("mcp_fake_wipe", {}, _permission_ctx(root))
    assert destructive.behavior == "ask"
    assert seen[0] == ("mcp_fake_echo", "")
    assert seen[1][0] == "mcp_fake_wipe"
    assert "destructiveHint" in seen[1][1]


def test_exposed_name_sanitizes():
    assert exposed_tool_name("notion", "API-post-search") == "mcp_notion_API-post-search"
    assert exposed_tool_name("my server", "foo/bar") == "mcp_my_server_foo_bar"
