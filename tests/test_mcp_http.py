"""Streamable HTTP MCP transport against an in-process Starlette server."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from mcp_fake_http_server import SESSION_ID, FakeHttpMcp

from orbweaver.mcp import call_mcp_tool, mcp_tool_annotations, mcp_tool_specs, reset_mcp_sessions
from orbweaver.mcp.client import HttpMcpSession, McpError
from orbweaver.mcp.config import McpServerSpec
from orbweaver.permissions.pipeline import can_use_tool
from orbweaver.workspace import LocalWorkspace

URL = "http://mcp.test/mcp"


def _client_factory(fake: FakeHttpMcp):
    def make() -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.ASGITransport(app=fake.app))

    return make


def _session(fake: FakeHttpMcp, **overrides) -> HttpMcpSession:
    spec = McpServerSpec(name="remote", url=URL, **overrides)
    return HttpMcpSession(spec, client_factory=_client_factory(fake))


@pytest.fixture
async def mcp_clean():
    await reset_mcp_sessions()
    yield
    await reset_mcp_sessions()


def _posts(fake: FakeHttpMcp, method: str) -> list[dict]:
    return [r for r in fake.requests if r["method"] == "POST" and (r["body"] or {}).get("method") == method]


@pytest.mark.asyncio
async def test_initialize_session_headers_and_delete():
    fake = FakeHttpMcp(require_bearer="tok-1")
    session = _session(fake, headers={"Authorization": "Bearer tok-1"})
    tools = await session.list_tools()
    assert {t["name"] for t in tools} >= {"lookup", "purge", "stream_echo"}
    assert session.protocol_version == "2025-06-18"
    assert "resources" in session.server_capabilities()

    init = _posts(fake, "initialize")[0]
    assert init["headers"]["accept"] == "application/json, text/event-stream"
    assert init["headers"]["authorization"] == "Bearer tok-1"
    assert "mcp-session-id" not in init["headers"]
    assert "mcp-protocol-version" not in init["headers"]
    assert init["body"]["params"]["protocolVersion"] == "2025-06-18"

    initialized = _posts(fake, "notifications/initialized")[0]
    assert initialized["headers"]["mcp-session-id"] == SESSION_ID
    assert initialized["headers"]["mcp-protocol-version"] == "2025-06-18"
    listed = _posts(fake, "tools/list")[0]
    assert listed["headers"]["mcp-session-id"] == SESSION_ID
    assert listed["headers"]["mcp-protocol-version"] == "2025-06-18"

    await session.close()
    assert fake.deleted == [SESSION_ID]
    delete = next(r for r in fake.requests if r["method"] == "DELETE")
    assert delete["headers"]["authorization"] == "Bearer tok-1"


@pytest.mark.asyncio
async def test_call_tool_json_and_sse_responses():
    fake = FakeHttpMcp()
    session = _session(fake)
    try:
        assert await session.call_tool("lookup", {"q": "x"}) == "found x"
        # SSE: a progress notification precedes the JSON-RPC response.
        assert await session.call_tool("stream_echo", {"text": "hi"}) == "sse:hi"
        # Server-initiated sampling request is rejected with a JSON-RPC error and
        # the tools/call response that follows it is still delivered.
        assert await session.call_tool("ask_me", {}) == "answered anyway"
        assert fake.client_responses == [
            {
                "jsonrpc": "2.0",
                "id": "srv-1",
                "error": {"code": -32601, "message": "orbweaver does not support sampling/createMessage"},
            }
        ]
        with pytest.raises(McpError, match="unknown nope"):
            await session.call_tool("nope", {})
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_unauthorized_is_an_mcp_error():
    fake = FakeHttpMcp(require_bearer="right")
    session = _session(fake, headers={"Authorization": "Bearer wrong"})
    try:
        with pytest.raises(McpError, match="401"):
            await session.list_tools()
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_resources_read():
    fake = FakeHttpMcp()
    session = _session(fake)
    try:
        assert [r["uri"] for r in await session.list_resources()] == ["memo://hello"]
        assert await session.read_resource("memo://hello") == "hello resource"
        with pytest.raises(McpError, match="not found"):
            await session.read_resource("memo://missing")
    finally:
        await session.close()


def _configure(tmp_path: Path, monkeypatch, fake: FakeHttpMcp, **extra) -> LocalWorkspace:
    root = tmp_path / "ws"
    (root / ".orbweaver").mkdir(parents=True)
    (root / ".orbweaver" / "mcp.json").write_text(
        json.dumps({"mcpServers": {"remote": {"url": URL, **extra}}}), encoding="utf-8"
    )
    monkeypatch.setattr("orbweaver.mcp.config.Path.home", lambda: tmp_path / "nohome")

    def new_session(spec: McpServerSpec):
        assert spec.transport == "http"
        return HttpMcpSession(spec, client_factory=_client_factory(fake))

    monkeypatch.setattr("orbweaver.mcp.tools._new_session", new_session)
    return LocalWorkspace("workspace:default", str(root))


@pytest.mark.asyncio
async def test_tools_annotations_resources_and_permissions(tmp_path: Path, monkeypatch, mcp_clean):
    fake = FakeHttpMcp()
    ws = _configure(tmp_path, monkeypatch, fake)
    specs = await mcp_tool_specs(ws)
    names = {t["name"] for t in specs}
    assert {
        "mcp_remote_lookup",
        "mcp_remote_purge",
        "mcp_remote_read_resource",
        "mcp_remote_list_resources",
    } <= names
    lookup = mcp_tool_annotations("mcp_remote_lookup")
    assert lookup is not None and lookup.read_only and not lookup.destructive
    assert lookup.title == "Lookup"
    purge = mcp_tool_annotations("mcp_remote_purge")
    assert purge is not None and purge.destructive and not purge.read_only
    assert mcp_tool_annotations("mcp_remote_read_resource").read_only

    async def boom(*_a, **_k):
        raise AssertionError("read-only MCP tools must not reach the classifier")

    monkeypatch.setattr("orbweaver.permissions.pipeline.classify_action", boom)
    from orbweaver.permissions.denial import DenialTrackingState, reset_denial_states

    reset_denial_states()
    ctx = {
        "workspace": ws,
        "workspace_kind": "local",
        "headless": False,
        "session_id": __import__("uuid").uuid4(),
        "denial_state": DenialTrackingState(),
        "events": [],
    }
    for name in ("mcp_remote_lookup", "mcp_remote_read_resource", "mcp_remote_list_resources"):
        decision = await can_use_tool(name, {"q": "x", "uri": "memo://hello"}, ctx)
        assert decision.behavior == "allow", name
        assert decision.fast_path == "mcp_readonly"

    assert await call_mcp_tool("mcp_remote_lookup", {"q": "cats"}, ws) == "found cats"
    assert await call_mcp_tool("mcp_remote_read_resource", {"uri": "memo://hello"}, ws) == "hello resource"
    listed = json.loads(await call_mcp_tool("mcp_remote_list_resources", {}, ws))
    assert listed == [{"uri": "memo://hello", "name": "hello", "mimeType": "text/plain"}]


@pytest.mark.asyncio
async def test_list_changed_triggers_relist_on_next_call(tmp_path: Path, monkeypatch, mcp_clean):
    fake = FakeHttpMcp()
    ws = _configure(tmp_path, monkeypatch, fake)
    await mcp_tool_specs(ws)
    assert mcp_tool_annotations("mcp_remote_extra") is None
    lists_before = len(_posts(fake, "tools/list"))

    assert await call_mcp_tool("mcp_remote_add_tool", {}, ws) == "added"
    from orbweaver.mcp import tools as mcp_tools

    session = next(iter(mcp_tools._sessions.values()))
    assert session.tools_dirty is True

    # Next dispatch re-lists lazily (exactly one extra tools/list) and the new
    # tool resolves without another mcp_tool_specs() round.
    assert await call_mcp_tool("mcp_remote_extra", {}, ws) == "extra ok"
    assert session.tools_dirty is False
    assert len(_posts(fake, "tools/list")) == lists_before + 1
    assert mcp_tool_annotations("mcp_remote_extra").read_only

    await reset_mcp_sessions()
    assert fake.deleted == [SESSION_ID]


@pytest.mark.asyncio
async def test_http_timeout_is_error_not_hang():
    import asyncio

    from starlette.applications import Starlette
    from starlette.responses import JSONResponse
    from starlette.routing import Route

    async def slow(request):
        body = json.loads(await request.body())
        if body.get("method") == "initialize":
            return JSONResponse(
                {
                    "jsonrpc": "2.0",
                    "id": body["id"],
                    "result": {"protocolVersion": "2025-03-26", "capabilities": {}},
                }
            )
        if "id" not in body:
            return JSONResponse({}, status_code=202)
        await asyncio.sleep(5)
        return JSONResponse({"jsonrpc": "2.0", "id": body["id"], "result": {"tools": []}})

    app = Starlette(routes=[Route("/mcp", slow, methods=["POST", "GET", "DELETE"])])
    spec = McpServerSpec(name="slow", url=URL, timeout_s=0.2, startup_timeout_s=0.2)
    session = HttpMcpSession(
        spec, client_factory=lambda: httpx.AsyncClient(transport=httpx.ASGITransport(app=app))
    )
    try:
        with pytest.raises(McpError, match="timed out after 0.2s"):
            await session.call_tool("anything", {})
        assert session.protocol_version == "2025-03-26"
    finally:
        await session.close()
