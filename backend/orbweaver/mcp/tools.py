"""Expose configured MCP tools as Anthropic tool specs and dispatch calls."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from orbweaver.mcp.client import McpError, StdioMcpSession
from orbweaver.mcp.config import McpConfig, McpServerSpec, load_mcp_config

log = logging.getLogger(__name__)

_SAFE = re.compile(r"[^A-Za-z0-9_-]+")

_sessions: dict[str, StdioMcpSession] = {}
_tool_index: dict[str, tuple[str, str]] = {}
_loaded_fp = ""


async def reset_mcp_sessions() -> None:
    """Close cached stdio sessions (tests and config reloads)."""
    global _loaded_fp
    _loaded_fp = ""
    _tool_index.clear()
    sessions = list(_sessions.values())
    _sessions.clear()
    for session in sessions:
        await session.close()


def sanitize_part(raw: str, fallback: str) -> str:
    text = _SAFE.sub("_", (raw or "").strip()).strip("_")
    return text or fallback


def exposed_tool_name(server: str, tool: str) -> str:
    name = f"mcp_{sanitize_part(server, 'server')}_{sanitize_part(tool, 'tool')}"
    return name[:64]


def _session_key(spec: McpServerSpec) -> str:
    return spec.fingerprint()


def _config_for(workspace) -> McpConfig:
    root = getattr(workspace, "root", None)
    return load_mcp_config(root)


async def _sync_sessions(config: McpConfig) -> None:
    global _loaded_fp
    fp = config.fingerprint()
    if fp == _loaded_fp and _sessions:
        return
    await reset_mcp_sessions()
    for spec in config.enabled():
        _sessions[_session_key(spec)] = StdioMcpSession(spec)
    _loaded_fp = fp


def _anthropic_schema(tool: dict[str, Any]) -> dict[str, Any]:
    schema = tool.get("inputSchema") or tool.get("input_schema") or {"type": "object"}
    if not isinstance(schema, dict):
        schema = {"type": "object"}
    schema = dict(schema)
    schema.setdefault("type", "object")
    return schema


async def mcp_tool_specs(workspace=None, *, config: McpConfig | None = None) -> list[dict[str, Any]]:
    """List configured MCP tools as Anthropic Messages API tool objects."""
    cfg = config if config is not None else _config_for(workspace)
    await _sync_sessions(cfg)
    specs: list[dict[str, Any]] = []
    used: set[str] = set()
    for spec in cfg.enabled():
        session = _sessions.get(_session_key(spec))
        if session is None:
            continue
        try:
            tools = await session.list_tools()
        except Exception as e:
            log.warning("MCP server %s tools/list failed: %s", spec.name, e)
            continue
        for tool in tools:
            original = str(tool.get("name") or "")
            exposed = exposed_tool_name(spec.name, original)
            if exposed in used:
                exposed = f"{exposed[:60]}_{len(used)}"
            used.add(exposed)
            _tool_index[exposed] = (spec.name, original)
            desc = str(tool.get("description") or original or "MCP tool")
            specs.append(
                {
                    "name": exposed,
                    "description": f"MCP / {spec.name}: {desc}",
                    "input_schema": _anthropic_schema(tool),
                }
            )
    return specs


async def call_mcp_tool(name: str, inp: dict[str, Any], workspace=None) -> str:
    if name not in _tool_index:
        # Fresh list in case the cache was empty (direct run_tools in tests).
        await mcp_tool_specs(workspace)
    mapped = _tool_index.get(name)
    if mapped is None:
        return f"unknown MCP tool {name}"
    server_name, original = mapped
    cfg = _config_for(workspace)
    spec = next((s for s in cfg.enabled() if s.name == server_name), None)
    if spec is None:
        return f"MCP server {server_name!r} is not configured"
    await _sync_sessions(cfg)
    session = _sessions.get(_session_key(spec))
    if session is None:
        session = StdioMcpSession(spec)
        _sessions[_session_key(spec)] = session
    try:
        return await session.call_tool(original, inp)
    except McpError as e:
        return f"MCP error ({server_name}/{original}): {e}"


def workspace_root_path(workspace) -> Path | None:
    root = getattr(workspace, "root", None)
    if root is None:
        return None
    return Path(root)
