"""Expose configured MCP tools as Anthropic tool specs and dispatch calls."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from orbweaver.mcp.client import McpError, McpSession, make_session
from orbweaver.mcp.config import McpConfig, McpServerSpec, load_mcp_config

log = logging.getLogger(__name__)

_SAFE = re.compile(r"[^A-Za-z0-9_-]+")

KIND_TOOL = "tool"
KIND_READ_RESOURCE = "read_resource"
KIND_LIST_RESOURCES = "list_resources"


@dataclass(frozen=True)
class McpToolAnnotations:
    """Subset of MCP ``ToolAnnotations`` the permission pipeline cares about."""

    read_only: bool = False
    destructive: bool = False
    title: str = ""
    server: str = ""
    original: str = ""

    @property
    def auto_allow_candidate(self) -> bool:
        return self.read_only and not self.destructive


@dataclass(frozen=True)
class _Registered:
    server: str
    original: str
    kind: str
    annotations: McpToolAnnotations


_sessions: dict[str, McpSession] = {}
_tool_index: dict[str, _Registered] = {}
_loaded_fp = ""


async def reset_mcp_sessions() -> None:
    """Close cached sessions (tests and config reloads)."""
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


def mcp_tool_annotations(name: str) -> McpToolAnnotations | None:
    """Annotations for an exposed ``mcp_*`` tool, or None when unknown."""
    entry = _tool_index.get(name)
    return entry.annotations if entry else None


def _session_key(spec: McpServerSpec) -> str:
    return spec.fingerprint()


def _config_for(workspace) -> McpConfig:
    root = getattr(workspace, "root", None)
    return load_mcp_config(root)


def _new_session(spec: McpServerSpec) -> McpSession:
    """Indirection so tests can inject an ``httpx`` transport."""
    return make_session(spec)


async def _sync_sessions(config: McpConfig) -> None:
    global _loaded_fp
    fp = config.fingerprint()
    if fp == _loaded_fp and _sessions:
        return
    await reset_mcp_sessions()
    for spec in config.enabled():
        _sessions[_session_key(spec)] = _new_session(spec)
    _loaded_fp = fp


def _anthropic_schema(tool: dict[str, Any]) -> dict[str, Any]:
    schema = tool.get("inputSchema") or tool.get("input_schema") or {"type": "object"}
    if not isinstance(schema, dict):
        schema = {"type": "object"}
    schema = dict(schema)
    schema.setdefault("type", "object")
    return schema


def _parse_annotations(tool: dict[str, Any], server: str, original: str) -> McpToolAnnotations:
    raw = tool.get("annotations")
    if not isinstance(raw, dict):
        raw = {}
    return McpToolAnnotations(
        read_only=raw.get("readOnlyHint") is True,
        destructive=raw.get("destructiveHint") is True,
        title=str(raw.get("title") or tool.get("title") or ""),
        server=server,
        original=original,
    )


def _reserve(used: set[str], exposed: str) -> str:
    if exposed in used:
        exposed = f"{exposed[:60]}_{len(used)}"
    used.add(exposed)
    return exposed


def _describe(spec: McpServerSpec, ann: McpToolAnnotations, desc: str) -> str:
    label = f"MCP / {spec.name}"
    if ann.title:
        label += f" ({ann.title})"
    hints = []
    if ann.read_only:
        hints.append("read-only")
    if ann.destructive:
        hints.append("destructive")
    suffix = f" [{', '.join(hints)}]" if hints else ""
    return f"{label}: {desc}{suffix}"


def _resource_specs(
    spec: McpServerSpec, session: McpSession, used: set[str]
) -> list[dict[str, Any]]:
    caps = session.server_capabilities()
    if not isinstance(caps.get("resources"), dict):
        return []
    out: list[dict[str, Any]] = []
    read_name = _reserve(used, exposed_tool_name(spec.name, "read_resource"))
    _tool_index[read_name] = _Registered(
        spec.name,
        "resources/read",
        KIND_READ_RESOURCE,
        McpToolAnnotations(read_only=True, title="Read resource", server=spec.name, original="resources/read"),
    )
    out.append(
        {
            "name": read_name,
            "description": (
                f"MCP / {spec.name}: read a resource by URI "
                f"(see {exposed_tool_name(spec.name, 'list_resources')}) [read-only]"
            ),
            "input_schema": {
                "type": "object",
                "properties": {"uri": {"type": "string", "description": "Resource URI"}},
                "required": ["uri"],
            },
        }
    )
    list_name = _reserve(used, exposed_tool_name(spec.name, "list_resources"))
    _tool_index[list_name] = _Registered(
        spec.name,
        "resources/list",
        KIND_LIST_RESOURCES,
        McpToolAnnotations(read_only=True, title="List resources", server=spec.name, original="resources/list"),
    )
    out.append(
        {
            "name": list_name,
            "description": f"MCP / {spec.name}: list available resources (uri, name, mimeType) [read-only]",
            "input_schema": {"type": "object", "properties": {}},
        }
    )
    return out


async def _index_server(spec: McpServerSpec, session: McpSession, used: set[str]) -> list[dict[str, Any]]:
    tools = await session.list_tools()
    specs: list[dict[str, Any]] = []
    for tool in tools:
        original = str(tool.get("name") or "")
        exposed = _reserve(used, exposed_tool_name(spec.name, original))
        ann = _parse_annotations(tool, spec.name, original)
        _tool_index[exposed] = _Registered(spec.name, original, KIND_TOOL, ann)
        desc = str(tool.get("description") or original or "MCP tool")
        specs.append(
            {
                "name": exposed,
                "description": _describe(spec, ann, desc),
                "input_schema": _anthropic_schema(tool),
            }
        )
    specs.extend(_resource_specs(spec, session, used))
    return specs
def mcp_tool_read_only(name: str) -> bool:
    """Whether an exposed ``mcp_*`` tool is annotated read-only (and not destructive).

    Unknown or not-yet-listed tools are not read-only; the agent treats them as
    unsafe to run concurrently.
    """
    ann = mcp_tool_annotations(name)
    return bool(ann and ann.auto_allow_candidate)




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
            specs.extend(await _index_server(spec, session, used))
        except Exception as e:
            log.warning("MCP server %s tools/list failed: %s", spec.name, e)
            continue
    return specs


def _drop_server(server_name: str) -> None:
    for exposed in [k for k, v in _tool_index.items() if v.server == server_name]:
        _tool_index.pop(exposed, None)


async def _refresh_if_dirty(spec: McpServerSpec, session: McpSession) -> None:
    if not session.tools_dirty:
        return
    _drop_server(spec.name)
    used = {k for k, v in _tool_index.items() if v.server != spec.name}
    try:
        await _index_server(spec, session, used)
    except Exception as e:
        log.warning("MCP server %s re-list after list_changed failed: %s", spec.name, e)


async def call_mcp_tool(name: str, inp: dict[str, Any], workspace=None) -> str:
    cfg = _config_for(workspace)
    await _sync_sessions(cfg)
    entry = _tool_index.get(name)
    if entry is not None:
        spec = next((s for s in cfg.enabled() if s.name == entry.server), None)
        session = _sessions.get(_session_key(spec)) if spec else None
        if spec is not None and session is not None:
            await _refresh_if_dirty(spec, session)
            entry = _tool_index.get(name)
    if entry is None:
        # Fresh list in case the cache was empty (direct run_tools in tests) or
        # the server announced tools/list_changed.
        await mcp_tool_specs(workspace, config=cfg)
        entry = _tool_index.get(name)
    if entry is None:
        return f"unknown MCP tool {name}"
    spec = next((s for s in cfg.enabled() if s.name == entry.server), None)
    if spec is None:
        return f"MCP server {entry.server!r} is not configured"
    session = _sessions.get(_session_key(spec))
    if session is None:
        session = _new_session(spec)
        _sessions[_session_key(spec)] = session
    try:
        if entry.kind == KIND_READ_RESOURCE:
            uri = str(inp.get("uri") or "").strip()
            if not uri:
                return "MCP read_resource needs a uri"
            return await session.read_resource(uri)
        if entry.kind == KIND_LIST_RESOURCES:
            items = await session.list_resources()
            return json.dumps(
                [
                    {k: r.get(k) for k in ("uri", "name", "description", "mimeType") if r.get(k)}
                    for r in items
                ],
                indent=1,
            )
        return await session.call_tool(entry.original, inp)
    except McpError as e:
        return f"MCP error ({entry.server}/{entry.original}): {e}"


def workspace_root_path(workspace) -> Path | None:
    root = getattr(workspace, "root", None)
    if root is None:
        return None
    return Path(root)
