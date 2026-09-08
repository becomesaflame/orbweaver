"""Minimal MCP stdio JSON-RPC client (initialize, tools/list, tools/call)."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any

from orbweaver.mcp.config import McpServerSpec

log = logging.getLogger(__name__)

PROTOCOL_VERSION = "2024-11-05"
HANDSHAKE_TIMEOUT = 8.0
CALL_TIMEOUT = 30.0


class McpError(RuntimeError):
    """MCP server failed or returned an error payload."""


def _encode(msg: dict[str, Any]) -> bytes:
    return (json.dumps(msg, separators=(",", ":")) + "\n").encode("utf-8")


async def _read_message(reader: asyncio.StreamReader) -> dict[str, Any] | None:
    """Read one JSON-RPC message (newline JSON, or LSP Content-Length framing)."""
    first = await reader.readline()
    if not first:
        return None
    header = first.decode("utf-8", errors="replace")
    if header.lower().startswith("content-length:"):
        try:
            length = int(header.split(":", 1)[1].strip())
        except ValueError as e:
            raise McpError(f"bad Content-Length header: {header!r}") from e
        # Consume remaining headers until the blank line.
        while True:
            line = await reader.readline()
            if not line or line in {b"\r\n", b"\n"}:
                break
        body = await reader.readexactly(length)
        data = json.loads(body.decode("utf-8"))
        return data if isinstance(data, dict) else {}
    text = header.strip()
    if not text:
        return await _read_message(reader)
    data = json.loads(text)
    return data if isinstance(data, dict) else {}


def _text_from_result(result: Any) -> str:
    if result is None:
        return ""
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        if result.get("isError"):
            content = result.get("content")
            detail = _flatten_content(content) if content else json.dumps(result, default=str)
            raise McpError(detail or "mcp tool error")
        if "content" in result:
            return _flatten_content(result.get("content"))
        return json.dumps(result, default=str)
    return json.dumps(result, default=str)


def _flatten_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return json.dumps(content, default=str)
    parts: list[str] = []
    for item in content:
        if isinstance(item, str):
            parts.append(item)
            continue
        if not isinstance(item, dict):
            parts.append(json.dumps(item, default=str))
            continue
        if item.get("type") == "text" or "text" in item:
            parts.append(str(item.get("text") or ""))
        else:
            parts.append(json.dumps(item, default=str))
    return "\n".join(p for p in parts if p)


class StdioMcpSession:
    """One long-lived stdio MCP server process."""

    def __init__(self, spec: McpServerSpec):
        self.spec = spec
        self._proc: asyncio.subprocess.Process | None = None
        self._id = 0
        self._lock = asyncio.Lock()
        self._ready = False

    async def close(self) -> None:
        proc = self._proc
        self._proc = None
        self._ready = False
        if proc is None:
            return
        if proc.returncode is None:
            proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), timeout=2)
            except TimeoutError:
                proc.kill()
                await proc.wait()

    async def _ensure(self) -> None:
        if self._proc is not None and self._proc.returncode is None and self._ready:
            return
        await self.close()
        env = dict(os.environ)
        env.update(self.spec.env)
        cwd = self.spec.cwd or None
        try:
            self._proc = await asyncio.create_subprocess_exec(
                self.spec.command,
                *self.spec.args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                cwd=cwd,
            )
        except OSError as e:
            raise McpError(f"failed to start MCP server {self.spec.name!r}: {e}") from e
        from orbweaver import __version__

        init = await self._rpc(
            "initialize",
            {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "orbweaver", "version": __version__},
            },
            timeout=HANDSHAKE_TIMEOUT,
        )
        if not isinstance(init, dict):
            raise McpError(f"MCP server {self.spec.name!r} initialize failed")
        await self._notify("notifications/initialized")
        self._ready = True

    async def _write(self, msg: dict[str, Any]) -> None:
        proc = self._proc
        if proc is None or proc.stdin is None:
            raise McpError(f"MCP server {self.spec.name!r} is not running")
        proc.stdin.write(_encode(msg))
        await proc.stdin.drain()

    async def _notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        msg: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            msg["params"] = params
        await self._write(msg)

    async def _rpc(self, method: str, params: dict[str, Any] | None = None, *, timeout: float) -> Any:
        self._id += 1
        req_id = self._id
        msg: dict[str, Any] = {"jsonrpc": "2.0", "id": req_id, "method": method}
        if params is not None:
            msg["params"] = params
        await self._write(msg)
        return await asyncio.wait_for(self._wait_response(req_id), timeout=timeout)

    async def _wait_response(self, req_id: int) -> Any:
        proc = self._proc
        if proc is None or proc.stdout is None:
            raise McpError(f"MCP server {self.spec.name!r} has no stdout")
        while True:
            if proc.returncode is not None:
                err = b""
                if proc.stderr is not None:
                    err = await proc.stderr.read()
                raise McpError(
                    f"MCP server {self.spec.name!r} exited {proc.returncode}: "
                    f"{err.decode('utf-8', errors='replace')[:400]}"
                )
            raw = await _read_message(proc.stdout)
            if raw is None:
                raise McpError(f"MCP server {self.spec.name!r} closed stdout")
            if raw.get("method") and "id" in raw:
                if raw.get("method") == "ping":
                    await self._write({"jsonrpc": "2.0", "id": raw["id"], "result": {}})
                continue
            if "id" in raw and raw.get("id") == req_id:
                if "error" in raw:
                    err = raw["error"]
                    raise McpError(json.dumps(err, default=str) if not isinstance(err, str) else err)
                return raw.get("result")

    async def list_tools(self) -> list[dict[str, Any]]:
        async with self._lock:
            await self._ensure()
            result = await self._rpc("tools/list", {}, timeout=HANDSHAKE_TIMEOUT)
        tools = (result or {}).get("tools") if isinstance(result, dict) else None
        if not isinstance(tools, list):
            return []
        return [t for t in tools if isinstance(t, dict) and t.get("name")]

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        async with self._lock:
            await self._ensure()
            result = await self._rpc(
                "tools/call",
                {"name": name, "arguments": arguments or {}},
                timeout=CALL_TIMEOUT,
            )
        return _text_from_result(result)
