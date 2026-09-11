"""MCP JSON-RPC client: stdio and Streamable HTTP transports behind one protocol.

Both sessions speak the same subset (initialize, tools/list, tools/call,
resources/list, resources/read) and share one message dispatcher that answers
``ping``, rejects server-initiated ``sampling``/``elicitation``/``roots``
requests with a JSON-RPC error, records ``notifications/tools/list_changed``
for a lazy re-list, and logs ``notifications/progress``.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
from collections.abc import AsyncIterator, Callable
from typing import Any, Protocol

import httpx

from orbweaver.mcp.config import McpServerSpec
from orbweaver.mcp.environment import build_mcp_env

log = logging.getLogger(__name__)

PROTOCOL_VERSION = "2025-06-18"
SUPPORTED_PROTOCOL_VERSIONS = ("2025-06-18", "2025-03-26", "2024-11-05")
HANDSHAKE_TIMEOUT = 8.0
CALL_TIMEOUT = 30.0

SESSION_HEADER = "Mcp-Session-Id"
PROTOCOL_HEADER = "MCP-Protocol-Version"
EVENT_STREAM = "text/event-stream"

# Server → client requests we refuse. Answering with an error keeps the reader
# moving; stalling would block every later tools/call on that server.
_REJECTED_SERVER_REQUESTS = frozenset(
    {"sampling/createMessage", "elicitation/create", "roots/list"}
)
_METHOD_NOT_FOUND = -32601


class McpError(RuntimeError):
    """MCP server failed or returned an error payload."""


class McpSession(Protocol):
    """What ``mcp/tools.py`` needs from a transport."""

    spec: McpServerSpec
    tools_dirty: bool

    def server_capabilities(self) -> dict[str, Any]: ...

    async def list_tools(self) -> list[dict[str, Any]]: ...

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str: ...

    async def list_resources(self) -> list[dict[str, Any]]: ...

    async def read_resource(self, uri: str) -> str: ...

    async def close(self) -> None: ...


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
        if "contents" in result:
            return _flatten_content(result.get("contents"))
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
        elif "blob" in item:
            size = len(str(item.get("blob") or ""))
            parts.append(
                f"[binary {item.get('mimeType') or 'application/octet-stream'} "
                f"{item.get('uri') or ''} base64 {size} chars]".replace("  ", " ")
            )
        else:
            parts.append(json.dumps(item, default=str))
    return "\n".join(p for p in parts if p)


async def _iter_sse_data(lines: AsyncIterator[str]) -> AsyncIterator[str]:
    """Yield the ``data`` payload of each SSE event (``event: message`` / ``data:`` lines)."""
    data_lines: list[str] = []
    async for raw_line in lines:
        line = raw_line.rstrip("\r\n")
        if not line:
            if data_lines:
                yield "\n".join(data_lines)
                data_lines = []
            continue
        if line.startswith(":"):
            continue
        field, _, value = line.partition(":")
        value = value.removeprefix(" ")
        if field == "data":
            data_lines.append(value)
    if data_lines:
        yield "\n".join(data_lines)


class _BaseMcpSession:
    """Transport-independent JSON-RPC bookkeeping shared by stdio and HTTP."""

    def __init__(self, spec: McpServerSpec):
        self.spec = spec
        self.tools_dirty = False
        self._id = 0
        self._lock = asyncio.Lock()
        self._ready = False
        self._negotiated = False
        self._capabilities: dict[str, Any] = {}
        self._protocol_version = PROTOCOL_VERSION

    # -- transport hooks -------------------------------------------------

    async def _connect(self) -> None:  # pragma: no cover - overridden
        raise NotImplementedError

    async def _send_notification(self, msg: dict[str, Any]) -> None:  # pragma: no cover
        raise NotImplementedError

    async def _send_response(self, msg: dict[str, Any]) -> None:  # pragma: no cover
        raise NotImplementedError

    async def _request(self, msg: dict[str, Any], *, timeout: float) -> Any:  # pragma: no cover
        raise NotImplementedError

    async def _on_timeout(self) -> None:
        """Hook for transports that must reset after a lost response."""

    async def close(self) -> None:  # pragma: no cover - overridden
        raise NotImplementedError

    # -- shared behaviour ------------------------------------------------

    def server_capabilities(self) -> dict[str, Any]:
        return dict(self._capabilities)

    @property
    def protocol_version(self) -> str:
        return self._protocol_version

    def _next_id(self) -> int:
        self._id += 1
        return self._id

    async def _rpc(self, method: str, params: dict[str, Any] | None = None, *, timeout: float) -> Any:
        msg: dict[str, Any] = {"jsonrpc": "2.0", "id": self._next_id(), "method": method}
        if params is not None:
            msg["params"] = params
        try:
            return await asyncio.wait_for(self._request(msg, timeout=timeout), timeout=timeout)
        except TimeoutError as e:
            await self._on_timeout()
            raise McpError(
                f"MCP server {self.spec.name!r} {method} timed out after {timeout:g}s"
            ) from e

    async def _notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        msg: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            msg["params"] = params
        await self._send_notification(msg)

    async def _handshake(self) -> None:
        from orbweaver import __version__

        init = await self._rpc(
            "initialize",
            {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "orbweaver", "version": __version__},
            },
            timeout=self.spec.startup_timeout_s,
        )
        if not isinstance(init, dict):
            raise McpError(f"MCP server {self.spec.name!r} initialize failed")
        version = str(init.get("protocolVersion") or PROTOCOL_VERSION)
        if version not in SUPPORTED_PROTOCOL_VERSIONS:
            raise McpError(
                f"MCP server {self.spec.name!r} wants protocol {version!r}; "
                f"supported: {', '.join(SUPPORTED_PROTOCOL_VERSIONS)}"
            )
        self._protocol_version = version
        self._negotiated = True
        caps = init.get("capabilities")
        self._capabilities = dict(caps) if isinstance(caps, dict) else {}
        await self._notify("notifications/initialized")
        self._ready = True

    async def _ensure(self) -> None:
        if self._ready:
            return
        await self._connect()
        await self._handshake()

    def _handle_unsolicited(self, raw: dict[str, Any]) -> tuple[bool, dict[str, Any] | None]:
        """Consume a server request or notification.

        Returns ``(consumed, reply)``. ``consumed`` is False when ``raw`` is a
        response the caller should match against its pending id. ``reply`` is a
        JSON-RPC message to send back (ping result, or an error for
        sampling/elicitation/roots requests we do not support).
        """
        method = raw.get("method")
        if not method:
            return False, None
        if "id" in raw:
            if method == "ping":
                return True, {"jsonrpc": "2.0", "id": raw["id"], "result": {}}
            if method in _REJECTED_SERVER_REQUESTS:
                log.info("MCP server %s requested %s; refused", self.spec.name, method)
            else:
                log.warning("MCP server %s sent unknown request %s", self.spec.name, method)
            return True, {
                "jsonrpc": "2.0",
                "id": raw["id"],
                "error": {
                    "code": _METHOD_NOT_FOUND,
                    "message": f"orbweaver does not support {method}",
                },
            }
        raw_params = raw.get("params")
        params: dict[str, Any] = raw_params if isinstance(raw_params, dict) else {}
        if method == "notifications/tools/list_changed":
            self.tools_dirty = True
            log.info("MCP server %s: tools/list_changed; re-listing on next call", self.spec.name)
        elif method == "notifications/progress":
            log.info(
                "MCP server %s progress %s: %s/%s %s",
                self.spec.name,
                params.get("progressToken"),
                params.get("progress"),
                params.get("total") or "?",
                params.get("message") or "",
            )
        elif method == "notifications/message":
            log.debug("MCP server %s log: %s", self.spec.name, params.get("data"))
        else:
            log.debug("MCP server %s notification %s ignored", self.spec.name, method)
        return True, None

    async def _dispatch(self, raw: dict[str, Any], req_id: int) -> tuple[bool, Any]:
        """Route one incoming message. Returns (matched, result)."""
        consumed, reply = self._handle_unsolicited(raw)
        if consumed:
            if reply is not None:
                await self._send_response(reply)
            return False, None
        if "id" in raw and raw.get("id") == req_id:
            if "error" in raw:
                err = raw["error"]
                raise McpError(json.dumps(err, default=str) if not isinstance(err, str) else err)
            return True, raw.get("result")
        return False, None

    async def list_tools(self) -> list[dict[str, Any]]:
        async with self._lock:
            await self._ensure()
            result = await self._rpc("tools/list", {}, timeout=self.spec.startup_timeout_s)
            self.tools_dirty = False
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
                timeout=self.spec.timeout_s,
            )
        return _text_from_result(result)

    async def list_resources(self) -> list[dict[str, Any]]:
        async with self._lock:
            await self._ensure()
            result = await self._rpc("resources/list", {}, timeout=self.spec.startup_timeout_s)
        items = (result or {}).get("resources") if isinstance(result, dict) else None
        if not isinstance(items, list):
            return []
        return [r for r in items if isinstance(r, dict) and r.get("uri")]

    async def read_resource(self, uri: str) -> str:
        async with self._lock:
            await self._ensure()
            result = await self._rpc("resources/read", {"uri": uri}, timeout=self.spec.timeout_s)
        return _text_from_result(result)


class StdioMcpSession(_BaseMcpSession):
    """One long-lived stdio MCP server process with a minimal environment."""

    def __init__(self, spec: McpServerSpec, *, environ: dict[str, str] | None = None):
        super().__init__(spec)
        self._proc: asyncio.subprocess.Process | None = None
        self._environ = environ

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

    async def _on_timeout(self) -> None:
        # The late reply (or a half-read frame) would poison the stream.
        await self.close()

    def process_env(self) -> dict[str, str]:
        source = self._environ if self._environ is not None else dict(os.environ)
        return build_mcp_env(
            source, explicit=self.spec.env, passthrough=self.spec.env_passthrough
        )

    async def _ensure(self) -> None:
        if self._proc is not None and self._proc.returncode is None and self._ready:
            return
        await self.close()
        await self._connect()
        await self._handshake()

    async def _connect(self) -> None:
        cwd = self.spec.cwd or None
        try:
            self._proc = await asyncio.create_subprocess_exec(
                self.spec.command,
                *self.spec.args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=self.process_env(),
                cwd=cwd,
            )
        except OSError as e:
            raise McpError(f"failed to start MCP server {self.spec.name!r}: {e}") from e

    async def _write(self, msg: dict[str, Any]) -> None:
        proc = self._proc
        if proc is None or proc.stdin is None:
            raise McpError(f"MCP server {self.spec.name!r} is not running")
        proc.stdin.write(_encode(msg))
        await proc.stdin.drain()

    async def _send_notification(self, msg: dict[str, Any]) -> None:
        await self._write(msg)

    async def _send_response(self, msg: dict[str, Any]) -> None:
        await self._write(msg)

    async def _request(self, msg: dict[str, Any], *, timeout: float) -> Any:
        del timeout  # enforced by _rpc's wait_for
        await self._write(msg)
        return await self._wait_response(int(msg["id"]))

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
            matched, result = await self._dispatch(raw, req_id)
            if matched:
                return result


ClientFactory = Callable[[], httpx.AsyncClient]


class HttpMcpSession(_BaseMcpSession):
    """Streamable HTTP transport (MCP 2025-03-26 / 2025-06-18).

    Every JSON-RPC request is one POST. The server answers with
    ``application/json`` or an SSE stream whose events are JSON-RPC messages;
    we dispatch each until the response with our id arrives. ``Mcp-Session-Id``
    from ``initialize`` and ``MCP-Protocol-Version`` ride on later requests. An
    optional GET stream carries server-initiated notifications; ``close`` sends
    DELETE so the server can drop the session.
    """

    def __init__(self, spec: McpServerSpec, *, client_factory: ClientFactory | None = None):
        super().__init__(spec)
        self._client_factory = client_factory
        self._client: httpx.AsyncClient | None = None
        self._session_id: str | None = None
        self._listener: asyncio.Task[None] | None = None

    # -- lifecycle -------------------------------------------------------

    async def close(self) -> None:
        self._ready = False
        listener, self._listener = self._listener, None
        if listener is not None:
            listener.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await listener
        client, self._client = self._client, None
        session_id, self._session_id = self._session_id, None
        if client is None:
            return
        try:
            if session_id:
                headers = self._headers(session_id=session_id)
                resp = await client.delete(self.spec.url, headers=headers, timeout=5.0)
                if resp.status_code not in {200, 202, 204, 404, 405}:
                    log.debug(
                        "MCP server %s DELETE session -> %s", self.spec.name, resp.status_code
                    )
        except (httpx.HTTPError, OSError) as e:
            log.debug("MCP server %s DELETE session failed: %s", self.spec.name, e)
        finally:
            await client.aclose()

    async def _connect(self) -> None:
        if self._client is None:
            self._client = (
                self._client_factory()
                if self._client_factory is not None
                else httpx.AsyncClient(follow_redirects=False)
            )
        listener, self._listener = self._listener, None
        if listener is not None:
            listener.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await listener
        self._session_id = None
        self._negotiated = False
        self._protocol_version = PROTOCOL_VERSION

    async def _handshake(self) -> None:
        await super()._handshake()
        if self._listener is None:
            self._listener = asyncio.create_task(self._listen())

    async def _on_timeout(self) -> None:
        # A lost POST does not corrupt anything; keep the session id.
        return None

    # -- HTTP plumbing ---------------------------------------------------

    def _headers(self, *, session_id: str | None = None, json_body: bool = False) -> dict[str, str]:
        headers: dict[str, str] = {"Accept": f"application/json, {EVENT_STREAM}"}
        if json_body:
            headers["Content-Type"] = "application/json"
        headers.update(self.spec.headers)
        sid = session_id if session_id is not None else self._session_id
        if sid:
            headers[SESSION_HEADER] = sid
        if self._negotiated:
            headers[PROTOCOL_HEADER] = self._protocol_version
        return headers

    def _client_or_raise(self) -> httpx.AsyncClient:
        if self._client is None:
            raise McpError(f"MCP server {self.spec.name!r} is not connected")
        return self._client

    def _raise_for_status(self, resp: httpx.Response, body: str = "") -> None:
        if resp.status_code == 404 and self._session_id:
            self._ready = False
            self._session_id = None
            raise McpError(f"MCP server {self.spec.name!r} session expired (404); retry")
        if resp.status_code == 401:
            raise McpError(
                f"MCP server {self.spec.name!r} returned 401 Unauthorized "
                "(set headers.Authorization in mcp.json)"
            )
        if resp.status_code >= 400:
            raise McpError(
                f"MCP server {self.spec.name!r} HTTP {resp.status_code}: {body[:300]}"
            )

    async def _post(self, msg: dict[str, Any], *, timeout: float) -> httpx.Response:
        client = self._client_or_raise()
        try:
            return await client.send(
                client.build_request(
                    "POST",
                    self.spec.url,
                    content=json.dumps(msg, separators=(",", ":")).encode("utf-8"),
                    headers=self._headers(json_body=True),
                    timeout=timeout,
                ),
                stream=True,
            )
        except httpx.HTTPError as e:
            raise McpError(f"MCP server {self.spec.name!r} request failed: {e}") from e

    async def _send_notification(self, msg: dict[str, Any]) -> None:
        resp = await self._post(msg, timeout=self.spec.startup_timeout_s)
        try:
            await resp.aread()
            self._raise_for_status(resp, resp.text)
        finally:
            await resp.aclose()

    async def _send_response(self, msg: dict[str, Any]) -> None:
        try:
            resp = await self._post(msg, timeout=self.spec.startup_timeout_s)
        except McpError as e:
            log.debug("MCP server %s: could not deliver response: %s", self.spec.name, e)
            return
        await resp.aclose()

    async def _request(self, msg: dict[str, Any], *, timeout: float) -> Any:
        req_id = int(msg["id"])
        is_init = msg.get("method") == "initialize"
        resp = await self._post(msg, timeout=timeout)
        try:
            if is_init:
                sid = resp.headers.get(SESSION_HEADER)
                if sid:
                    self._session_id = sid
            ctype = resp.headers.get("content-type", "").split(";")[0].strip().lower()
            if resp.status_code >= 400 or ctype not in {EVENT_STREAM, "application/json"}:
                await resp.aread()
                self._raise_for_status(resp, resp.text)
                if resp.status_code == 202:
                    raise McpError(
                        f"MCP server {self.spec.name!r} accepted {msg.get('method')} "
                        "without a response"
                    )
                raise McpError(
                    f"MCP server {self.spec.name!r} returned unexpected "
                    f"content-type {ctype or 'none'}"
                )
            if ctype == "application/json":
                await resp.aread()
                payload = resp.json()
                messages = payload if isinstance(payload, list) else [payload]
                for raw in messages:
                    if not isinstance(raw, dict):
                        continue
                    matched, result = await self._dispatch(raw, req_id)
                    if matched:
                        return result
                raise McpError(
                    f"MCP server {self.spec.name!r} response omitted id {req_id}"
                )
            async for data in _iter_sse_data(resp.aiter_lines()):
                try:
                    raw = json.loads(data)
                except json.JSONDecodeError:
                    log.debug("MCP server %s: non-JSON SSE data ignored", self.spec.name)
                    continue
                if not isinstance(raw, dict):
                    continue
                matched, result = await self._dispatch(raw, req_id)
                if matched:
                    return result
            raise McpError(
                f"MCP server {self.spec.name!r} closed the SSE stream before answering id {req_id}"
            )
        finally:
            await resp.aclose()

    async def _listen(self) -> None:
        """Optional server → client GET stream. Servers may answer 405; that is fine."""
        client = self._client
        if client is None:
            return
        headers = self._headers()
        headers["Accept"] = EVENT_STREAM
        try:
            async with client.stream(
                "GET", self.spec.url, headers=headers, timeout=httpx.Timeout(10.0, read=None)
            ) as resp:
                if resp.status_code != 200:
                    return
                ctype = resp.headers.get("content-type", "").split(";")[0].strip().lower()
                if ctype != EVENT_STREAM:
                    return
                async for data in _iter_sse_data(resp.aiter_lines()):
                    try:
                        raw = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(raw, dict):
                        _consumed, reply = self._handle_unsolicited(raw)
                        if reply is not None:
                            await self._send_response(reply)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.debug("MCP server %s GET stream ended: %s", self.spec.name, e)


def make_session(
    spec: McpServerSpec, *, client_factory: ClientFactory | None = None
) -> McpSession:
    """Pick the transport for ``spec``."""
    if spec.transport == "http":
        return HttpMcpSession(spec, client_factory=client_factory)
    return StdioMcpSession(spec)
