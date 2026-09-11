"""In-process Streamable HTTP MCP server (Starlette) for client tests.

Served through ``httpx.ASGITransport`` so no socket is opened. Records every
request so tests can assert headers (``Mcp-Session-Id``, ``MCP-Protocol-Version``,
``Authorization``) and the DELETE on close.
"""

from __future__ import annotations

import json
from typing import Any

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse, Response
from starlette.routing import Route

SESSION_ID = "sess-abc-123"
SERVER_PROTOCOL = "2025-06-18"

BASE_TOOLS: list[dict[str, Any]] = [
    {
        "name": "lookup",
        "description": "Look something up.",
        "inputSchema": {
            "type": "object",
            "properties": {"q": {"type": "string"}},
            "required": ["q"],
        },
        "annotations": {"readOnlyHint": True, "destructiveHint": False, "title": "Lookup"},
    },
    {
        "name": "purge",
        "description": "Purge records.",
        "inputSchema": {"type": "object", "properties": {}},
        "annotations": {"destructiveHint": True},
    },
    {
        "name": "stream_echo",
        "description": "Echo text, answered over SSE with a progress notification first.",
        "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}}},
    },
    {
        "name": "add_tool",
        "description": "Register `extra` and announce tools/list_changed over SSE.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "ask_me",
        "description": "Server asks the client to sample before answering.",
        "inputSchema": {"type": "object", "properties": {}},
    },
]

EXTRA_TOOL: dict[str, Any] = {
    "name": "extra",
    "description": "Appeared after list_changed.",
    "inputSchema": {"type": "object", "properties": {}},
    "annotations": {"readOnlyHint": True},
}

RESOURCES = [
    {"uri": "memo://hello", "name": "hello", "mimeType": "text/plain"},
]


def _sse(messages: list[dict[str, Any]]) -> Response:
    body = "".join(
        f"event: message\ndata: {json.dumps(m, separators=(',', ':'))}\n\n" for m in messages
    )
    return Response(body, media_type="text/event-stream")


def _result(mid: Any, result: dict[str, Any]) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": mid, "result": result}


def _text(mid: Any, text: str) -> dict[str, Any]:
    return _result(mid, {"content": [{"type": "text", "text": text}]})


class FakeHttpMcp:
    def __init__(self, *, require_bearer: str | None = None):
        self.requests: list[dict[str, Any]] = []
        self.deleted: list[str] = []
        self.tools = list(BASE_TOOLS)
        self.require_bearer = require_bearer
        self.pending_server_requests: list[dict[str, Any]] = []
        self.client_responses: list[dict[str, Any]] = []
        self.app = Starlette(
            routes=[
                Route("/mcp", self.handle, methods=["POST", "GET", "DELETE"]),
            ]
        )

    def _record(self, request: Request, body: Any) -> dict[str, Any]:
        entry = {
            "method": request.method,
            "headers": {k.lower(): v for k, v in request.headers.items()},
            "body": body,
        }
        self.requests.append(entry)
        return entry

    async def handle(self, request: Request) -> Response:
        if request.method == "GET":
            self._record(request, None)
            return PlainTextResponse("no standalone stream", status_code=405)
        if request.method == "DELETE":
            self._record(request, None)
            self.deleted.append(request.headers.get("mcp-session-id", ""))
            return Response(status_code=204)
        raw = await request.body()
        msg = json.loads(raw) if raw else {}
        self._record(request, msg)
        if self.require_bearer and request.headers.get("authorization") != f"Bearer {self.require_bearer}":
            return PlainTextResponse("unauthorized", status_code=401)
        if "method" not in msg:
            # Client response to a server request (sampling etc.).
            self.client_responses.append(msg)
            return Response(status_code=202)
        method = msg["method"]
        mid = msg.get("id")
        params = msg.get("params") or {}
        if method == "initialize":
            resp = JSONResponse(
                _result(
                    mid,
                    {
                        "protocolVersion": SERVER_PROTOCOL,
                        "capabilities": {"tools": {"listChanged": True}, "resources": {}},
                        "serverInfo": {"name": "fake-http", "version": "1"},
                    },
                )
            )
            resp.headers["Mcp-Session-Id"] = SESSION_ID
            return resp
        if request.headers.get("mcp-session-id") != SESSION_ID:
            return PlainTextResponse("missing session", status_code=400)
        if mid is None:
            return Response(status_code=202)  # notification
        if method == "tools/list":
            return JSONResponse(_result(mid, {"tools": self.tools}))
        if method == "resources/list":
            return JSONResponse(_result(mid, {"resources": RESOURCES}))
        if method == "resources/read":
            uri = params.get("uri")
            if uri != "memo://hello":
                return JSONResponse(
                    {"jsonrpc": "2.0", "id": mid, "error": {"code": -32002, "message": "not found"}}
                )
            return JSONResponse(
                _result(
                    mid,
                    {"contents": [{"uri": uri, "mimeType": "text/plain", "text": "hello resource"}]},
                )
            )
        if method == "tools/call":
            name = params.get("name")
            args = params.get("arguments") or {}
            if name == "lookup":
                return JSONResponse(_text(mid, f"found {args.get('q')}"))
            if name == "purge":
                return JSONResponse(_text(mid, "purged"))
            if name == "stream_echo":
                return _sse(
                    [
                        {
                            "jsonrpc": "2.0",
                            "method": "notifications/progress",
                            "params": {"progressToken": "t1", "progress": 1, "total": 2},
                        },
                        _text(mid, f"sse:{args.get('text')}"),
                    ]
                )
            if name == "add_tool":
                if EXTRA_TOOL not in self.tools:
                    self.tools.append(EXTRA_TOOL)
                return _sse(
                    [
                        {"jsonrpc": "2.0", "method": "notifications/tools/list_changed"},
                        _text(mid, "added"),
                    ]
                )
            if name == "extra":
                return JSONResponse(_text(mid, "extra ok"))
            if name == "ask_me":
                return _sse(
                    [
                        {
                            "jsonrpc": "2.0",
                            "id": "srv-1",
                            "method": "sampling/createMessage",
                            "params": {"messages": []},
                        },
                        _text(mid, "answered anyway"),
                    ]
                )
            return JSONResponse(
                _result(
                    mid,
                    {"content": [{"type": "text", "text": f"unknown {name}"}], "isError": True},
                )
            )
        return JSONResponse(
            {"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": f"no {method}"}}
        )
