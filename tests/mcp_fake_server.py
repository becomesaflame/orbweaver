"""Local stdio MCP server for tests. No network, no third-party credentials."""

from __future__ import annotations

import json
import os
import sys
import time

SUPPORTED = {"2025-06-18", "2025-03-26", "2024-11-05"}

# Names the env_probe tool reports so tests can assert the gateway's own
# secrets never reach a server process.
PROBE_VARS = (
    "ANTHROPIC_API_KEY",
    "ORBWEAVER_JWT_SECRET",
    "TELEGRAM_BOT_TOKEN",
    "DATABASE_URL",
    "FAKE_MCP_TOKEN",
    "FAKE_ALLOWED",
    "FAKE_EXPLICIT",
    "PATH",
    "HOME",
)


def _read() -> dict | None:
    line = sys.stdin.readline()
    if not line:
        return None
    if line.lower().startswith("content-length:"):
        try:
            length = int(line.split(":", 1)[1].strip())
        except ValueError:
            return None
        while True:
            header = sys.stdin.readline()
            if header in {"", "\n", "\r\n"}:
                break
        body = sys.stdin.read(length)
        return json.loads(body)
    return json.loads(line)


def _write(msg: dict) -> None:
    sys.stdout.write(json.dumps(msg, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def _ok(mid, result: dict) -> None:
    _write({"jsonrpc": "2.0", "id": mid, "result": result})


def _text(mid, text: str) -> None:
    _ok(mid, {"content": [{"type": "text", "text": text}]})


TOOLS = [
    {
        "name": "echo",
        "description": "Echo text back.",
        "inputSchema": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    },
    {
        "name": "token_probe",
        "description": "Return FAKE_MCP_TOKEN from the process env.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "env_probe",
        "description": "Return selected env vars as JSON.",
        "inputSchema": {"type": "object", "properties": {}},
        "annotations": {"readOnlyHint": True, "title": "Environment probe"},
    },
    {
        "name": "sleep",
        "description": "Block for `seconds` before answering.",
        "inputSchema": {
            "type": "object",
            "properties": {"seconds": {"type": "number"}},
        },
    },
    {
        "name": "wipe",
        "description": "Pretend to delete everything.",
        "inputSchema": {"type": "object", "properties": {}},
        "annotations": {"destructiveHint": True, "readOnlyHint": False},
    },
]


def main() -> None:
    while True:
        msg = _read()
        if msg is None:
            return
        method = msg.get("method")
        mid = msg.get("id")
        if method == "initialize":
            wanted = (msg.get("params") or {}).get("protocolVersion")
            version = wanted if wanted in SUPPORTED else "2024-11-05"
            _ok(
                mid,
                {
                    "protocolVersion": version,
                    "capabilities": {"tools": {"listChanged": True}},
                    "serverInfo": {"name": "orbweaver-fake", "version": "1.0"},
                },
            )
        elif method == "notifications/initialized":
            continue
        elif method == "ping":
            if mid is not None:
                _ok(mid, {})
        elif method == "tools/list":
            _ok(mid, {"tools": TOOLS})
        elif method == "tools/call":
            params = msg.get("params") or {}
            name = params.get("name")
            args = params.get("arguments") or {}
            if name == "echo":
                _text(mid, str(args.get("text") or ""))
            elif name == "token_probe":
                _text(mid, os.environ.get("FAKE_MCP_TOKEN", ""))
            elif name == "env_probe":
                seen = {k: os.environ[k] for k in PROBE_VARS if k in os.environ}
                _text(mid, json.dumps(seen, sort_keys=True))
            elif name == "sleep":
                time.sleep(float(args.get("seconds") or 1.0))
                _text(mid, "woke")
            elif name == "wipe":
                _text(mid, "nothing deleted (fake)")
            else:
                _ok(
                    mid,
                    {
                        "content": [{"type": "text", "text": f"unknown tool {name}"}],
                        "isError": True,
                    },
                )
        elif mid is not None:
            _write(
                {
                    "jsonrpc": "2.0",
                    "id": mid,
                    "error": {"code": -32601, "message": f"unknown method {method}"},
                }
            )


if __name__ == "__main__":
    main()
