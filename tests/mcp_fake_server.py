"""Local stdio MCP server for tests. No network, no third-party credentials."""

from __future__ import annotations

import json
import os
import sys


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


def main() -> None:
    while True:
        msg = _read()
        if msg is None:
            return
        method = msg.get("method")
        mid = msg.get("id")
        if method == "initialize":
            _ok(
                mid,
                {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "orbweaver-fake", "version": "1.0"},
                },
            )
        elif method == "notifications/initialized":
            continue
        elif method == "ping":
            if mid is not None:
                _ok(mid, {})
        elif method == "tools/list":
            _ok(
                mid,
                {
                    "tools": [
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
                    ]
                },
            )
        elif method == "tools/call":
            params = msg.get("params") or {}
            name = params.get("name")
            args = params.get("arguments") or {}
            if name == "echo":
                _ok(
                    mid,
                    {"content": [{"type": "text", "text": str(args.get("text") or "")}]},
                )
            elif name == "token_probe":
                _ok(
                    mid,
                    {
                        "content": [
                            {"type": "text", "text": os.environ.get("FAKE_MCP_TOKEN", "")}
                        ]
                    },
                )
            else:
                _ok(
                    mid,
                    {
                        "content": [{"type": "text", "text": f"unknown tool {name}"}],
                        "isError": True,
                    },
                )


if __name__ == "__main__":
    main()
