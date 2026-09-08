"""CLI / stub diagnostics for files the agent just edited."""

from __future__ import annotations

import ast
import json
import shlex
from pathlib import Path
from typing import Any

from orbweaver.config import settings
from orbweaver.store import Event

EDIT_TOOLS = frozenset({"Write", "StrReplace", "ProposePatch", "NotebookEdit"})
MAX_PATHS = 20


def lint_paths_from_input(inp: dict[str, Any]) -> list[str]:
    raw = inp.get("paths", inp.get("path"))
    if raw is None:
        return []
    if isinstance(raw, str):
        text = raw.strip()
        return [text] if text else []
    out: list[str] = []
    for item in raw:
        text = str(item).strip()
        if text:
            out.append(text)
        if len(out) >= MAX_PATHS:
            break
    return out


def recent_edited_paths(events: list[Event] | None, limit: int = MAX_PATHS) -> list[str]:
    paths: list[str] = []
    seen: set[str] = set()
    for ev in reversed(events or []):
        if ev.kind != "tool_call":
            continue
        if str(ev.payload.get("name") or "") not in EDIT_TOOLS:
            continue
        path = str((ev.payload.get("input") or {}).get("path") or "").strip()
        if not path or path in seen:
            continue
        seen.add(path)
        paths.append(path)
        if len(paths) >= limit:
            break
    paths.reverse()
    return paths


def parse_compiler_output(text: str, source: str = "linter") -> list[dict[str, Any]]:
    """Parse `file:line:col: message` or `file:line: message` lines."""
    diags: list[dict[str, Any]] = []
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split(":", 3)
        if len(parts) < 3:
            continue
        try:
            lineno = int(parts[1])
        except ValueError:
            continue
        col = 0
        message = parts[2].strip()
        if len(parts) == 4:
            try:
                col = int(parts[2])
                message = parts[3].strip()
            except ValueError:
                message = ":".join(parts[2:]).strip()
        diags.append(
            {
                "path": parts[0],
                "line": lineno,
                "col": col,
                "severity": "error",
                "message": message,
                "source": source,
            }
        )
    return diags


def _stub_python(path: str, source: str) -> list[dict[str, Any]]:
    try:
        ast.parse(source, filename=path)
    except SyntaxError as e:
        return [
            {
                "path": path,
                "line": e.lineno or 1,
                "col": e.offset or 0,
                "severity": "error",
                "message": e.msg,
                "source": "python-ast",
            }
        ]
    return []


def _sandbox_failed(output: str) -> bool:
    text = (output or "").lower()
    return "sandbox_unavailable" in text or (
        "bwrap" in text and "operation not permitted" in text
    )


def run_configured_linter(workspace, command: str) -> str:
    """Run ORBWEAVER_LINTER. Prefer bubblewrap; fall back if bwrap cannot start.

    Does not change the Bash tool's production sandbox policy. The fallback uses
    workspace.bash(..., sandbox=False) only for this operator-configured command.
    """
    output = workspace.bash(command)
    if parse_compiler_output(output) or not _sandbox_failed(output):
        return output
    return workspace.bash(command, sandbox=False)


def read_lints(workspace, inp: dict[str, Any], events: list[Event] | None = None) -> str:
    paths = lint_paths_from_input(inp)
    if not paths:
        paths = recent_edited_paths(events)
    if not paths:
        return json.dumps({"diagnostics": [], "note": "no files to lint"})

    command = (settings.orbweaver_linter or "").strip()
    if command:
        quoted = " ".join(shlex.quote(p) for p in paths)
        filled = command.replace("{paths}", quoted) if "{paths}" in command else f"{command} {quoted}"
        output = run_configured_linter(workspace, filled)
        parsed = parse_compiler_output(output, source="linter")
        return json.dumps(
            {"diagnostics": parsed, "output": output[-20_000:], "command": filled}
        )

    diags: list[dict[str, Any]] = []
    notes: list[str] = []
    for path in paths:
        try:
            text = workspace.read(path)
        except (OSError, PermissionError, UnicodeDecodeError, IsADirectoryError) as e:
            diags.append(
                {
                    "path": path,
                    "line": 1,
                    "col": 0,
                    "severity": "error",
                    "message": str(e),
                    "source": "read",
                }
            )
            continue
        if Path(path).suffix == ".py":
            diags.extend(_stub_python(path, text))
        else:
            notes.append(f"{path}: no stub linter (set ORBWEAVER_LINTER)")
    payload: dict[str, Any] = {"diagnostics": diags}
    if notes:
        payload["note"] = "; ".join(notes)
    return json.dumps(payload)
