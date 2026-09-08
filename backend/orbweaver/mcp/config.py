"""Load MCP server specs from host, workspace, and ORBWEAVER_MCP_CONFIG."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def expand_path(raw: str, *, relative_to: Path | None = None) -> Path:
    text = (raw or "").strip()
    if not text:
        raise ValueError("empty path")
    if text.startswith("~/"):
        return Path(text).expanduser().resolve()
    if text.startswith("/") or (len(text) > 1 and text[1] == ":"):
        return Path(text).resolve()
    base = relative_to or Path.cwd()
    return (base / text).resolve()


@dataclass(frozen=True)
class McpServerSpec:
    name: str
    command: str
    args: tuple[str, ...] = ()
    env: dict[str, str] = field(default_factory=dict)
    cwd: str | None = None
    disabled: bool = False

    def fingerprint(self) -> str:
        return json.dumps(
            {
                "name": self.name,
                "command": self.command,
                "args": list(self.args),
                "env": self.env,
                "cwd": self.cwd,
                "disabled": self.disabled,
            },
            sort_keys=True,
        )


@dataclass(frozen=True)
class McpConfig:
    servers: tuple[McpServerSpec, ...] = ()

    def enabled(self) -> tuple[McpServerSpec, ...]:
        return tuple(s for s in self.servers if not s.disabled and s.command)

    def fingerprint(self) -> str:
        return json.dumps([s.fingerprint() for s in self.enabled()])


def _as_str_map(raw: Any) -> dict[str, str]:
    if not isinstance(raw, dict):
        return {}
    out: dict[str, str] = {}
    for key, val in raw.items():
        if val is None:
            continue
        out[str(key)] = str(val)
    return out


def _parse_server(name: str, raw: Any) -> McpServerSpec | None:
    if not isinstance(raw, dict):
        return None
    if raw.get("url") and not raw.get("command"):
        # HTTP/SSE transports are out of v1.
        return None
    command = str(raw.get("command") or "").strip()
    args_raw = raw.get("args") or []
    if isinstance(args_raw, str):
        args = (args_raw,)
    elif isinstance(args_raw, list):
        args = tuple(str(x) for x in args_raw)
    else:
        args = ()
    cwd_raw = raw.get("cwd")
    cwd = str(cwd_raw).strip() if cwd_raw else None
    disabled = bool(raw.get("disabled"))
    return McpServerSpec(
        name=name,
        command=command,
        args=args,
        env=_as_str_map(raw.get("env")),
        cwd=cwd,
        disabled=disabled,
    )


def _servers_from_file(data: dict[str, Any]) -> dict[str, McpServerSpec]:
    blob = data.get("mcpServers")
    if not isinstance(blob, dict):
        return {}
    out: dict[str, McpServerSpec] = {}
    for raw_name, raw_spec in blob.items():
        name = str(raw_name).strip()
        if not name:
            continue
        spec = _parse_server(name, raw_spec)
        if spec is not None:
            out[name] = spec
    return out


def _merge_server(base: McpServerSpec | None, incoming: McpServerSpec) -> McpServerSpec:
    if base is None:
        return incoming
    env = dict(base.env)
    env.update(incoming.env)
    command = incoming.command or base.command
    args = incoming.args if incoming.command else base.args
    cwd = incoming.cwd if incoming.cwd is not None else base.cwd
    return McpServerSpec(
        name=incoming.name,
        command=command,
        args=args,
        env=env,
        cwd=cwd,
        disabled=incoming.disabled,
    )


def load_mcp_config(
    workspace_root: Path | str | None = None,
    *,
    settings=None,
    environ: dict[str, str] | None = None,
    home: Path | None = None,
) -> McpConfig:
    """Merge ~/.orbweaver/mcp.json, workspace .orbweaver/mcp.json, then extra file.

    Later sources override command/args/cwd/disabled. ``env`` maps are deep-merged
    so host tokens survive a workspace command override.
    """
    from orbweaver.config import settings as default_settings

    cfg = settings or default_settings
    env = environ if environ is not None else dict(os.environ)
    root = Path(workspace_root or getattr(cfg, "workspace_root", ".") or ".").resolve()
    home_dir = (home or Path.home()).resolve()

    merged: dict[str, McpServerSpec] = {}
    sources: list[Path] = [
        home_dir / ".orbweaver" / "mcp.json",
        root / ".orbweaver" / "mcp.json",
    ]
    extra = (getattr(cfg, "orbweaver_mcp_config", None) or env.get("ORBWEAVER_MCP_CONFIG") or "").strip()
    if extra:
        sources.append(expand_path(extra, relative_to=root))

    for path in sources:
        if not path.is_file():
            continue
        for name, spec in _servers_from_file(_read_json(path)).items():
            merged[name] = _merge_server(merged.get(name), spec)

    return McpConfig(servers=tuple(merged.values()))
