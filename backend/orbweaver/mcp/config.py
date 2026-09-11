"""Load MCP server specs from host, workspace, and ORBWEAVER_MCP_CONFIG."""

from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from orbweaver.mcp.environment import expand_env_refs

log = logging.getLogger(__name__)

DEFAULT_TIMEOUT_S = 30.0
DEFAULT_STARTUP_TIMEOUT_S = 8.0

_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_*?]*$")


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
    """One configured server. ``command`` selects stdio, ``url`` Streamable HTTP."""

    name: str
    command: str = ""
    args: tuple[str, ...] = ()
    env: dict[str, str] = field(default_factory=dict)
    cwd: str | None = None
    disabled: bool = False
    url: str = ""
    headers: dict[str, str] = field(default_factory=dict)
    env_passthrough: tuple[str, ...] = ()
    timeout_s: float = DEFAULT_TIMEOUT_S
    startup_timeout_s: float = DEFAULT_STARTUP_TIMEOUT_S

    @property
    def transport(self) -> str:
        return "http" if self.url and not self.command else "stdio"

    def fingerprint(self) -> str:
        return json.dumps(
            {
                "name": self.name,
                "command": self.command,
                "args": list(self.args),
                "env": self.env,
                "cwd": self.cwd,
                "disabled": self.disabled,
                "url": self.url,
                "headers": self.headers,
                "envPassthrough": list(self.env_passthrough),
                "timeout_s": self.timeout_s,
                "startup_timeout_s": self.startup_timeout_s,
            },
            sort_keys=True,
        )


@dataclass(frozen=True)
class McpConfig:
    servers: tuple[McpServerSpec, ...] = ()

    def enabled(self) -> tuple[McpServerSpec, ...]:
        return tuple(s for s in self.servers if not s.disabled and (s.command or s.url))

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


def _as_name_list(raw: Any) -> tuple[str, ...]:
    if isinstance(raw, str):
        raw = [p for p in re.split(r"[,\s]+", raw) if p]
    if not isinstance(raw, list):
        return ()
    out: list[str] = []
    for item in raw:
        name = str(item).strip()
        if name and _ENV_NAME.match(name) and name not in out:
            out.append(name)
    return tuple(out)


def _as_timeout(raw: Any, default: float) -> float:
    if raw is None or isinstance(raw, bool):
        return default
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


def _parse_server(name: str, raw: Any) -> McpServerSpec | None:
    if not isinstance(raw, dict):
        return None
    command = str(raw.get("command") or "").strip()
    url = str(raw.get("url") or "").strip()
    if url and not url.lower().startswith(("http://", "https://")):
        log.warning("MCP server %s: unsupported url %r (need http(s)://)", name, url)
        url = ""
    args_raw = raw.get("args") or []
    args: tuple[str, ...]
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
        url=url,
        headers=_as_str_map(raw.get("headers")),
        env_passthrough=_as_name_list(raw.get("envPassthrough")),
        timeout_s=_as_timeout(raw.get("timeout_s", raw.get("timeoutS")), DEFAULT_TIMEOUT_S),
        startup_timeout_s=_as_timeout(
            raw.get("startup_timeout_s", raw.get("startupTimeoutS")), DEFAULT_STARTUP_TIMEOUT_S
        ),
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
    headers = dict(base.headers)
    headers.update(incoming.headers)
    replaces_transport = bool(incoming.command or incoming.url)
    command = incoming.command if replaces_transport else base.command
    url = incoming.url if replaces_transport else base.url
    args = incoming.args if replaces_transport else base.args
    cwd = incoming.cwd if incoming.cwd is not None else base.cwd
    passthrough = tuple(dict.fromkeys([*base.env_passthrough, *incoming.env_passthrough]))
    return McpServerSpec(
        name=incoming.name,
        command=command,
        args=args,
        env=env,
        cwd=cwd,
        disabled=incoming.disabled,
        url=url,
        headers=headers,
        env_passthrough=passthrough,
        timeout_s=incoming.timeout_s,
        startup_timeout_s=incoming.startup_timeout_s,
    )


def _expand_spec(
    spec: McpServerSpec, environ: Mapping[str, str], global_passthrough: Iterable[str]
) -> McpServerSpec:
    """Resolve ``${VAR}`` in env/headers from the passthrough allowlist only."""
    allow = tuple(dict.fromkeys([*global_passthrough, *spec.env_passthrough]))
    env = {k: expand_env_refs(v, environ, allow, where=f"{spec.name} env {k}") for k, v in spec.env.items()}
    headers = {
        k: expand_env_refs(v, environ, allow, where=f"{spec.name} header {k}")
        for k, v in spec.headers.items()
    }
    return McpServerSpec(
        name=spec.name,
        command=spec.command,
        args=spec.args,
        env=env,
        cwd=spec.cwd,
        disabled=spec.disabled,
        url=spec.url,
        headers=headers,
        env_passthrough=allow,
        timeout_s=spec.timeout_s,
        startup_timeout_s=spec.startup_timeout_s,
    )


def load_mcp_config(
    workspace_root: Path | str | None = None,
    *,
    settings=None,
    environ: dict[str, str] | None = None,
    home: Path | None = None,
) -> McpConfig:
    """Merge ~/.orbweaver/mcp.json, workspace .orbweaver/mcp.json, then extra file.

    Later sources override command/args/cwd/disabled. ``env`` and ``headers``
    maps are deep-merged so host tokens survive a workspace command override.
    ``${VAR}`` references in ``env``/``headers`` values expand only from the
    per-server ``envPassthrough`` list, the file-level ``envPassthrough`` list,
    or ``ORBWEAVER_MCP_ENV_PASSTHROUGH``; anything else expands to "".
    """
    from orbweaver.config import settings as default_settings

    cfg = settings or default_settings
    env = environ if environ is not None else dict(os.environ)
    root = Path(workspace_root or getattr(cfg, "workspace_root", ".") or ".").resolve()
    home_dir = (home or Path.home()).resolve()

    merged: dict[str, McpServerSpec] = {}
    global_passthrough: list[str] = list(
        _as_name_list(getattr(cfg, "orbweaver_mcp_env_passthrough", "") or "")
    )
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
        data = _read_json(path)
        for name in _as_name_list(data.get("envPassthrough")):
            if name not in global_passthrough:
                global_passthrough.append(name)
        for name, spec in _servers_from_file(data).items():
            merged[name] = _merge_server(merged.get(name), spec)

    servers = tuple(_expand_spec(s, env, global_passthrough) for s in merged.values())
    return McpConfig(servers=servers)
