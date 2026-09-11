"""Explicit environment for sandboxed Bash: allowlist in, gateway secrets out."""

from __future__ import annotations

import fnmatch
from collections.abc import Iterable, Mapping
from pathlib import Path

# Passed through from the gateway process by default. Glob patterns (fnmatch,
# case-insensitive) are allowed so LC_ALL, LC_CTYPE, ... ride on one entry.
DEFAULT_ENV_ALLOW: tuple[str, ...] = (
    "PATH",
    "HOME",
    "USER",
    "LOGNAME",
    "SHELL",
    "TERM",
    "COLORTERM",
    "LANG",
    "LANGUAGE",
    "LC_*",
    "TZ",
    "TMPDIR",
    # Host proxy settings. `wrap_command_with_proxy` exports its own values
    # inside the sandbox; these only matter for permissions: ["full_network"].
    "http_proxy",
    "https_proxy",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "NO_PROXY",
    "no_proxy",
)

# Never passed through, even when an operator's `env.allow` names them.
# Codex CLI's default excludes plus the gateway's own configuration namespace.
DEFAULT_ENV_EXCLUDE: tuple[str, ...] = (
    "*KEY*",
    "*SECRET*",
    "*TOKEN*",
    "*PASSWORD*",
    "ORBWEAVER_*",
    "DATABASE_URL",
    "ANTHROPIC_*",
    "OPENAI_*",
    "OPENROUTER_*",
    "TELEGRAM_*",
    "HINDSIGHT_*",
)

FALLBACK_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"


def _matches_any(name: str, patterns: Iterable[str]) -> bool:
    lowered = name.lower()
    return any(fnmatch.fnmatchcase(lowered, pat.lower()) for pat in patterns if pat)


def env_excluded(name: str, exclude: Iterable[str] = DEFAULT_ENV_EXCLUDE) -> bool:
    return _matches_any(name, exclude)


def build_sandbox_env(
    environ: Mapping[str, str],
    *,
    allow: Iterable[str] = (),
    exclude: Iterable[str] = DEFAULT_ENV_EXCLUDE,
    granted_sockets: Iterable[Path] = (),
) -> dict[str, str]:
    """Return the variables a sandboxed command may see.

    Allowlist: DEFAULT_ENV_ALLOW plus `allow` (sandbox.json `env.allow`,
    ORBWEAVER_SANDBOX_ENV_ALLOW). Excludes win over every allow entry so a
    widened allowlist cannot leak `*_API_KEY` or `ORBWEAVER_JWT_SECRET`.
    SSH_AUTH_SOCK is included only when that socket is in `granted_sockets`.
    """
    patterns = [*DEFAULT_ENV_ALLOW, *allow]
    out: dict[str, str] = {}
    for name in sorted(environ):
        if name == "SSH_AUTH_SOCK":
            continue  # granted-socket check below, never via allow patterns
        if not _matches_any(name, patterns):
            continue
        if env_excluded(name, exclude):
            continue
        out[name] = environ[name]
    agent = (environ.get("SSH_AUTH_SOCK") or "").strip()
    if agent and not env_excluded("SSH_AUTH_SOCK", exclude):
        granted = {str(p) for p in granted_sockets}
        if agent in granted or str(Path(agent)) in granted:
            out["SSH_AUTH_SOCK"] = agent
    if not out.get("PATH"):
        out["PATH"] = FALLBACK_PATH
    return out


def setenv_args(env: Mapping[str, str]) -> list[str]:
    """`--clearenv` followed by one `--setenv` per variable, for bwrap argv."""
    args: list[str] = ["--clearenv"]
    for name in sorted(env):
        args.extend(["--setenv", name, env[name]])
    return args
