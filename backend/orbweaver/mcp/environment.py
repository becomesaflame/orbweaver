"""Minimal environment for stdio MCP servers: allowlist in, gateway secrets out.

The gateway process holds ``ANTHROPIC_API_KEY``, ``ORBWEAVER_JWT_SECRET``,
``TELEGRAM_BOT_TOKEN``, ``DATABASE_URL`` and friends. Third-party MCP server
processes get none of that: only ``MCP_BASE_ENV_ALLOW``, explicit ``env`` from
``mcp.json``, and names the operator listed in ``envPassthrough``.

The exclude patterns mirror ``sandbox/environment.py`` on the #93 branch
(``fix/sandbox-clearenv``); they are duplicated here because that module is
not on ``main`` yet. Keep the two lists in sync.
"""

from __future__ import annotations

import fnmatch
import logging
import re
from collections.abc import Iterable, Mapping

log = logging.getLogger(__name__)

# Passed through to every stdio server. fnmatch patterns, case-insensitive.
MCP_BASE_ENV_ALLOW: tuple[str, ...] = (
    "PATH",
    "HOME",
    "USER",
    "LOGNAME",
    "LANG",
    "LANGUAGE",
    "LC_*",
    "TERM",
    "TMPDIR",
    "TZ",
)

# Never inherited implicitly, and never honoured from ``envPassthrough`` either:
# these are the gateway's own configuration namespaces. Put a copy under a
# different host variable and map it in ``env`` if a server truly needs one.
MCP_ENV_NEVER_PASSTHROUGH: tuple[str, ...] = (
    "ORBWEAVER_*",
    "DATABASE_URL",
    "ANTHROPIC_*",
    "OPENAI_*",
    "OPENROUTER_*",
    "TELEGRAM_*",
    "HINDSIGHT_*",
)

# Codex CLI's default excludes plus the gateway namespaces. Applied to the base
# allowlist (defence in depth; ``PATH`` etc. never match) and used to refuse
# wildcard passthrough entries such as ``"*TOKEN*"``.
MCP_ENV_EXCLUDE: tuple[str, ...] = (
    "*KEY*",
    "*SECRET*",
    "*TOKEN*",
    "*PASSWORD*",
    *MCP_ENV_NEVER_PASSTHROUGH,
)

FALLBACK_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

_REF = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _matches_any(name: str, patterns: Iterable[str]) -> bool:
    lowered = name.lower()
    return any(fnmatch.fnmatchcase(lowered, pat.lower()) for pat in patterns if pat)


def env_excluded(name: str) -> bool:
    """True when ``name`` must never reach an MCP server implicitly."""
    return _matches_any(name, MCP_ENV_EXCLUDE)


def passthrough_allowed(name: str, allow: Iterable[str]) -> bool:
    """True when ``name`` may be forwarded because the operator listed it.

    Entries are exact names or fnmatch patterns. Patterns that would match a
    gateway secret namespace (``*``, ``*TOKEN*``, ``ORBWEAVER_*``) are refused
    so a wide allowlist cannot become the leak this module prevents.
    """
    if _matches_any(name, MCP_ENV_NEVER_PASSTHROUGH):
        return False
    for pat in allow:
        if not pat:
            continue
        if ("*" in pat or "?" in pat) and _pattern_too_wide(pat):
            continue
        if fnmatch.fnmatchcase(name.lower(), pat.lower()):
            return True
    return False


def _pattern_too_wide(pat: str) -> bool:
    """A passthrough glob is too wide when it matches a name every exclude protects."""
    probes = ("orbweaver_jwt_secret", "anthropic_api_key", "telegram_bot_token", "database_url")
    return any(fnmatch.fnmatchcase(p, pat.lower()) for p in probes)


def expand_env_refs(
    value: str, environ: Mapping[str, str], allow: Iterable[str], *, where: str = ""
) -> str:
    """Replace ``${VAR}`` with ``environ[VAR]`` when VAR is on the passthrough list.

    References to names that are not allowlisted (or not set) expand to "" and
    are logged once per reference so a misconfigured token fails loudly on the
    server side instead of silently shipping the literal ``${VAR}`` text.
    """
    allow = tuple(allow)

    def _sub(match: re.Match[str]) -> str:
        name = match.group(1)
        if not passthrough_allowed(name, allow):
            log.warning(
                "MCP %s: ${%s} is not in envPassthrough; expanding to empty string", where, name
            )
            return ""
        return environ.get(name, "")

    return _REF.sub(_sub, value)


def build_mcp_env(
    environ: Mapping[str, str],
    *,
    explicit: Mapping[str, str] | None = None,
    passthrough: Iterable[str] = (),
) -> dict[str, str]:
    """Environment for a stdio MCP server process.

    Order: base allowlist from ``environ`` (minus excludes), then names the
    operator listed in ``passthrough`` (minus gateway namespaces), then the
    explicit ``env`` map from ``mcp.json`` which always wins.
    """
    allow = tuple(passthrough)
    out: dict[str, str] = {}
    for name in sorted(environ):
        base = _matches_any(name, MCP_BASE_ENV_ALLOW) and not env_excluded(name)
        if base or (allow and passthrough_allowed(name, allow)):
            out[name] = environ[name]
    for key, val in (explicit or {}).items():
        out[str(key)] = str(val)
    if not out.get("PATH"):
        out["PATH"] = FALLBACK_PATH
    return out
