"""Allow / ask / deny rule matching and always-deny path checks."""

from __future__ import annotations

import fnmatch
import re
from pathlib import Path
from typing import Any

from orbweaver.config import settings

SAFE_ALLOWLIST = frozenset(
    {
        "Read",
        "Glob",
        "Grep",
        "AskUser",
        "MemorySearch",
        "MemoryRemember",
        "MemoryPin",
        "MemoryForget",
        "SendPhoto",
        "GenerateImage",
        "WebSearch",
    }
)

DANGEROUS_BASH_PREFIXES = (
    "python",
    "python3",
    "python2",
    "node",
    "deno",
    "tsx",
    "ruby",
    "perl",
    "php",
    "lua",
    "npx",
    "bunx",
    "npm run",
    "yarn run",
    "pnpm run",
    "bun run",
    "bash",
    "sh",
    "zsh",
    "fish",
    "eval",
    "exec",
    "env",
    "xargs",
    "sudo",
    "ssh",
)

_CRITICAL_RM = re.compile(
    r"""
    \b(?:rm|rmdir)\b
    .*
    (?:
        (?:\s|^)(?:-[\w-]*\s+)* /
        (?:\s|$|")
      | (?:\s|^)~(?:/|\s|$|")
      | \$HOME
      | /home/[^/\s"']+(?:\s|$|")
    )
    """,
    re.VERBOSE | re.IGNORECASE,
)

ALWAYS_DENY_NAMES = {
    ".env",
    ".env.local",
    ".env.production",
    "id_rsa",
    "id_ed25519",
    "id_ecdsa",
    "id_dsa",
}


def parse_rules(blob: str) -> list[tuple[str, str | None]]:
    """Parse 'Tool' or 'Tool(pattern)' lines into (tool, pattern|None)."""
    out: list[tuple[str, str | None]] = []
    for raw in (blob or "").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "(" in line and line.endswith(")"):
            tool, rest = line.split("(", 1)
            out.append((tool.strip(), rest[:-1].strip()))
        else:
            out.append((line, None))
    return out


def _subject(tool: str, inp: dict[str, Any]) -> str:
    if tool == "Bash":
        return str(inp.get("command") or "")
    if tool in {"Read", "Write", "ProposePatch", "SendPhoto"}:
        return str(inp.get("path") or "")
    if tool == "GenerateImage":
        return str(inp.get("prompt") or inp.get("path") or "")
    if tool == "WebFetch":
        return str(inp.get("url") or "")
    if tool == "WebSearch":
        return str(inp.get("query") or "")
    if tool == "SpawnSubagent":
        return str(inp.get("task") or "")
    return json_fallback(inp)


def json_fallback(inp: dict[str, Any]) -> str:
    import json

    return json.dumps(inp, default=str)


def rule_matches(tool: str, inp: dict[str, Any], rule_tool: str, pattern: str | None) -> bool:
    if tool != rule_tool:
        return False
    if pattern is None or pattern in {"", "*"}:
        return True
    subject = _subject(tool, inp)
    return fnmatch.fnmatch(subject, pattern) or subject.startswith(pattern.rstrip("*"))


def is_dangerous_allow(rule_tool: str, pattern: str | None) -> bool:
    if rule_tool != "Bash":
        return False
    if pattern is None or pattern in {"", "*"}:
        return True
    content = pattern.strip().lower()
    if content == "*":
        return True
    for prefix in DANGEROUS_BASH_PREFIXES:
        p = prefix.lower()
        if content in {p, f"{p}:*", f"{p}*", f"{p} *"}:
            return True
        if content.startswith(f"{p} -") and content.endswith("*"):
            return True
    return False


def matching_rule(
    blob: str, tool: str, inp: dict[str, Any], *, strip_dangerous: bool = False
) -> tuple[str, str | None] | None:
    for rule_tool, pattern in parse_rules(blob):
        if strip_dangerous and is_dangerous_allow(rule_tool, pattern):
            continue
        if rule_matches(tool, inp, rule_tool, pattern):
            return rule_tool, pattern
    return None


_GIT_PUSH = re.compile(r"\bgit\s+push\b", re.IGNORECASE)
_FORCE_PUSH_FLAG = re.compile(
    r"(?:^|[\s;|&])(?:--force(?:-with-lease)?|-f)(?=[\s;|&]|$)",
    re.IGNORECASE,
)
_FORCE_REFSPEC = re.compile(
    r"\bgit\s+push\b[\s\S]*?(?:origin|upstream)\s+\+\S",
    re.IGNORECASE,
)
_PROTECTED_REF_PUSH = re.compile(
    r"""
    \bgit\s+push\b
    [\s\S]*?
    (?:
        (?:origin|upstream)
        \s+
        \+?
        (?:
            (?:refs/heads/)?(?:main|master)
          | \S+:(?:refs/heads/)?(?:main|master)
          | :(?:refs/heads/)?(?:main|master)
        )
      | HEAD:(?:refs/heads/)?(?:main|master)
    )
    (?:\s|$)
    """,
    re.VERBOSE | re.IGNORECASE,
)


def is_critical_rm(command: str) -> bool:
    return bool(_CRITICAL_RM.search(command or ""))


def is_protected_git_push(command: str) -> bool:
    """True for force-push or a push to main/master (ask-gated, never auto-allowed)."""
    text = command or ""
    if not _GIT_PUSH.search(text):
        return False
    if _FORCE_PUSH_FLAG.search(text) or _FORCE_REFSPEC.search(text):
        return True
    return bool(_PROTECTED_REF_PUSH.search(text))


def path_is_always_denied(rel: str) -> bool:
    name = Path(rel).name.lower()
    parts = Path(rel).parts
    lowered = [p.lower() for p in parts]
    if name in ALWAYS_DENY_NAMES:
        return True
    if name == "mcp.json" and ".orbweaver" in lowered:
        return True
    if name.endswith((".pem", ".key")):
        return True
    if ".ssh" in lowered or ".gnupg" in lowered:
        return True
    return str(rel).startswith("/etc") or "/etc/" in str(rel)


def _workspace_root(workspace) -> Path | None:
    root = getattr(workspace, "root", None)
    if root is None:
        local = getattr(workspace, "local", None)
        root = getattr(local, "root", None)
    return Path(root) if root is not None else None


def in_working_set(rel: str, workspace, *, write: bool = False) -> bool:
    """True when the path is in the sandbox working set (rw roots if write)."""
    if not rel or path_is_always_denied(rel):
        return False
    root = _workspace_root(workspace)
    if root is None:
        return not str(rel).startswith("/")
    from orbweaver.sandbox.policy import in_roots, load_sandbox_policy

    try:
        if rel.startswith(("/", "~")):
            path = Path(rel).expanduser().resolve()
        else:
            path = (root / rel).resolve()
        policy = load_sandbox_policy(root)
        roots = policy.readwrite_roots(root) if write else policy.working_set_roots(root)
        if not in_roots(path, roots):
            return False
    except (OSError, ValueError):
        return False
    return True


def in_project_path(rel: str, workspace) -> bool:
    """True when the path is writable in the working set."""
    return in_working_set(rel, workspace, write=True)


def deny_rules() -> str:
    return settings.orbweaver_permission_deny


def ask_rules() -> str:
    return settings.orbweaver_permission_ask


def allow_rules() -> str:
    return settings.orbweaver_permission_allow
