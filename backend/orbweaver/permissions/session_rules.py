"""Per-session allow rules granted by an "Allow for session" approval.

A rule is ``{"tool": <name>, "subject": <str>, "permissions": [...]}`` stored on the
session entity under ``jsonld["permission_rules"]``. The subject is deliberately
simple so the user can predict what a session-wide allow covers:

- ``Bash``: the command prefix, i.e. the first two whitespace-separated words
  (``git push``, ``rm -rf``, ``make``). A later command matches when its first
  words are identical. The rule also records the sandbox escalation flags
  (``permissions``: ``all`` / ``full_network``); a later call must not ask for
  more than the approved call did.
- File tools (``Read``, ``Write``, ``StrReplace``, ``NotebookEdit``, ``Delete``,
  ``ProposePatch``, ``SendPhoto``, ``GenerateImage``): the parent directory of the
  path. A later path matches when it is that directory or below it.
- ``WebFetch`` / ``Browser``: the URL host. A later URL matches on the same host.
- Anything else (``WebSearch``, ``SpawnSubagent``, ``mcp_*`` ...): ``*`` — any
  input for that tool.

Session rules are consulted in :func:`orbweaver.permissions.pipeline.can_use_tool`
right before the classifier, so deny rules, ask rules (including protected git
pushes), always-denied paths and the sandbox-unavailable refusal still win. A
critical ``rm`` (home directory, ``/``) is never covered by a session rule.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import PurePosixPath
from typing import Any
from urllib.parse import urlparse

_PATH_TOOLS = frozenset(
    {
        "Read",
        "Write",
        "StrReplace",
        "NotebookEdit",
        "Delete",
        "ProposePatch",
        "SendPhoto",
        "GenerateImage",
    }
)
_URL_TOOLS = frozenset({"WebFetch", "Browser"})
BASH_PREFIX_WORDS = 2


def _bash_prefix(command: str) -> str:
    words = str(command or "").split()
    return " ".join(words[:BASH_PREFIX_WORDS])


def _parent_dir(path: str) -> str:
    raw = str(path or "").strip()
    if not raw:
        return "."
    parent = PurePosixPath(raw).parent
    return str(parent) if str(parent) else "."


def _host(url: str) -> str:
    try:
        return (urlparse(str(url or "")).hostname or "").lower()
    except ValueError:
        return ""


def approval_subject(name: str, inp: dict[str, Any]) -> str:
    """The part of ``inp`` a session-wide allow generalizes over (see module doc)."""
    if name == "Bash":
        return _bash_prefix(str(inp.get("command") or ""))
    if name in _PATH_TOOLS:
        return _parent_dir(str(inp.get("path") or ""))
    if name in _URL_TOOLS:
        return _host(str(inp.get("url") or ""))
    return "*"


def make_session_rule(name: str, inp: dict[str, Any]) -> dict[str, Any]:
    from orbweaver.permissions.pipeline import bash_permissions

    rule: dict[str, Any] = {
        "tool": name,
        "subject": approval_subject(name, inp),
        "added_at": datetime.now(UTC).isoformat(),
    }
    if name == "Bash":
        rule["permissions"] = sorted(bash_permissions(inp))
    return rule


def _path_under(path: str, subject: str) -> bool:
    raw = str(path or "").strip()
    if not raw:
        return False
    if subject in {"", "."}:
        return not raw.startswith(("/", "~"))
    candidate = PurePosixPath(raw)
    base = PurePosixPath(subject)
    return candidate == base or base in candidate.parents


def rule_allows(rule: dict[str, Any], name: str, inp: dict[str, Any]) -> bool:
    if str(rule.get("tool") or "") != name:
        return False
    subject = str(rule.get("subject") or "")
    if name == "Bash":
        from orbweaver.permissions.pipeline import bash_permissions
        from orbweaver.permissions.rules import is_critical_rm

        cmd = str(inp.get("command") or "")
        if is_critical_rm(cmd):
            return False
        if _bash_prefix(cmd) != subject or not subject:
            return False
        allowed = set(rule.get("permissions") or [])
        return bash_permissions(inp) <= allowed
    if name in _PATH_TOOLS:
        return _path_under(str(inp.get("path") or ""), subject)
    if name in _URL_TOOLS:
        return bool(subject) and _host(str(inp.get("url") or "")) == subject
    return subject == "*"


def matching_session_rule(
    rules: list[dict[str, Any]] | None, name: str, inp: dict[str, Any]
) -> dict[str, Any] | None:
    for rule in rules or []:
        if isinstance(rule, dict) and rule_allows(rule, name, inp):
            return rule
    return None


def session_rules_from(jsonld: dict[str, Any] | None) -> list[dict[str, Any]]:
    raw = (jsonld or {}).get("permission_rules") or []
    return [dict(r) for r in raw if isinstance(r, dict)]


def add_session_rule(rules: list[dict[str, Any]], rule: dict[str, Any]) -> bool:
    """Append ``rule`` unless an equivalent (tool, subject, permissions) exists."""
    key = (rule.get("tool"), rule.get("subject"), tuple(rule.get("permissions") or ()))
    for existing in rules:
        if (
            existing.get("tool"),
            existing.get("subject"),
            tuple(existing.get("permissions") or ()),
        ) == key:
            return False
    rules.append(rule)
    return True
