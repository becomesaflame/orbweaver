"""Strip known secrets from tool output before it is stored or sent to the model."""

from __future__ import annotations

import re

PLACEHOLDER = "[redacted]"

_MIN_LIVE_VALUE_LEN = 8

# Config counters, not credentials (e.g. ORBWEAVER_PINNED_TOKEN_CAP).
_SAFE_SUFFIXES = (
    "_TOKEN_CAP",
    "_TOKEN_LIMIT",
    "_TOKEN_COUNT",
    "_TOKEN_BUDGET",
    "_TOKEN_TTL",
    "_TOKENS",
)

_SECRET_SUFFIXES = (
    "_API_KEY",
    "_SECRET",
    "_PASSWORD",
    "_PASSWD",
    "_PASSPHRASE",
    "_PRIVATE_KEY",
    "_ACCESS_KEY",
    "_ACCESS_TOKEN",
    "_BOT_TOKEN",
    "_SESSION_TOKEN",
    "_TOKEN",
)

_EXACT_SECRET_NAMES = frozenset(
    {
        "ANTHROPIC_API_KEY",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "DATABASE_URL",
        "GH_TOKEN",
        "GITHUB_PAT",
        "GITHUB_TOKEN",
        "HF_TOKEN",
        "OPENAI_API_KEY",
        "ORBWEAVER_IMAGE_API_KEY",
        "ORBWEAVER_JWT_SECRET",
        "TELEGRAM_BOT_TOKEN",
        "HINDSIGHT_API_KEY",
    }
)

_ASSIGN_LINE_RE = re.compile(
    r"(?m)^(?P<pre>[ \t]*(?:export[ \t]+)?)"
    r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)"
    r"(?P<sep>[ \t]*[=:][ \t]*)"
    r"(?P<val>.*)$"
)
_JSON_RE = re.compile(
    r'(?i)(?P<key>"(?P<name>[A-Za-z_][A-Za-z0-9_]*)")\s*:\s*'
    r'(?P<val>"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|[^\s,}\]]+)'
)
_INLINE_ASSIGN_RE = re.compile(
    r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)=(?P<val>[^\s;|&]+)"
)


def is_secret_name(name: str) -> bool:
    n = (name or "").upper()
    if not n or any(n.endswith(s) for s in _SAFE_SUFFIXES):
        return False
    if n in _EXACT_SECRET_NAMES:
        return True
    return any(n.endswith(s) for s in _SECRET_SUFFIXES)


def _live_secret_values() -> list[str]:
    from orbweaver.config import settings

    raw = [
        settings.anthropic_api_key,
        settings.orbweaver_jwt_secret,
        settings.telegram_bot_token,
        settings.orbweaver_image_api_key,
        settings.database_url,
        settings.hindsight_api_key,
    ]
    values = [v for v in raw if isinstance(v, str) and len(v) >= _MIN_LIVE_VALUE_LEN]
    values.sort(key=len, reverse=True)
    return values


def _redact_assign_line(match: re.Match[str]) -> str:
    if not is_secret_name(match.group("name")):
        return match.group(0)
    return f"{match.group('pre')}{match.group('name')}{match.group('sep')}{PLACEHOLDER}"


def _redact_json(match: re.Match[str]) -> str:
    if not is_secret_name(match.group("name")):
        return match.group(0)
    return f'{match.group("key")}: "{PLACEHOLDER}"'


def _redact_inline(match: re.Match[str]) -> str:
    if not is_secret_name(match.group("name")):
        return match.group(0)
    return f"{match.group('name')}={PLACEHOLDER}"


def redact_secrets(text: str) -> str:
    """Replace secret assignments and configured credential values with a placeholder."""
    if not text:
        return text
    out = _ASSIGN_LINE_RE.sub(_redact_assign_line, text)
    out = _JSON_RE.sub(_redact_json, out)
    out = _INLINE_ASSIGN_RE.sub(_redact_inline, out)
    for value in _live_secret_values():
        if value and value in out:
            out = out.replace(value, PLACEHOLDER)
    return out
