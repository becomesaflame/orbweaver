"""Detect context-window API failures and parse a usable compact threshold."""

from __future__ import annotations

import json
import re

CONTEXT_FULL_MESSAGE = (
    "Context is full after compact. The prompt still exceeds the model window. "
    "Start a new session or continue with a narrower task."
)
OVERFLOW_KEEP_ROUNDS = (4, 2, 1, 0)
WINDOW_COMPACT_RATIO = 0.70

_OVERFLOW_TYPES = frozenset(
    {"prompt_too_long", "context_length_exceeded", "request_too_large"}
)
_OVERFLOW_NEEDLES = (
    "prompt_too_long",
    "prompt is too long",
    "maximum context length",
    "context length exceeded",
    "exceeds the available context",
    "exceed context",
    "exceeds the context",
    "n_prompt + n_predict",
    "too many tokens",
)


def _error_type(exc: BaseException) -> str:
    err_type = str(getattr(exc, "type", "") or "")
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        inner = body.get("error")
        if isinstance(inner, dict):
            err_type = str(inner.get("type") or err_type)
        elif not err_type:
            err_type = str(body.get("type") or "")
    return err_type.lower()


def _error_text(exc: BaseException) -> str:
    parts = [str(exc)]
    message = getattr(exc, "message", None)
    if message:
        parts.append(str(message))
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        parts.append(json.dumps(body))
    return "\n".join(parts)


def is_context_overflow(exc: BaseException) -> bool:
    """True for prompt_too_long, HTTP 413, or known context-window error text."""
    name = type(exc).__name__
    if name in {"RequestTooLargeError"}:
        return True
    status = getattr(exc, "status_code", None)
    if status == 413:
        return True
    if _error_type(exc) in _OVERFLOW_TYPES:
        return True
    text = _error_text(exc).lower()
    return any(needle in text for needle in _OVERFLOW_NEEDLES)


def extract_context_window_tokens(exc: BaseException) -> int | None:
    """Best-effort window size from an overflow error (the 'maximum' side)."""
    text = _error_text(exc)
    tokens_gt = re.search(
        r"(\d+)\s*(?:tokens?)?\s*>\s*(\d+)\s*(?:maximum|max|tokens?)?",
        text,
        re.IGNORECASE,
    )
    if tokens_gt:
        return int(tokens_gt.group(2))
    patterns = (
        r"maximum context length(?: is| of)?\s*(\d+)",
        r"context (?:window|length|size)(?: is| of)?\s*(\d+)",
        r"available context size(?: of)?\s*(\d+)",
        r"n_ctx[=:\s]+(\d+)",
    )
    for pat in patterns:
        match = re.search(pat, text, re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None


def overflow_compact_budget(exc: BaseException) -> int | None:
    window = extract_context_window_tokens(exc)
    if window is None:
        return None
    return max(256, int(window * WINDOW_COMPACT_RATIO))


class ContextFullError(Exception):
    """Prompt still exceeds the window after compact retries."""

    def __init__(self, message: str = CONTEXT_FULL_MESSAGE) -> None:
        super().__init__(message)
        self.message = message
