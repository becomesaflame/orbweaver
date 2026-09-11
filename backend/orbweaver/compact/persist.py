"""Write oversized tool results to the workspace; store a head + tail preview in the event payload."""

from __future__ import annotations

from typing import Any

from orbweaver.config import settings
from orbweaver.redact import redact_secrets
from orbweaver.tooltext import (
    TRUNCATE_MAX_LINES,
    line_count,
    needs_truncation,
    truncate_head_tail,
)

SKIP_PERSIST = frozenset(
    {
        "Read",
        "WebFetch",
        "WebSearch",
        "Browser",
        "MemorySearch",
        "WorkspaceSearch",
        "MemoryGraph",
        "MemoryRemember",
        "MemoryReflect",
        "MemoryPin",
        "MemoryForget",
        "TodoWrite",
        "Delete",
        "ReadLints",
    }
)
PREVIEW_CHARS = 2000
_OPEN = "<persisted-output>"
_CLOSE = "</persisted-output>"


def preview_budget_chars() -> int:
    return max(PREVIEW_CHARS, int(settings.compact_tool_result_chars))


def persist_tool_result(
    workspace: Any | None,
    tool_use_id: str,
    name: str,
    content: str,
) -> tuple[str, str | None]:
    """Return (stored_content, relative_path_or_none). No-op when small, skipped, or no workspace.

    Oversized output (by chars or by lines) is written whole to the workspace and the stored
    result keeps the head and the tail around a marker, so a failing test summary or a
    traceback at the end of the output is still in the model's view.
    """
    text = content if isinstance(content, str) else str(content)
    text = redact_secrets(text)
    threshold = int(settings.compact_tool_result_chars)
    if workspace is None or name in SKIP_PERSIST:
        return text, None
    if not needs_truncation(text, max_chars=threshold, max_lines=TRUNCATE_MAX_LINES):
        return text, None
    budget = preview_budget_chars()
    rel = f".orbweaver/tool-results/{tool_use_id}.txt"
    try:
        workspace.write(rel, text)
    except (OSError, PermissionError, TypeError):
        return truncate_head_tail(text, max_chars=budget, max_lines=TRUNCATE_MAX_LINES), None
    head = f"{_OPEN}\nOutput too large ({len(text)} chars, {line_count(text)} lines). Saved to: {rel}\n"
    inner_budget = max(PREVIEW_CHARS // 2, budget - len(head) - len(_CLOSE) - 2)
    preview = truncate_head_tail(
        text, max_chars=inner_budget, max_lines=TRUNCATE_MAX_LINES - 3, path=rel
    )
    if not preview.endswith("\n"):
        preview += "\n"
    return f"{head}{preview}{_CLOSE}", rel
