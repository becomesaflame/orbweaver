"""Write oversized tool results to the workspace; store a preview in the event payload."""

from __future__ import annotations

from typing import Any

from orbweaver.config import settings

SKIP_PERSIST = frozenset({"Read", "MemorySearch", "MemoryRemember", "MemoryPin", "MemoryForget"})
PREVIEW_CHARS = 2000


def persist_tool_result(
    workspace: Any | None,
    tool_use_id: str,
    name: str,
    content: str,
) -> tuple[str, str | None]:
    """Return (stored_content, relative_path_or_none). No-op when small, skipped, or no workspace."""
    text = content if isinstance(content, str) else str(content)
    threshold = int(settings.compact_tool_result_chars)
    if workspace is None or name in SKIP_PERSIST or len(text) <= threshold:
        return text, None
    rel = f".orbweaver/tool-results/{tool_use_id}.txt"
    try:
        workspace.write(rel, text)
    except (OSError, PermissionError, TypeError):
        return text[: max(threshold, PREVIEW_CHARS)], None
    preview = text[:PREVIEW_CHARS]
    stored = (
        f"<persisted-output>\n"
        f"Output too large ({len(text)} chars). Saved to: {rel}\n"
        f"Preview:\n{preview}\n...\n"
        f"</persisted-output>"
    )
    return stored, rel
