"""Re-inject recently read files after a compact boundary so the next turn can continue."""

from __future__ import annotations

from typing import Any

from orbweaver.compact.project import last_boundary
from orbweaver.config import settings
from orbweaver.image import is_image_path
from orbweaver.store import Event
from orbweaver.tokens import estimate_tokens


def recent_read_paths(events: list[Event], keep_from_seq: int, limit: int) -> list[str]:
    paths: list[str] = []
    seen: set[str] = set()
    for ev in reversed(events):
        if ev.seq >= keep_from_seq:
            continue
        if ev.kind != "tool_call":
            continue
        if str(ev.payload.get("name") or "") != "Read":
            continue
        inp = ev.payload.get("input") or {}
        path = str(inp.get("path") or "").strip()
        if not path or path in seen:
            continue
        seen.add(path)
        paths.append(path)
        if len(paths) >= limit:
            break
    paths.reverse()
    return paths


def paths_in_live_reads(live: list[Event]) -> set[str]:
    present: set[str] = set()
    id_to_path: dict[str, str] = {}
    for ev in live:
        if ev.kind == "tool_call" and str(ev.payload.get("name") or "") == "Read":
            path = str((ev.payload.get("input") or {}).get("path") or "")
            tid = str(ev.payload.get("id") or ev.id)
            if path:
                id_to_path[tid] = path
        elif ev.kind == "tool_result" and str(ev.payload.get("name") or "") == "Read":
            tid = str(ev.payload.get("tool_use_id") or "")
            found = id_to_path.get(tid)
            if found:
                present.add(found)
    return present


def rehydrate_messages(
    messages: list[dict[str, Any]],
    events: list[Event],
    workspace: Any | None,
) -> list[dict[str, Any]]:
    if workspace is None or not events:
        return messages
    boundary = last_boundary(events)
    if boundary is None:
        return messages
    keep_from = int(boundary.payload.get("keep_from_seq") or (boundary.seq + 1))
    live = [e for e in events if e.seq >= keep_from and e.id != boundary.id]
    already = paths_in_live_reads(live)
    n_files = max(0, int(settings.compact_rehydrate_files))
    per_file = max(500, int(settings.compact_rehydrate_chars_per_file))
    budget = max(per_file, int(settings.compact_rehydrate_token_budget))
    used = 0
    blocks: list[str] = []
    for path in recent_read_paths(events, keep_from, n_files):
        if path in already:
            continue
        if is_image_path(path):
            continue
        try:
            text = workspace.read(path)
        except (OSError, PermissionError, UnicodeDecodeError, IsADirectoryError):
            continue
        snippet = text[:per_file]
        tokens = estimate_tokens(snippet)
        if used + tokens > budget:
            break
        used += tokens
        blocks.append(f"### {path}\n{snippet}")
    if not blocks:
        return messages
    blob = "[recently read files]\n" + "\n\n".join(blocks)
    extra = {"role": "user", "content": blob}
    if messages and messages[0].get("role") == "user":
        return [messages[0], extra, *messages[1:]]
    return [extra, *messages]
