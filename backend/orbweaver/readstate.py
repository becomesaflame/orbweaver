"""Per-session record of which files the model has Read, and their stamp at read time.

Edit tools refuse to touch an existing file the model has not read this
session, or one that changed on disk since the read (user edit, formatter,
another session). Stamps live in memory keyed by session id; on the first
edit after a gateway restart the map is rebuilt from the session's
``tool_call`` events (best effort: the stamp is the file's current one).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import UUID

log = logging.getLogger(__name__)

READ_TOOLS = frozenset({"Read"})
EDIT_TOOLS = frozenset({"Write", "StrReplace", "NotebookEdit"})

NOT_READ = "{path} has not been read yet. Read it first before editing it."
CHANGED = "{path} changed since it was read; Read it again before editing."


def file_stamp(path: Path) -> tuple[int, int] | None:
    try:
        st = os.stat(path)
    except (OSError, TypeError, ValueError):
        return None
    return st.st_mtime_ns, st.st_size


@dataclass
class ReadState:
    stamps: dict[str, tuple[int, int]] = field(default_factory=dict)
    seeded: bool = False

    def record(self, path: Path) -> None:
        stamp = file_stamp(path)
        if stamp is None:
            self.stamps.pop(str(path), None)
        else:
            self.stamps[str(path)] = stamp

    def forget(self, path: Path) -> None:
        self.stamps.pop(str(path), None)

    def check(self, path: Path, display: str) -> str | None:
        """Error text when ``path`` exists but was not read, or changed since; else None."""
        current = file_stamp(path)
        if current is None or path.is_dir():
            return None
        seen = self.stamps.get(str(path))
        if seen is None:
            return NOT_READ.format(path=display)
        if seen != current:
            return CHANGED.format(path=display)
        return None

    def seed(self, events: list[Any] | None, resolve) -> None:
        """Rebuild from tool_call events once (survives gateway restarts)."""
        if self.seeded:
            return
        self.seeded = True
        events = list(events or [])
        completed: set[str] = set()
        for ev in events:
            if getattr(ev, "kind", None) != "tool_result":
                continue
            payload = getattr(ev, "payload", None) or {}
            content = payload.get("content")
            if isinstance(content, str) and content.startswith("error"):
                continue
            completed.add(str(payload.get("tool_use_id") or ""))
        for ev in events:
            if getattr(ev, "kind", None) != "tool_call":
                continue
            payload = getattr(ev, "payload", None) or {}
            name = payload.get("name")
            if name not in READ_TOOLS and name not in EDIT_TOOLS:
                continue
            if str(payload.get("id") or "") not in completed:
                continue
            raw = str((payload.get("input") or {}).get("path") or "")
            if not raw:
                continue
            try:
                path = resolve(raw)
            except (PermissionError, OSError, ValueError):
                log.debug("read-state seed: skipping unresolvable path %r", raw)
                continue
            if str(path) not in self.stamps:
                self.record(path)


_states: dict[UUID, ReadState] = {}


def read_state_for(session_id: UUID) -> ReadState:
    state = _states.get(session_id)
    if state is None:
        state = ReadState()
        _states[session_id] = state
    return state


def reset_read_states() -> None:
    _states.clear()
