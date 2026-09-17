"""Workspace path, git branch, and prompt-size snapshot for the web LCD."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import UUID

from orbweaver.compact.usage import estimate_prompt_tokens
from orbweaver.config import settings
from orbweaver.git_ritual import current_branch
from orbweaver.model_routing import resolve_turn_model
from orbweaver.open_models import context_window_for
from orbweaver.store import Event
from orbweaver.uris import WorkspaceURIError, resolve_workspace_uri


def workspace_label(
    uri: str,
    workspace_root: str | Path,
    *,
    path: Path | None = None,
) -> str:
    """Human path for the LCD, preferring ``~/…`` when the checkout is under home."""
    try:
        resolved = path or resolve_workspace_uri(uri or "workspace:default", workspace_root)
    except (WorkspaceURIError, OSError, ValueError):
        raw = str(uri or "").strip()
        if raw.startswith("workspace:"):
            return raw.split(":", 1)[1] or "default"
        return raw
    try:
        rel = resolved.resolve().relative_to(Path.home().resolve())
        return "~/" + str(rel).replace("\\", "/")
    except ValueError:
        return str(resolved)


def context_snapshot(
    session_id: UUID,
    events: list[Event],
    *,
    session: dict[str, Any] | None = None,
    workspace: Any | None = None,
    model: str | None = None,
    channel: str | None = None,
) -> dict[str, Any]:
    """Tokens / window / workspace / branch for GET events and live ``context`` frames."""
    jsonld = session or {}
    uri = str(getattr(workspace, "uri", None) or jsonld.get("workspace_uri") or "workspace:default")
    root = getattr(workspace, "root", None)
    root_path = root if isinstance(root, Path) else _resolved_root(uri, settings.workspace_root)
    turn_model = resolve_turn_model(
        override=model, session=jsonld, channel=channel or str(jsonld.get("channel") or "")
    )
    window = context_window_for(turn_model, settings.context_window)
    tokens = estimate_prompt_tokens(session_id, events)
    return {
        "tokens": int(tokens),
        "window": int(window),
        "workspace": workspace_label(uri, settings.workspace_root, path=root_path),
        "branch": current_branch(root_path),
    }


def context_frame(
    session_id: UUID,
    events: list[Event],
    **kwargs: Any,
) -> dict[str, Any]:
    return {"kind": "context", **context_snapshot(session_id, events, **kwargs)}


def _resolved_root(uri: str, workspace_root: str | Path) -> Path | None:
    try:
        return resolve_workspace_uri(uri or "workspace:default", workspace_root)
    except (WorkspaceURIError, OSError, ValueError):
        return None
