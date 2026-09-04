"""Workspace URI validation. Absolute host paths are forbidden in stored data."""

from __future__ import annotations

import re
from pathlib import Path

_ABS = re.compile(r"^([A-Za-z]:[\\/]|/|\\\\)")
_OK = re.compile(
    r"^(git\+https://|git\+ssh://|file:\./|file:[A-Za-z0-9._-]|workspace:)"
)


class WorkspaceURIError(ValueError):
    pass


def validate_workspace_uri(uri: str) -> str:
    uri = uri.strip()
    if not uri:
        raise WorkspaceURIError("workspace_uri is required")
    if uri.startswith("file:/") and not uri.startswith("file:./"):
        raise WorkspaceURIError("absolute file URIs are forbidden; use file:./relative")
    if _ABS.match(uri) or uri.startswith("file:///"):
        raise WorkspaceURIError(f"absolute host path is forbidden in stored data: {uri}")
    if uri.startswith("git+https://") or uri.startswith("git+ssh://"):
        return uri
    if uri.startswith("file:./") or uri.startswith("workspace:"):
        return uri
    if uri.startswith("file:"):
        rest = uri[5:]
        if rest.startswith("/") or rest.startswith("\\"):
            raise WorkspaceURIError("absolute file URIs are forbidden")
        return "file:./" + rest.lstrip("./")
    raise WorkspaceURIError(
        "workspace_uri must be git+https/git+ssh, file:./relative, or workspace:name"
    )


def resolve_workspace_uri(uri: str, workspace_root: str | Path) -> Path:
    uri = validate_workspace_uri(uri)
    root = Path(workspace_root).resolve()
    if uri.startswith("file:./"):
        rel = uri[len("file:./") :]
        path = (root / rel).resolve()
        if root not in path.parents and path != root:
            raise WorkspaceURIError("workspace path escapes WORKSPACE_ROOT")
        return path
    if uri.startswith("workspace:"):
        name = uri.split(":", 1)[1]
        if name == "default":
            return root
        path = (root / name).resolve()
        if root not in path.parents and path != root:
            raise WorkspaceURIError("workspace path escapes WORKSPACE_ROOT")
        return path
    # git URIs: checkout is expected at root / repo name; caller remaps
    return root
