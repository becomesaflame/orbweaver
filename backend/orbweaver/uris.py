"""Workspace URI validation. Absolute host paths are forbidden in stored data."""

from __future__ import annotations

import re
from pathlib import Path

_ABS = re.compile(r"^([A-Za-z]:[\\/]|/|\\\\)")
_OK = re.compile(
    r"^(git\+https://|git\+ssh://|file:\./|file:[A-Za-z0-9._-]|workspace:)"
)
_SEGMENT = re.compile(r"^[A-Za-z0-9._-]+$")
_SKIP_DIR_NAMES = frozenset(
    {
        ".git",
        "__pycache__",
        ".venv",
        "venv",
        "node_modules",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".orbweaver",
    }
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
    if uri.startswith(("git+https://", "git+ssh://")):
        return uri
    if uri.startswith(("file:./", "workspace:")):
        return uri
    if uri.startswith("file:"):
        rest = uri[5:]
        if rest.startswith(("/", "\\")):
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


def normalize_rel(rel: str) -> str:
    rel = (rel or "").strip().replace("\\", "/").strip("/")
    if rel in {"", "."}:
        return ""
    parts = [p for p in rel.split("/") if p]
    if any(p in {"", ".", ".."} or not _SEGMENT.match(p) for p in parts):
        raise WorkspaceURIError("invalid relative workspace path")
    return "/".join(parts)


def portable_uri_for_rel(rel: str) -> str:
    rel = normalize_rel(rel)
    if not rel:
        return "workspace:default"
    if "/" not in rel:
        return f"workspace:{rel}"
    return f"file:./{rel}"


def resolve_rel(workspace_root: str | Path, rel: str) -> Path:
    root = Path(workspace_root).resolve()
    rel = normalize_rel(rel)
    path = root if not rel else (root / rel).resolve()
    if root not in path.parents and path != root:
        raise WorkspaceURIError("workspace path escapes WORKSPACE_ROOT")
    return path


def list_workspace_dirs(workspace_root: str | Path, rel: str = "") -> dict:
    root = Path(workspace_root).resolve()
    path = resolve_rel(root, rel)
    if not path.exists():
        raise FileNotFoundError(path)
    if not path.is_dir():
        raise WorkspaceURIError("not a directory")
    rel = normalize_rel(rel)
    dirs: list[dict[str, str]] = []
    try:
        entries = sorted(path.iterdir(), key=lambda p: p.name.lower())
    except OSError as e:
        raise WorkspaceURIError(f"cannot list directory: {e}") from e
    for child in entries:
        name = child.name
        if name in _SKIP_DIR_NAMES or name.startswith("."):
            continue
        try:
            if not child.is_dir():
                continue
            resolved = child.resolve()
        except OSError:
            continue
        if root not in resolved.parents and resolved != root:
            continue
        child_rel = f"{rel}/{name}" if rel else name
        dirs.append({"name": name, "rel": child_rel, "uri": portable_uri_for_rel(child_rel)})
    parent = "/".join(rel.split("/")[:-1]) if rel else None
    return {
        "rel": rel,
        "uri": portable_uri_for_rel(rel),
        "parent": parent,
        "host_path": str(path),
        "workspace_root": str(root),
        "dirs": dirs,
    }


def mkdir_workspace(workspace_root: str | Path, parent: str, name: str) -> dict:
    name = (name or "").strip()
    if not _SEGMENT.match(name) or name in {".", ".."}:
        raise WorkspaceURIError("invalid folder name")
    parent_rel = normalize_rel(parent)
    rel = f"{parent_rel}/{name}" if parent_rel else name
    path = resolve_rel(workspace_root, rel)
    try:
        path.mkdir(parents=False, exist_ok=True)
    except FileNotFoundError as e:
        raise WorkspaceURIError("parent folder does not exist") from e
    except OSError as e:
        raise WorkspaceURIError(f"cannot create folder: {e}") from e
    if not path.is_dir():
        raise WorkspaceURIError("path exists and is not a directory")
    return list_workspace_dirs(workspace_root, rel)
