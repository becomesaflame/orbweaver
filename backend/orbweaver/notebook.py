"""Cell-aware edits for Jupyter notebooks."""

from __future__ import annotations

import json
from typing import Any

VALID_ACTIONS = frozenset({"replace", "insert", "delete"})
VALID_CELL_TYPES = frozenset({"code", "markdown", "raw"})


def _as_int(value: Any, default: int | None = None) -> int | None:
    if value is None or value == "":
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def cell_source(cell: dict[str, Any]) -> str:
    src = cell.get("source", "")
    if isinstance(src, list):
        return "".join(str(part) for part in src)
    return str(src)


def set_cell_source(cell: dict[str, Any], text: str) -> None:
    cell["source"] = text


def new_cell(cell_type: str, source: str) -> dict[str, Any]:
    cell: dict[str, Any] = {"cell_type": cell_type, "metadata": {}, "source": source}
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell


def parse_notebook(text: str) -> dict[str, Any]:
    try:
        nb = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"invalid notebook JSON: {e}") from e
    if not isinstance(nb, dict) or not isinstance(nb.get("cells"), list):
        raise TypeError("not a Jupyter notebook (missing cells)")
    return nb


def dumps_notebook(nb: dict[str, Any]) -> str:
    return json.dumps(nb, indent=1, ensure_ascii=False) + "\n"


def edit_notebook_dict(
    nb: dict[str, Any],
    *,
    cell_idx: int,
    action: str,
    source: str | None = None,
    old_string: str | None = None,
    new_string: str | None = None,
    cell_type: str | None = None,
) -> dict[str, Any]:
    cells: list[Any] = nb["cells"]
    n = len(cells)
    if action == "insert":
        if cell_idx < 0 or cell_idx > n:
            raise ValueError(f"cell_idx {cell_idx} out of range for insert (0..{n})")
        kind = (cell_type or "code").strip() or "code"
        if kind not in VALID_CELL_TYPES:
            raise ValueError(f"invalid cell_type {kind}")
        cells.insert(cell_idx, new_cell(kind, source or ""))
        return nb
    if cell_idx < 0 or cell_idx >= n:
        raise ValueError(f"cell_idx {cell_idx} out of range (0..{max(0, n - 1)})")
    if action == "delete":
        del cells[cell_idx]
        return nb
    cell = cells[cell_idx]
    if not isinstance(cell, dict):
        raise TypeError(f"cell {cell_idx} is not an object")
    current = cell_source(cell)
    if old_string:
        if old_string not in current:
            raise ValueError("old_string not found in cell")
        updated = current.replace(old_string, new_string or "", 1)
    elif new_string is not None and source is None:
        updated = new_string
    else:
        updated = source if source is not None else current
    set_cell_source(cell, updated)
    return nb


def apply_notebook_edit(workspace: Any, inp: dict[str, Any]) -> str:
    path = str(inp.get("path") or "").strip()
    if not path.lower().endswith(".ipynb"):
        return "error: NotebookEdit requires a .ipynb path"
    action = str(inp.get("action") or "replace").strip().lower() or "replace"
    if action not in VALID_ACTIONS:
        return f"error: action must be one of {sorted(VALID_ACTIONS)}"
    idx = _as_int(inp.get("cell_idx"))
    if idx is None:
        return "error: cell_idx is required"
    try:
        raw = workspace.read(path)
    except (OSError, PermissionError, UnicodeDecodeError, FileNotFoundError, IsADirectoryError) as e:
        return f"error reading {path}: {e}"
    try:
        nb = parse_notebook(raw)
        edit_notebook_dict(
            nb,
            cell_idx=idx,
            action=action,
            source=inp.get("source"),
            old_string=inp.get("old_string") or None,
            new_string=inp.get("new_string"),
            cell_type=inp.get("cell_type"),
        )
        workspace.write(path, dumps_notebook(nb))
    except (TypeError, ValueError) as e:
        return f"error: {e}"
    except (OSError, PermissionError) as e:
        return f"error writing {path}: {e}"
    n = len(nb["cells"])
    return json.dumps({"ok": True, "path": path, "action": action, "cell_idx": idx, "cells": n})
