import json
from uuid import uuid4

import pytest

from orbweaver.agent import TOOL_SPEC, run_tools
from orbweaver.notebook import (
    apply_notebook_edit,
    dumps_notebook,
    edit_notebook_dict,
    parse_notebook,
)
from orbweaver.store import reset_store_for_tests
from orbweaver.workspace import LocalWorkspace


def _nb(*sources: str, types: list[str] | None = None) -> dict:
    cells = []
    for i, src in enumerate(sources):
        kind = types[i] if types else "code"
        cell = {"cell_type": kind, "metadata": {}, "source": src}
        if kind == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        cells.append(cell)
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"}},
        "cells": cells,
    }


def _write_nb(ws: LocalWorkspace, path: str, nb: dict) -> None:
    ws.write(path, dumps_notebook(nb))


def _ctx(ws):
    return {"workspace": ws, "store": reset_store_for_tests(), "session_id": uuid4()}


def test_notebook_edit_in_tool_spec():
    assert any(t["name"] == "NotebookEdit" for t in TOOL_SPEC)


def test_replace_one_cell_preserves_others(tmp_path):
    ws = LocalWorkspace("workspace:default", str(tmp_path))
    _write_nb(ws, "n.ipynb", _nb("print(1)", "print(2)", "# title"))
    out = apply_notebook_edit(
        ws, {"path": "n.ipynb", "cell_idx": 1, "action": "replace", "source": "print(99)"}
    )
    body = json.loads(out)
    assert body["ok"] is True
    assert body["cells"] == 3
    nb = parse_notebook(ws.read("n.ipynb"))
    assert nb["cells"][0]["source"] == "print(1)"
    assert nb["cells"][1]["source"] == "print(99)"
    assert nb["cells"][2]["source"] == "# title"


def test_old_string_replace_in_cell():
    nb = _nb("x = 1\nprint(x)")
    edit_notebook_dict(nb, cell_idx=0, action="replace", old_string="x = 1", new_string="x = 2")
    assert "x = 2" in "".join(nb["cells"][0]["source"])
    with pytest.raises(ValueError, match="old_string"):
        edit_notebook_dict(nb, cell_idx=0, action="replace", old_string="missing", new_string="y")


def test_insert_and_delete_cells():
    nb = _nb("a", "c")
    edit_notebook_dict(nb, cell_idx=1, action="insert", source="b", cell_type="markdown")
    assert [c["source"] for c in nb["cells"]] == ["a", "b", "c"]
    assert nb["cells"][1]["cell_type"] == "markdown"
    edit_notebook_dict(nb, cell_idx=0, action="delete")
    assert [c["source"] for c in nb["cells"]] == ["b", "c"]


def test_rejects_non_notebook_and_bad_index(tmp_path):
    ws = LocalWorkspace("workspace:default", str(tmp_path))
    ws.write("notes.txt", "hello")
    assert "requires a .ipynb" in apply_notebook_edit(ws, {"path": "notes.txt", "cell_idx": 0})
    _write_nb(ws, "n.ipynb", _nb("only"))
    assert "out of range" in apply_notebook_edit(ws, {"path": "n.ipynb", "cell_idx": 4, "source": "x"})


@pytest.mark.asyncio
async def test_notebook_edit_tool(tmp_path):
    ws = LocalWorkspace("workspace:default", str(tmp_path))
    _write_nb(ws, "lab.ipynb", _nb("print('hi')"))
    result = await run_tools(
        "NotebookEdit",
        {"path": "lab.ipynb", "cell_idx": 0, "old_string": "hi", "new_string": "bye"},
        _ctx(ws),
    )
    assert json.loads(result)["ok"] is True
    assert "bye" in ws.read("lab.ipynb")
    assert "print('hi')" not in ws.read("lab.ipynb")
