from uuid import uuid4

import pytest

from orbweaver.agent import run_tools, static_system, tools_for_channel
from orbweaver.lints import EDIT_TOOLS, recent_edited_paths
from orbweaver.permissions import can_use_tool
from orbweaver.store import Event, reset_store_for_tests
from orbweaver.workspace import LocalWorkspace


def _ctx(tmp_path, events=None):
    return {
        "workspace": LocalWorkspace("workspace:default", str(tmp_path)),
        "store": reset_store_for_tests(),
        "session_id": uuid4(),
        "workspace_kind": "local",
        "events": events or [],
    }


@pytest.mark.asyncio
async def test_strreplace_unique_match_writes(tmp_path):
    ws = LocalWorkspace("workspace:default", str(tmp_path))
    ws.write("src/a.py", "alpha\nkeep\n")
    result = await run_tools(
        "StrReplace",
        {"path": "src/a.py", "old_string": "alpha", "new_string": "beta"},
        _ctx(tmp_path),
    )
    assert result.startswith("updated")
    assert (tmp_path / "src" / "a.py").read_text(encoding="utf-8") == "beta\nkeep\n"


@pytest.mark.asyncio
async def test_strreplace_duplicate_match_leaves_file(tmp_path):
    ws = LocalWorkspace("workspace:default", str(tmp_path))
    original = "x = 1\nx = 2\n"
    ws.write("src/a.py", original)
    result = await run_tools(
        "StrReplace",
        {"path": "src/a.py", "old_string": "x = ", "new_string": "y = "},
        _ctx(tmp_path),
    )
    assert "matched 2 times" in result
    assert "replace_all" in result
    assert (tmp_path / "src" / "a.py").read_text(encoding="utf-8") == original


@pytest.mark.asyncio
async def test_strreplace_replace_all(tmp_path):
    ws = LocalWorkspace("workspace:default", str(tmp_path))
    ws.write("src/a.py", "x = 1\nx = 2\n")
    result = await run_tools(
        "StrReplace",
        {
            "path": "src/a.py",
            "old_string": "x = ",
            "new_string": "y = ",
            "replace_all": True,
        },
        _ctx(tmp_path),
    )
    assert "2 occurrences" in result
    assert (tmp_path / "src" / "a.py").read_text(encoding="utf-8") == "y = 1\ny = 2\n"


@pytest.mark.asyncio
async def test_strreplace_conflict_hunk(tmp_path):
    ws = LocalWorkspace("workspace:default", str(tmp_path))
    conflicted = (
        "keep\n"
        "<<<<<<< HEAD\n"
        "from_main\n"
        "=======\n"
        "from_feature\n"
        ">>>>>>> feat\n"
        "tail\n"
    )
    hunk = (
        "<<<<<<< HEAD\n"
        "from_main\n"
        "=======\n"
        "from_feature\n"
        ">>>>>>> feat\n"
    )
    ws.write("src/a.py", conflicted)
    result = await run_tools(
        "StrReplace",
        {"path": "src/a.py", "old_string": hunk, "new_string": "from_main\n"},
        _ctx(tmp_path),
    )
    assert result.startswith("updated")
    assert (tmp_path / "src" / "a.py").read_text(encoding="utf-8") == "keep\nfrom_main\ntail\n"


@pytest.mark.asyncio
async def test_strreplace_missing_file(tmp_path):
    result = await run_tools(
        "StrReplace",
        {"path": "src/missing.py", "old_string": "a", "new_string": "b"},
        _ctx(tmp_path),
    )
    assert "not found" in result


@pytest.mark.asyncio
async def test_strreplace_empty_old_string(tmp_path):
    ws = LocalWorkspace("workspace:default", str(tmp_path))
    ws.write("src/a.py", "hello\n")
    result = await run_tools(
        "StrReplace",
        {"path": "src/a.py", "old_string": "", "new_string": "x"},
        _ctx(tmp_path),
    )
    assert "old_string is required" in result
    assert (tmp_path / "src" / "a.py").read_text(encoding="utf-8") == "hello\n"


@pytest.mark.asyncio
async def test_strreplace_blocks_env(tmp_path):
    (tmp_path / ".env").write_text("SECRET=1\n", encoding="utf-8")
    result = await run_tools(
        "StrReplace",
        {"path": ".env", "old_string": "SECRET=1", "new_string": "SECRET=2"},
        _ctx(tmp_path),
    )
    assert "error replacing" in result
    assert (tmp_path / ".env").read_text(encoding="utf-8") == "SECRET=1\n"


@pytest.mark.asyncio
async def test_in_project_strreplace_skips_classifier(tmp_path, monkeypatch):
    async def boom(*_a, **_k):
        raise AssertionError("classifier should not run")

    monkeypatch.setattr("orbweaver.permissions.pipeline.classify_action", boom)
    decision = await can_use_tool(
        "StrReplace",
        {"path": "src/a.py", "old_string": "a", "new_string": "b"},
        _ctx(tmp_path),
    )
    assert decision.behavior == "allow"
    assert decision.fast_path == "acceptEdits"


@pytest.mark.asyncio
async def test_env_strreplace_denied(tmp_path, monkeypatch):
    async def boom(*_a, **_k):
        raise AssertionError("classifier should not run")

    monkeypatch.setattr("orbweaver.permissions.pipeline.classify_action", boom)
    decision = await can_use_tool(
        "StrReplace",
        {"path": ".env", "old_string": "a", "new_string": "b"},
        _ctx(tmp_path),
    )
    assert decision.behavior == "deny"


def test_strreplace_on_all_channels():
    for channel in ("telegram", "web", "cron", "", "vscode"):
        names = {t["name"] for t in tools_for_channel(channel)}
        assert "StrReplace" in names
    assert "StrReplace" in static_system()
    assert "StrReplace" in static_system("telegram")


def test_readlints_counts_strreplace_as_edit():
    sid = uuid4()
    events = [
        Event(
            id=uuid4(),
            session_id=sid,
            seq=1,
            kind="tool_call",
            payload={"name": "StrReplace", "input": {"path": "src/edited.py"}},
        )
    ]
    assert "StrReplace" in EDIT_TOOLS
    assert recent_edited_paths(events) == ["src/edited.py"]
