import os
from uuid import uuid4

import pytest

from orbweaver.agent import run_tools, static_system, tools_for_channel
from orbweaver.config import settings
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


async def _read(ctx, path):
    result = await run_tools("Read", {"path": path}, ctx)
    assert not result.startswith("error"), result
    return result


async def _replace(ctx, path, old, new, **extra):
    return await run_tools(
        "StrReplace",
        {"path": path, "old_string": old, "new_string": new, **extra},
        ctx,
    )


@pytest.mark.asyncio
async def test_strreplace_unique_match_writes(tmp_path):
    ws = LocalWorkspace("workspace:default", str(tmp_path))
    ws.write("src/a.py", "alpha\nkeep\n")
    ctx = _ctx(tmp_path)
    await _read(ctx, "src/a.py")
    result = await _replace(ctx, "src/a.py", "alpha", "beta")
    assert result.startswith("updated src/a.py (1 occurrence)")
    assert "via" not in result.splitlines()[0]
    assert (tmp_path / "src" / "a.py").read_text(encoding="utf-8") == "beta\nkeep\n"


@pytest.mark.asyncio
async def test_strreplace_result_contains_unified_diff(tmp_path):
    ws = LocalWorkspace("workspace:default", str(tmp_path))
    ws.write("src/a.py", "alpha\nkeep\n")
    ctx = _ctx(tmp_path)
    await _read(ctx, "src/a.py")
    result = await _replace(ctx, "src/a.py", "alpha", "beta")
    assert "--- a/src/a.py" in result
    assert "+++ b/src/a.py" in result
    assert "\n-alpha\n" in result
    assert "\n+beta\n" in result
    assert "\n keep" in result


@pytest.mark.asyncio
async def test_strreplace_duplicate_match_leaves_file(tmp_path):
    ws = LocalWorkspace("workspace:default", str(tmp_path))
    original = "x = 1\nx = 2\n"
    ws.write("src/a.py", original)
    ctx = _ctx(tmp_path)
    await _read(ctx, "src/a.py")
    result = await _replace(ctx, "src/a.py", "x = ", "y = ")
    assert "matched 2 times" in result
    assert "replace_all" in result
    assert (tmp_path / "src" / "a.py").read_text(encoding="utf-8") == original


@pytest.mark.asyncio
async def test_strreplace_replace_all(tmp_path):
    ws = LocalWorkspace("workspace:default", str(tmp_path))
    ws.write("src/a.py", "x = 1\nx = 2\n")
    ctx = _ctx(tmp_path)
    await _read(ctx, "src/a.py")
    result = await _replace(ctx, "src/a.py", "x = ", "y = ", replace_all=True)
    assert "2 occurrences" in result
    assert (tmp_path / "src" / "a.py").read_text(encoding="utf-8") == "y = 1\ny = 2\n"


@pytest.mark.asyncio
async def test_strreplace_not_found_lists_nearest_lines(tmp_path):
    ws = LocalWorkspace("workspace:default", str(tmp_path))
    ws.write("src/a.py", "def compute_total(items):\n    return sum(items)\n")
    ctx = _ctx(tmp_path)
    await _read(ctx, "src/a.py")
    result = await _replace(ctx, "src/a.py", "def compute_totals(item):", "def total(items):")
    assert result.startswith("error: old_string not found")
    assert "Nearest lines" in result
    assert "1|def compute_total(items):" in result


@pytest.mark.asyncio
async def test_strreplace_tab_vs_space_indent_matches_trimmed_tier(tmp_path):
    ws = LocalWorkspace("workspace:default", str(tmp_path))
    ws.write("src/a.py", "def f():\n\tif x:\n\t\treturn 1\n\treturn 0\n")
    ctx = _ctx(tmp_path)
    await _read(ctx, "src/a.py")
    result = await _replace(
        ctx,
        "src/a.py",
        "    if x:\n        return 1\n",
        "    if x and y:\n        return 2\n",
    )
    assert result.startswith("updated src/a.py (1 occurrence via line-trimmed match")
    assert (tmp_path / "src" / "a.py").read_text(encoding="utf-8") == (
        "def f():\n\tif x and y:\n\t\treturn 2\n\treturn 0\n"
    )


@pytest.mark.asyncio
async def test_strreplace_trailing_whitespace_matches_trimmed_tier(tmp_path):
    ws = LocalWorkspace("workspace:default", str(tmp_path))
    ws.write("src/a.py", "a = 1   \nb = 2\n")
    ctx = _ctx(tmp_path)
    await _read(ctx, "src/a.py")
    result = await _replace(ctx, "src/a.py", "a = 1\nb = 2\n", "a = 10\nb = 20\n")
    assert "via line-trimmed match" in result
    assert (tmp_path / "src" / "a.py").read_text(encoding="utf-8") == "a = 10\nb = 20\n"


@pytest.mark.asyncio
async def test_strreplace_line_number_prefix_stripped(tmp_path):
    ws = LocalWorkspace("workspace:default", str(tmp_path))
    ws.write("src/a.py", "import os\n\ndef main():\n    pass\n")
    ctx = _ctx(tmp_path)
    await _read(ctx, "src/a.py")
    result = await _replace(
        ctx,
        "src/a.py",
        "3|def main():\n4|    pass",
        "def main():\n    return 1",
    )
    assert "via line-number prefixes stripped" in result
    assert (tmp_path / "src" / "a.py").read_text(encoding="utf-8") == (
        "import os\n\ndef main():\n    return 1\n"
    )


@pytest.mark.asyncio
async def test_strreplace_anchor_tier_matches_changed_middle(tmp_path):
    ws = LocalWorkspace("workspace:default", str(tmp_path))
    ws.write("src/a.py", "head\nstart\nmiddle actual\nother actual\nend\ntail\n")
    ctx = _ctx(tmp_path)
    await _read(ctx, "src/a.py")
    result = await _replace(
        ctx,
        "src/a.py",
        "start\nmiddle remembered\nother remembered\nend\n",
        "replaced\n",
    )
    assert "via first/last line anchor match" in result
    assert (tmp_path / "src" / "a.py").read_text(encoding="utf-8") == "head\nreplaced\ntail\n"


@pytest.mark.asyncio
async def test_strreplace_fallback_ambiguity_still_errors(tmp_path):
    ws = LocalWorkspace("workspace:default", str(tmp_path))
    original = "def a():\n\treturn 1\n\ndef b():\n\treturn 1\n"
    ws.write("src/a.py", original)
    ctx = _ctx(tmp_path)
    await _read(ctx, "src/a.py")
    result = await _replace(ctx, "src/a.py", "    return 1", "    return 2")
    assert result.startswith("error: old_string matched 2 times")
    assert "line-trimmed" in result
    assert (tmp_path / "src" / "a.py").read_text(encoding="utf-8") == original


@pytest.mark.asyncio
async def test_strreplace_requires_prior_read(tmp_path):
    ws = LocalWorkspace("workspace:default", str(tmp_path))
    ws.write("src/a.py", "alpha\n")
    result = await _replace(_ctx(tmp_path), "src/a.py", "alpha", "beta")
    assert result == "error: src/a.py has not been read yet. Read it first before editing it."
    assert (tmp_path / "src" / "a.py").read_text(encoding="utf-8") == "alpha\n"


@pytest.mark.asyncio
async def test_strreplace_refused_after_external_change(tmp_path):
    ws = LocalWorkspace("workspace:default", str(tmp_path))
    ws.write("src/a.py", "alpha\n")
    ctx = _ctx(tmp_path)
    await _read(ctx, "src/a.py")
    target = tmp_path / "src" / "a.py"
    target.write_text("alpha\nformatter added a line\n", encoding="utf-8")
    os.utime(target, ns=(target.stat().st_atime_ns, target.stat().st_mtime_ns + 5_000_000))
    result = await _replace(ctx, "src/a.py", "alpha", "beta")
    assert result == "error: src/a.py changed since it was read; Read it again before editing."
    assert target.read_text(encoding="utf-8") == "alpha\nformatter added a line\n"
    await _read(ctx, "src/a.py")
    result = await _replace(ctx, "src/a.py", "alpha", "beta")
    assert result.startswith("updated")


@pytest.mark.asyncio
async def test_strreplace_consecutive_edits_need_one_read(tmp_path):
    ws = LocalWorkspace("workspace:default", str(tmp_path))
    ws.write("src/a.py", "one\ntwo\n")
    ctx = _ctx(tmp_path)
    await _read(ctx, "src/a.py")
    assert (await _replace(ctx, "src/a.py", "one", "1")).startswith("updated")
    assert (await _replace(ctx, "src/a.py", "two", "2")).startswith("updated")
    assert (tmp_path / "src" / "a.py").read_text(encoding="utf-8") == "1\n2\n"


@pytest.mark.asyncio
async def test_edit_require_read_setting_disables_guard(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "edit_require_read", False)
    ws = LocalWorkspace("workspace:default", str(tmp_path))
    ws.write("src/a.py", "alpha\n")
    result = await _replace(_ctx(tmp_path), "src/a.py", "alpha", "beta")
    assert result.startswith("updated")


@pytest.mark.asyncio
async def test_read_state_rebuilt_from_session_events(tmp_path):
    ws = LocalWorkspace("workspace:default", str(tmp_path))
    ws.write("src/a.py", "alpha\n")
    sid = uuid4()
    events = [
        Event(
            id=uuid4(),
            session_id=sid,
            seq=1,
            kind="tool_call",
            payload={"id": "tu_1", "name": "Read", "input": {"path": "src/a.py"}},
        ),
        Event(
            id=uuid4(),
            session_id=sid,
            seq=2,
            kind="tool_result",
            payload={"tool_use_id": "tu_1", "name": "Read", "content": "src/a.py: lines 1-1 of 1"},
        ),
        Event(
            id=uuid4(),
            session_id=sid,
            seq=3,
            kind="tool_call",
            payload={"id": "tu_2", "name": "StrReplace", "input": {"path": "src/a.py"}},
        ),
    ]
    ctx = _ctx(tmp_path, events=events)
    ctx["session_id"] = sid
    result = await _replace(ctx, "src/a.py", "alpha", "beta")
    assert result.startswith("updated"), result


@pytest.mark.asyncio
async def test_read_state_seed_ignores_pending_edit_call(tmp_path):
    """The in-flight edit's own tool_call event must not count as a Read."""
    ws = LocalWorkspace("workspace:default", str(tmp_path))
    ws.write("src/a.py", "alpha\n")
    sid = uuid4()
    events = [
        Event(
            id=uuid4(),
            session_id=sid,
            seq=1,
            kind="tool_call",
            payload={"id": "tu_2", "name": "Write", "input": {"path": "src/a.py"}},
        ),
    ]
    ctx = _ctx(tmp_path, events=events)
    ctx["session_id"] = sid
    result = await _replace(ctx, "src/a.py", "alpha", "beta")
    assert "has not been read yet" in result


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
    ctx = _ctx(tmp_path)
    await _read(ctx, "src/a.py")
    result = await _replace(ctx, "src/a.py", hunk, "from_main\n")
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
    ctx = _ctx(tmp_path)
    await _read(ctx, "src/a.py")
    result = await _replace(ctx, "src/a.py", "", "x")
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
