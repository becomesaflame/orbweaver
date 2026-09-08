import json
import subprocess
from pathlib import Path
from uuid import uuid4

import pytest

from orbweaver.agent import TOOL_SPEC, run_tools
from orbweaver.codesearch import run_workspace_search, search_workspace
from orbweaver.compact.persist import persist_tool_result
from orbweaver.memory import remember
from orbweaver.store import reset_store_for_tests
from orbweaver.workspace import LocalWorkspace


def _ws(tmp_path: Path) -> LocalWorkspace:
    return LocalWorkspace("workspace:default", str(tmp_path))


def test_workspace_search_finds_source(tmp_path: Path):
    ws = _ws(tmp_path)
    ws.write("src/valve.py", "def control_airbed_valve():\n    return 'open'\n")
    ws.write("README.md", "unrelated docs about weather\n")
    hits = search_workspace(ws.root, "airbed valve control")
    assert hits
    assert any(h["path"] == "src/valve.py" for h in hits)


def test_workspace_search_skips_gitignore_git_binary_and_huge(tmp_path: Path):
    ws = _ws(tmp_path)
    ws.write(".gitignore", "secret.txt\nbuild/\n")
    ws.write("src/ok.py", "the airbed firmware uses co3ntrol-rs\n")
    ws.write("secret.txt", "the airbed firmware uses co3ntrol-rs\n")
    ws.write("build/out.py", "the airbed firmware uses co3ntrol-rs\n")
    git_obj = tmp_path / ".git" / "objects" / "pack"
    git_obj.parent.mkdir(parents=True)
    git_obj.write_text("the airbed firmware uses co3ntrol-rs\n", encoding="utf-8")
    (tmp_path / "blob.bin").write_bytes(b"\x00\x01airbed")
    (tmp_path / "huge.py").write_text("airbed " * 80_000, encoding="utf-8")
    hits = search_workspace(ws.root, "airbed firmware co3ntrol")
    paths = {h["path"] for h in hits}
    assert "src/ok.py" in paths
    assert "secret.txt" not in paths
    assert "build/out.py" not in paths
    assert not any(p.startswith(".git/") for p in paths)
    assert "blob.bin" not in paths
    assert "huge.py" not in paths


def test_workspace_search_respects_git_exclude_standard(tmp_path: Path):
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    ws = _ws(tmp_path)
    ws.write(".gitignore", "ignored.py\n")
    ws.write("kept.py", "def airbed_valve():\n    pass\n")
    ws.write("ignored.py", "def airbed_valve():\n    pass\n")
    hits = search_workspace(ws.root, "airbed_valve")
    paths = {h["path"] for h in hits}
    assert "kept.py" in paths
    assert "ignored.py" not in paths


@pytest.mark.asyncio
async def test_workspace_search_tool_is_distinct_from_memory_search(tmp_path: Path):
    store = reset_store_for_tests()
    ws = _ws(tmp_path)
    ws.write("src/mod.py", "class OrbweaverGateway:\n    pass\n")
    await remember(store, "the user prefers vim keybindings")
    ctx = {"workspace": ws, "store": store, "session_id": uuid4()}
    code = await run_tools("WorkspaceSearch", {"query": "OrbweaverGateway"}, ctx)
    mem = await run_tools("MemorySearch", {"query": "vim keybindings"}, ctx)
    assert "src/mod.py" in code
    assert "OrbweaverGateway" in code
    assert "WorkspaceSearch:" in code
    assert "vim" in mem
    parsed = json.loads(mem)
    assert "chunk_ids" in parsed
    assert "src/mod.py" not in parsed.get("text", "")
    names = [t["name"] for t in TOOL_SPEC]
    assert "WorkspaceSearch" in names
    assert "MemorySearch" in names
    assert names.index("WorkspaceSearch") != names.index("MemorySearch")


def test_workspace_search_requires_query(tmp_path: Path):
    ws = _ws(tmp_path)
    assert "requires a query" in run_workspace_search(ws, {})


def test_workspace_search_skips_persist(tmp_path: Path):
    ws = _ws(tmp_path)
    big = "WorkspaceSearch: q\n" + ("hit " * 5000)
    stored, rel = persist_tool_result(ws, "toolu_ws", "WorkspaceSearch", big)
    assert rel is None
    assert stored == big
