from pathlib import Path

import pytest

from orbweaver.uris import (
    WorkspaceURIError,
    list_workspace_dirs,
    mkdir_workspace,
    portable_uri_for_rel,
    resolve_workspace_uri,
    validate_workspace_uri,
)


def test_rejects_absolute_paths():
    with pytest.raises(WorkspaceURIError):
        validate_workspace_uri("/Users/me/proj")
    with pytest.raises(WorkspaceURIError):
        validate_workspace_uri("file:///Users/me/proj")
    with pytest.raises(WorkspaceURIError):
        validate_workspace_uri("C:\\Users\\me\\proj")


def test_accepts_portable_forms(tmp_path: Path):
    assert validate_workspace_uri("file:./proj").startswith("file:./")
    assert validate_workspace_uri("workspace:default") == "workspace:default"
    assert validate_workspace_uri("git+https://github.com/org/repo.git#refs/heads/main").startswith("git+")
    p = resolve_workspace_uri("file:./.", tmp_path)
    assert p == tmp_path.resolve()



def test_portable_uri_for_rel():
    assert portable_uri_for_rel("") == "workspace:default"
    assert portable_uri_for_rel("orbweaver") == "workspace:orbweaver"
    assert portable_uri_for_rel("a/b") == "file:./a/b"


def test_list_and_mkdir_workspace(tmp_path: Path):
    (tmp_path / "orbweaver").mkdir()
    (tmp_path / "skipme.txt").write_text("x", encoding="utf-8")
    (tmp_path / ".git").mkdir()
    listing = list_workspace_dirs(tmp_path, "")
    names = [d["name"] for d in listing["dirs"]]
    assert names == ["orbweaver"]
    assert listing["uri"] == "workspace:default"
    nested = mkdir_workspace(tmp_path, "orbweaver", "scratch")
    assert nested["uri"] == "file:./orbweaver/scratch"
    assert (tmp_path / "orbweaver" / "scratch").is_dir()


def test_list_rejects_escape(tmp_path: Path):
    with pytest.raises(WorkspaceURIError):
        list_workspace_dirs(tmp_path, "../etc")
    with pytest.raises(WorkspaceURIError):
        mkdir_workspace(tmp_path, "", "..")
