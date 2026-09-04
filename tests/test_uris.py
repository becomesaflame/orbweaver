from orbweaver.uris import WorkspaceURIError, validate_workspace_uri, resolve_workspace_uri
import pytest
from pathlib import Path


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
