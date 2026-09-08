from pathlib import Path

from orbweaver.workspace import LocalWorkspace


def test_local_glob_and_grep(tmp_path: Path):
    ws = LocalWorkspace("workspace:default", str(tmp_path))
    ws.write("src/a.py", "hello airbed\n")
    ws.write("README.md", "docs")
    assert "src/a.py" in ws.glob("**/*.py")
    hits = ws.grep("airbed")
    assert hits and "src/a.py" in hits[0]
    alt = ws.grep(r"airbed\|docs")
    assert any("src/a.py" in h for h in alt)
    assert any("README.md" in h for h in alt)


def test_path_escape_rejected(tmp_path: Path):
    ws = LocalWorkspace("workspace:default", str(tmp_path))
    try:
        ws.read("../secret")
        raise AssertionError("should have refused escape")
    except PermissionError:
        pass


def test_additional_readonly_root(tmp_path: Path, monkeypatch):
    from orbweaver.sandbox.policy import SandboxPolicy

    notes = tmp_path / "notes"
    notes.mkdir()
    (notes / "a.txt").write_text("hello", encoding="utf-8")
    ws_root = tmp_path / "ws"
    ws_root.mkdir()
    policy = SandboxPolicy(additional_readonly=(notes.resolve(),))
    monkeypatch.setattr("orbweaver.workspace.load_sandbox_policy", lambda _root: policy)
    ws = LocalWorkspace("workspace:default", str(ws_root))
    assert ws.read(str(notes / "a.txt")) == "hello"
    try:
        ws.write(str(notes / "a.txt"), "no")
        raise AssertionError("readonly extra root must not be writable")
    except PermissionError:
        pass


def test_host_reads_can_be_disabled(tmp_path: Path):
    ws = LocalWorkspace("workspace:default", str(tmp_path), host_reads=False)
    try:
        ws.read("/etc/passwd")
        raise AssertionError("host_reads=False must not read host files")
    except PermissionError:
        pass


def test_legacy_docker_kind_becomes_local(tmp_path: Path):
    from orbweaver.workspace import (
        apply_local_workspace_kind,
        make_workspace,
        normalize_workspace_kind,
    )

    assert normalize_workspace_kind("docker") == "local"
    jsonld = {"workspace_uri": "workspace:default", "workspace_kind": "docker"}
    assert apply_local_workspace_kind(jsonld) is True
    assert jsonld["workspace_kind"] == "local"
    ws = make_workspace("docker", "workspace:default", str(tmp_path))
    assert isinstance(ws, LocalWorkspace)
    ws.write("x.txt", "ok")
    assert ws.read("x.txt") == "ok"


def test_delete_working_set_file(tmp_path: Path):
    ws = LocalWorkspace("workspace:default", str(tmp_path))
    ws.write("src/gone.py", "x")
    assert (tmp_path / "src" / "gone.py").is_file()
    assert ws.delete("src/gone.py") == "deleted src/gone.py"
    assert not (tmp_path / "src" / "gone.py").exists()
    assert ws.delete("src/gone.py") == "not found: src/gone.py"


def test_delete_denies_env_and_escape(tmp_path: Path):
    ws = LocalWorkspace("workspace:default", str(tmp_path))
    (tmp_path / ".env").write_text("SECRET=1", encoding="utf-8")
    try:
        ws.delete(".env")
        raise AssertionError("should have refused .env")
    except PermissionError:
        pass
    assert (tmp_path / ".env").read_text(encoding="utf-8") == "SECRET=1"
    try:
        ws.delete("../escape.txt")
        raise AssertionError("should have refused escape")
    except PermissionError:
        pass


def test_write_and_read_bytes(tmp_path: Path):
    ws = LocalWorkspace("workspace:default", str(tmp_path))
    rel = ws.write_bytes("attachments/a.bin", b"\x00\x01")
    assert rel == "attachments/a.bin"
    assert ws.read_bytes(rel) == b"\x00\x01"
    try:
        ws.write_bytes("../escape.bin", b"no")
        raise AssertionError("should have refused escape")
    except PermissionError:
        pass
