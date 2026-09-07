from pathlib import Path

from orbweaver.workspace import DockerWorkspace, LocalWorkspace


def test_local_glob_and_grep(tmp_path: Path):
    ws = LocalWorkspace("workspace:default", str(tmp_path))
    ws.write("src/a.py", "hello airbed\n")
    ws.write("README.md", "docs")
    assert "src/a.py" in ws.glob("**/*.py")
    hits = ws.grep("airbed")
    assert hits and "src/a.py" in hits[0]


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


def test_docker_rejects_host_read(tmp_path: Path):
    ws = DockerWorkspace("workspace:default", str(tmp_path))
    try:
        ws.read("/etc/passwd")
        raise AssertionError("docker workspace must not read host files")
    except PermissionError:
        pass


def test_docker_bash_without_binary(tmp_path: Path, monkeypatch):
    def boom(*_a, **_k):
        raise FileNotFoundError("docker")

    monkeypatch.setattr("orbweaver.workspace.subprocess.run", boom)
    ws = DockerWorkspace("workspace:default", str(tmp_path))
    ws.write("x.txt", "ok")
    assert ws.read("x.txt") == "ok"
    out = ws.bash("echo hi")
    assert "docker is not available" in out


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
