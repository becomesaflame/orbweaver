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
