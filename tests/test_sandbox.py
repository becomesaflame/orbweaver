from pathlib import Path

from orbweaver.config import settings
from orbweaver.sandbox.bwrap import build_bwrap_argv, sandbox_available
from orbweaver.workspace import DockerWorkspace, LocalWorkspace


def test_bwrap_argv_has_isolation(tmp_path: Path):
    argv = build_bwrap_argv("echo hi", tmp_path, tmp_path / "tmp")
    assert "--unshare-net" in argv
    assert "--unshare-user" in argv
    assert "--unshare-pid" in argv
    assert "--die-with-parent" in argv
    joined = " ".join(argv)
    assert str(tmp_path.resolve()) in joined
    assert argv[-3:] == ["bash", "-lc", "echo hi"]


def test_local_bash_fail_closed_without_bwrap(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(settings, "orbweaver_sandbox", True)
    monkeypatch.setattr(settings, "orbweaver_sandbox_fail_if_unavailable", True)
    monkeypatch.setattr("orbweaver.sandbox.bwrap.is_containerized", lambda: False)
    monkeypatch.setattr("orbweaver.sandbox.bwrap.bwrap_path", lambda: None)

    def boom(*_a, **_k):
        raise AssertionError("must not run unsandboxed")

    monkeypatch.setattr("orbweaver.workspace.LocalWorkspace._raw_bash", boom)
    ws = LocalWorkspace("workspace:default", str(tmp_path))
    out = ws.bash("echo hi")
    assert "sandbox_unavailable" in out


def test_docker_bash_without_binary(tmp_path: Path, monkeypatch):
    def boom(*_a, **_k):
        raise FileNotFoundError("docker")

    monkeypatch.setattr("orbweaver.workspace.subprocess.run", boom)
    ws = DockerWorkspace("workspace:default", str(tmp_path))
    ws.write("x.txt", "ok")
    assert ws.read("x.txt") == "ok"
    out = ws.bash("echo hi")
    assert "docker is not available" in out


def test_sandbox_available_in_container(monkeypatch):
    monkeypatch.setattr(settings, "orbweaver_sandbox", True)
    monkeypatch.setattr("orbweaver.sandbox.bwrap.is_containerized", lambda: True)
    assert sandbox_available() is True
