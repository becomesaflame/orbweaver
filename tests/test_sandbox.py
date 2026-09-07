from pathlib import Path

from orbweaver.config import settings
from orbweaver.sandbox.bwrap import build_bwrap_argv, sandbox_available
from orbweaver.sandbox.policy import SandboxPolicy
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
    assert "--ro-bind" in argv and "/" in argv
    assert argv[argv.index("--tmpfs") + 1] == "/tmp" or "/run" in argv
    assert "--tmpfs" in argv and "/run" in argv


def test_bwrap_full_network_skips_unshare_net(tmp_path: Path):
    argv = build_bwrap_argv("echo hi", tmp_path, tmp_path / "tmp", full_network=True)
    assert "--unshare-net" not in argv
    assert "--unshare-user" in argv


def test_bwrap_binds_sockets_and_hides_run(tmp_path: Path):
    sock = Path("/run/user/1000/systemd/journal/socket")
    policy = SandboxPolicy(allow_unix_sockets=(sock,))
    argv = build_bwrap_argv("true", tmp_path, tmp_path / "tmp", policy=policy)
    joined = " ".join(argv)
    assert "--tmpfs /run" in joined or argv.count("/run") >= 1
    assert str(sock) in argv
    assert "--dir" in argv
    assert "/run/user/1000/systemd" in argv
    assert "--ro-bind-try" in argv


def test_bwrap_deny_read_overlay(tmp_path: Path):
    hidden = tmp_path / "secrets"
    policy = SandboxPolicy(deny_read=(hidden,))
    argv = build_bwrap_argv("true", tmp_path, tmp_path / "tmp", policy=policy)
    assert str(hidden.resolve()) in argv


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
    out = ws.bash("echo hi", permissions=["all"])
    assert "docker is not available" in out


def test_sandbox_available_in_container(monkeypatch):
    monkeypatch.setattr(settings, "orbweaver_sandbox", True)
    monkeypatch.setattr("orbweaver.sandbox.bwrap.is_containerized", lambda: True)
    assert sandbox_available() is True
