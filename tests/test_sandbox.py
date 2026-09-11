from pathlib import Path

import pytest
from sandbox_mounts import assert_ro_bind_dests_creatable

from orbweaver.config import settings
from orbweaver.sandbox.bwrap import build_bwrap_argv, sandbox_available
from orbweaver.sandbox.policy import SandboxPolicy
from orbweaver.workspace import LocalWorkspace


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
    root_i = next(
        i
        for i, a in enumerate(argv[:-2])
        if a == "--ro-bind" and argv[i + 1] == "/" and argv[i + 2] == "/"
    )
    dev_i = argv.index("--dev")
    assert argv[dev_i + 1] == "/dev"
    assert root_i < dev_i, "host --ro-bind / / must not clobber --dev /dev"
    assert_ro_bind_dests_creatable(argv)


def test_ro_bind_openssh_onto_workspace_tmp_is_rejected():
    """Reproduce production 0.6.0: stash dest under workspace tmp while / is still RO."""
    dest = "/home/orbweaver/workspaces/.orbweaver-missing-openssh-stash"
    assert not Path(dest).exists()
    argv = [
        "bwrap",
        "--unshare-user",
        "--ro-bind",
        "/",
        "/",
        "--tmpfs",
        "/tmp",
        "--ro-bind",
        "/usr/bin/ssh",
        dest,
        "--bind",
        "/home/orbweaver/workspaces/orbweaver2",
        "/home/orbweaver/workspaces/orbweaver2",
        "--",
        "true",
    ]
    with pytest.raises(AssertionError, match="Can't create file"):
        assert_ro_bind_dests_creatable(argv)


def test_bwrap_full_network_skips_unshare_net(tmp_path: Path):
    argv = build_bwrap_argv("echo hi", tmp_path, tmp_path / "tmp", full_network=True)
    assert "--unshare-net" not in argv
    assert "--unshare-user" in argv
    assert_ro_bind_dests_creatable(argv)


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
    hidden.mkdir()
    policy = SandboxPolicy(deny_read=(hidden,))
    argv = build_bwrap_argv("true", tmp_path, tmp_path / "tmp", policy=policy)
    dest = str(hidden.resolve())
    assert argv[argv.index(dest) - 1] == "--tmpfs"


def test_bwrap_skips_missing_deny_read(tmp_path: Path):
    missing = tmp_path / "no-such-gnupg"
    policy = SandboxPolicy(deny_read=(missing,))
    argv = build_bwrap_argv("true", tmp_path, tmp_path / "tmp", policy=policy)
    assert str(missing) not in argv
    assert str(missing.resolve()) not in argv


def test_bwrap_hides_deny_read_file(tmp_path: Path):
    secret = tmp_path / "token"
    secret.write_text("nope", encoding="utf-8")
    policy = SandboxPolicy(deny_read=(secret,))
    argv = build_bwrap_argv("true", tmp_path, tmp_path / "tmp", policy=policy)
    dest = str(secret.resolve())
    i = argv.index(dest)
    assert argv[i - 2 : i] == ["--ro-bind", "/dev/null"]


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


def test_workspace_bash_forwards_timeout_to_sandbox(tmp_path: Path, monkeypatch):
    """One-shot path (persistent shell off): timeout reaches run_sandboxed unchanged."""
    monkeypatch.setattr(settings, "orbweaver_sandbox", True)
    monkeypatch.setattr(settings, "orbweaver_persistent_shell", False)
    monkeypatch.setattr("orbweaver.sandbox.bwrap.is_containerized", lambda: False)
    seen = {}

    def fake_run(command, root, timeout, **kwargs):
        seen["timeout"] = timeout
        seen["command"] = command
        return "ok"

    monkeypatch.setattr("orbweaver.sandbox.bwrap.run_sandboxed", fake_run)
    ws = LocalWorkspace("workspace:default", str(tmp_path))
    assert ws.bash("sleep 31", timeout=45) == f"[cwd {tmp_path.resolve()}]\nok"
    assert seen["timeout"] == 45
    assert "sleep 31\n" in seen["command"]  # wrapped with the cwd sentinel trap


def test_workspace_bash_forwards_timeout_to_session_shell(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(settings, "orbweaver_sandbox", True)
    monkeypatch.setattr(settings, "orbweaver_persistent_shell", True)
    monkeypatch.setattr("orbweaver.sandbox.bwrap.is_containerized", lambda: False)
    from orbweaver.sandbox.shell import ShellResult

    seen = {}

    class FakeShell:
        def run(self, command, timeout, *, cwd=None):
            seen["timeout"] = timeout
            seen["command"] = command
            seen["cwd"] = cwd
            return ShellResult("exited", 0, "ok\n", "/tmp/elsewhere")

    monkeypatch.setattr(
        "orbweaver.sandbox.shell.get_session_shell", lambda *a, **k: FakeShell()
    )
    ws = LocalWorkspace("workspace:default", str(tmp_path))
    assert ws.bash("sleep 31", timeout=45) == "[cwd /tmp/elsewhere]\nok\n"
    assert seen == {"timeout": 45, "command": "sleep 31", "cwd": None}
    assert ws.current_cwd() == "/tmp/elsewhere"


def test_sandbox_available_in_container(monkeypatch):
    monkeypatch.setattr(settings, "orbweaver_sandbox", True)
    monkeypatch.setattr("orbweaver.sandbox.bwrap.is_containerized", lambda: True)
    assert sandbox_available() is True


def test_bwrap_does_not_ro_bind_git_metadata(tmp_path: Path):
    """Workspace .git is working-set writeable so clone/init/fetch can run."""
    hooks = tmp_path / ".git" / "hooks"
    hooks.mkdir(parents=True)
    config = tmp_path / ".git" / "config"
    config.write_text("[core]\n\trepositoryformatversion = 0\n", encoding="utf-8")
    argv = build_bwrap_argv("true", tmp_path, tmp_path / "tmp")
    protected = {str(hooks.resolve()), str(config.resolve())}
    i = 0
    while i < len(argv):
        if argv[i] in {"--ro-bind", "--ro-bind-try"} and i + 2 < len(argv):
            assert argv[i + 2] not in protected
            i += 3
            continue
        i += 1
