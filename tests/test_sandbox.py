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
    assert argv[-3:-1] == ["bash", "-lc"]
    assert argv[-1].endswith("echo hi")
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
    # The fake runner returns bare output (no exit line, no cwd sentinel): only the cwd is known.
    header, body = ws.bash("sleep 31", timeout=45).split("\n", 1)
    assert header == f"[cwd {tmp_path.resolve()}]" and body == "ok"
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
    header, body = ws.bash("sleep 31", timeout=45).split("\n", 1)
    assert header.startswith("[cwd /tmp/elsewhere | exit 0 in ") and body == "ok\n"
    assert seen == {"timeout": 45, "command": "sleep 31", "cwd": None}
    assert ws.current_cwd() == "/tmp/elsewhere"


def test_sandbox_available_in_container(monkeypatch):
    monkeypatch.setattr(settings, "orbweaver_sandbox", True)
    monkeypatch.setattr("orbweaver.sandbox.bwrap.is_containerized", lambda: True)
    assert sandbox_available() is True


def _setenv_map(argv: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for i, a in enumerate(argv):
        if a == "--setenv":
            out[argv[i + 1]] = argv[i + 2]
    return out


_GATEWAY_ENV = {
    "PATH": "/usr/bin:/bin",
    "HOME": "/home/gw",
    "USER": "gw",
    "LANG": "C.UTF-8",
    "LC_ALL": "C.UTF-8",
    "ANTHROPIC_API_KEY": "sk-ant-FAKE",
    "OPENROUTER_API_KEY": "sk-or-FAKE",
    "ORBWEAVER_JWT_SECRET": "x",
    "TELEGRAM_BOT_TOKEN": "123:abc",
    "DATABASE_URL": "postgresql://u:p@localhost/db",
    "HINDSIGHT_API_KEY": "hs-FAKE",
    "MY_PASSWORD": "hunter2",
    "PGHOST": "localhost",
}


def test_bwrap_clears_env_and_sets_allowlist(tmp_path: Path):
    """#93: sandboxed Bash inherited ANTHROPIC_API_KEY, the JWT secret, and the bot token."""
    argv = build_bwrap_argv("env", tmp_path, tmp_path / "tmp", environ=_GATEWAY_ENV)
    assert "--clearenv" in argv
    assert argv.index("--clearenv") < argv.index("--setenv")
    env = _setenv_map(argv)
    assert env["PATH"] == "/usr/bin:/bin"
    assert env["HOME"] == "/home/gw"
    assert env["LC_ALL"] == "C.UTF-8"
    assert env["TMPDIR"] == str((tmp_path / "tmp").resolve())
    for secret in (
        "ANTHROPIC_API_KEY",
        "OPENROUTER_API_KEY",
        "ORBWEAVER_JWT_SECRET",
        "TELEGRAM_BOT_TOKEN",
        "DATABASE_URL",
        "HINDSIGHT_API_KEY",
        "MY_PASSWORD",
        "PGHOST",
    ):
        assert secret not in env
    joined = " ".join(argv)
    assert "sk-ant-FAKE" not in joined
    assert "hunter2" not in joined


def test_bwrap_env_allow_passes_names_but_excludes_win(tmp_path: Path):
    policy = SandboxPolicy(env_allow=("PGHOST", "ANTHROPIC_API_KEY", "MY_PASSWORD", "ORBWEAVER_JWT_SECRET"))
    argv = build_bwrap_argv("env", tmp_path, tmp_path / "tmp", policy=policy, environ=_GATEWAY_ENV)
    env = _setenv_map(argv)
    assert env["PGHOST"] == "localhost"
    assert "ANTHROPIC_API_KEY" not in env
    assert "MY_PASSWORD" not in env
    assert "ORBWEAVER_JWT_SECRET" not in env


def test_bwrap_ssh_auth_sock_only_when_socket_granted(tmp_path: Path):
    sock = tmp_path / "agent.sock"
    environ = {**_GATEWAY_ENV, "SSH_AUTH_SOCK": str(sock)}
    argv = build_bwrap_argv("env", tmp_path, tmp_path / "tmp", environ=environ)
    assert "SSH_AUTH_SOCK" not in _setenv_map(argv)
    granted = SandboxPolicy(allow_unix_sockets=(sock,))
    argv = build_bwrap_argv("env", tmp_path, tmp_path / "tmp", policy=granted, environ=environ)
    assert _setenv_map(argv)["SSH_AUTH_SOCK"] == str(sock)


def _ro_bind_dests(argv: list[str]) -> set[str]:
    dests: set[str] = set()
    i = 0
    while i < len(argv):
        if argv[i] in {"--ro-bind", "--ro-bind-try"} and i + 2 < len(argv):
            dests.add(argv[i + 2])
            i += 3
            continue
        i += 1
    return dests


def _bind_dests(argv: list[str]) -> set[str]:
    dests: set[str] = set()
    i = 0
    while i < len(argv):
        if argv[i] == "--bind" and i + 2 < len(argv):
            dests.add(argv[i + 2])
            i += 3
            continue
        i += 1
    return dests


def test_bwrap_ro_binds_git_config_hooks_gitmodules(tmp_path: Path):
    """Issue #94: .git/config, .git/hooks, and .gitmodules stay read-only inside a
    writable root so a hostile config cannot execute host commands."""
    hooks = tmp_path / ".git" / "hooks"
    hooks.mkdir(parents=True)
    config = tmp_path / ".git" / "config"
    config.write_text("[core]\n\trepositoryformatversion = 0\n", encoding="utf-8")
    gitmodules = tmp_path / ".gitmodules"
    gitmodules.write_text("[submodule \"x\"]\n", encoding="utf-8")
    argv = build_bwrap_argv("true", tmp_path, tmp_path / "tmp")
    ro = _ro_bind_dests(argv)
    assert str(config.resolve()) in ro
    assert str(hooks.resolve()) in ro
    assert str(gitmodules.resolve()) in ro
    # The workspace itself and the rest of .git stay writable.
    assert str(tmp_path.resolve()) in _bind_dests(argv)
    assert str((tmp_path / ".git").resolve()) not in ro
    # The git ro-binds come after the workspace rw --bind so they win.
    joined = argv.index(str(config.resolve()))
    bind_i = next(
        i for i, a in enumerate(argv)
        if a == "--bind" and argv[i + 1] == str(tmp_path.resolve())
    )
    assert bind_i < joined


def test_bwrap_ro_binds_nested_repo_git_metadata(tmp_path: Path):
    nested = tmp_path / "sub" / "pkg"
    (nested / ".git" / "hooks").mkdir(parents=True)
    (nested / ".git" / "config").write_text("[core]\n", encoding="utf-8")
    argv = build_bwrap_argv("true", tmp_path, tmp_path / "tmp")
    ro = _ro_bind_dests(argv)
    assert str((nested / ".git" / "config").resolve()) in ro
    assert str((nested / ".git" / "hooks").resolve()) in ro


def test_bwrap_allow_git_config_keeps_config_writable(tmp_path: Path):
    hooks = tmp_path / ".git" / "hooks"
    hooks.mkdir(parents=True)
    config = tmp_path / ".git" / "config"
    config.write_text("[core]\n", encoding="utf-8")
    policy = SandboxPolicy(allow_git_config=True)
    argv = build_bwrap_argv("true", tmp_path, tmp_path / "tmp", policy=policy)
    ro = _ro_bind_dests(argv)
    assert str(config.resolve()) not in ro
    # Hooks stay read-only regardless of allowGitConfig.
    assert str(hooks.resolve()) in ro


def test_git_protected_paths_respects_depth_and_worktree(tmp_path: Path):
    from orbweaver.sandbox.bwrap import GIT_SCAN_MAX_DEPTH, git_protected_paths

    (tmp_path / ".git" / "hooks").mkdir(parents=True)
    (tmp_path / ".git" / "config").write_text("x", encoding="utf-8")
    # A worktree: .git is a file, so hooks/config never exist there.
    worktree = tmp_path / "wt"
    worktree.mkdir()
    (worktree / ".git").write_text("gitdir: /elsewhere\n", encoding="utf-8")
    # Too deep to be scanned.
    deep = tmp_path
    for part in ["a", "b", "c", "d", "e", "f"][: GIT_SCAN_MAX_DEPTH + 2]:
        deep = deep / part
    (deep / ".git").mkdir(parents=True)
    (deep / ".git" / "config").write_text("x", encoding="utf-8")

    found = {str(p) for p in git_protected_paths(tmp_path)}
    assert str(tmp_path / ".git" / "config") in found
    assert str(tmp_path / ".git" / "hooks") in found
    assert str(worktree / ".git" / "config") not in found
    assert str(deep / ".git" / "config") not in found
