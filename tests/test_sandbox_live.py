"""Run bubblewrap the way production Bash does. Skip when bwrap cannot start."""

from __future__ import annotations

import os

import pytest

from orbweaver.config import settings
from orbweaver.sandbox.bwrap import (
    SandboxUnavailable,
    is_containerized,
    run_sandboxed,
    sandbox_available,
)
from orbweaver.workspace import LocalWorkspace


def test_ci_installs_bubblewrap():
    if not os.environ.get("GITHUB_ACTIONS"):
        return
    assert sandbox_available(), "CI must apt-install bubblewrap so live sandbox tests can run where netns allows"


def _loopback_blocked(text: object) -> bool:
    lowered = str(text).lower()
    return "rtm_newaddr" in lowered or "loopback:" in lowered


@pytest.fixture
def live_root(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "orbweaver_sandbox", True)
    monkeypatch.setattr(settings, "orbweaver_sandbox_fail_if_unavailable", True)
    if not sandbox_available() or is_containerized():
        pytest.skip("bubblewrap not available")
    try:
        out = run_sandboxed("true", tmp_path, timeout=10)
    except SandboxUnavailable as e:
        if _loopback_blocked(e):
            pytest.skip(f"bwrap netns loopback not permitted: {e}")
        raise
    if _loopback_blocked(out):
        pytest.skip(out[-200:])
    if "sandbox_unavailable" in out:
        pytest.skip(out[-200:])
    return tmp_path


def test_sandboxed_echo_does_not_hit_readonly_openssh(live_root):
    """Production 0.6.0: every Bash command died creating ow-ssh/openssh on a RO bind."""
    out = run_sandboxed("echo orbweaver-sandbox-ok", live_root, timeout=15)
    assert "Can't create file" not in out
    assert "Read-only file system" not in out
    assert "sandbox_denied" not in out
    assert "sandbox_unavailable" not in out
    assert "orbweaver-sandbox-ok" in out


def test_workspace_bash_echo_matches_production_path(live_root):
    ws = LocalWorkspace("workspace:default", str(live_root))
    out = ws.bash("echo orbweaver-sandbox-ok")
    assert "Can't create file" not in out
    assert "Read-only file system" not in out
    assert "sandbox_denied" not in out
    assert "orbweaver-sandbox-ok" in out


def test_sandboxed_ssh_skips_host_config_d(live_root):
    """Production clone: OpenSSH aborted on ssh_config.d ownership inside the userns."""
    out = run_sandboxed("ssh -G github.com", live_root, timeout=15)
    assert "Bad owner or permissions" not in out
    assert "sandbox_denied" not in out
    assert "proxycommand" in out.lower()
    assert "github.com" in out.lower()


def test_full_network_can_resolve_github(live_root):
    """Production clone with permissions full_network: /run tmpfs hid systemd-resolved."""
    out = run_sandboxed(
        "python3 -c \"import socket; print(socket.getaddrinfo('github.com', 22)[0])\"",
        live_root,
        timeout=15,
        full_network=True,
    )
    assert "Temporary failure" not in out
    assert "Could not resolve" not in out
    assert "sandbox_denied" not in out
    assert "gaierror" not in out.lower()
    assert "github.com" in out.lower() or "AddressFamily" in out


def test_sandboxed_ssh_uses_host_identity(live_root):
    from orbweaver.sandbox.ssh import ssh_private_identity_files

    keys = ssh_private_identity_files()
    if not keys:
        pytest.skip("no host SSH identity file")
    out = run_sandboxed("ssh -G github.com", live_root, timeout=15)
    assert any(key.name in out for key in keys)


def test_sandboxed_git_ls_remote_github(live_root):
    """Production orbweaver2: git clone git@github.com:… over sandboxed ssh."""
    from orbweaver.sandbox.ssh import ssh_private_identity_files

    if not ssh_private_identity_files():
        pytest.skip("no host SSH identity file")
    out = run_sandboxed(
        "git ls-remote git@github.com:becomesaflame/orbweaver.git",
        live_root,
        timeout=45,
    )
    assert "Could not resolve hostname" not in out
    assert "Bad owner or permissions" not in out
    assert "Permission denied (publickey)" not in out
    assert "refs/heads" in out or "HEAD" in out


def test_sandboxed_git_init_remote_and_rm(live_root):
    """Production clone: a later Bash turn could not write or rm .git/config and hooks."""
    init = run_sandboxed("git init", live_root, timeout=15)
    assert (live_root / ".git" / "config").is_file()
    assert (live_root / ".git" / "hooks").is_dir()
    assert "Read-only file system" not in init
    remote = run_sandboxed(
        "git remote add origin git@github.com:becomesaflame/orbweaver.git",
        live_root,
        timeout=15,
    )
    assert "Device or resource busy" not in remote
    assert "could not write config" not in remote.lower()
    assert "origin" in run_sandboxed("git remote", live_root, timeout=10)
    rm = run_sandboxed("rm -rf .git", live_root, timeout=15)
    assert "Read-only file system" not in rm
    assert "Device or resource busy" not in rm
    assert "sandbox_denied" not in rm
    assert not (live_root / ".git").exists()


def test_sandboxed_bash_timeout_expiry(live_root):
    ws = LocalWorkspace("workspace:default", str(live_root))
    out = ws.bash("sleep 5", timeout=1)
    assert "timed out after 1s" in out


def test_sandboxed_bash_raised_timeout_completes(live_root):
    """A command that would miss the old 30s default must succeed when timeout is raised."""
    ws = LocalWorkspace("workspace:default", str(live_root))
    out = ws.bash("sleep 31; echo sandbox-over-default", timeout=40)
    assert "sandbox-over-default" in out
    assert "timeout:" not in out


def test_sandboxed_bash_background_start_and_collect(live_root):
    import json

    ws = LocalWorkspace("workspace:default", str(live_root))
    started = json.loads(ws.bash("sleep 0.4; echo sandbox-job-done", background=True, timeout=15))
    assert started["status"] == "running"
    finished = json.loads(ws.collect_job(started["job_id"], wait_s=10))
    assert finished["status"] == "exited"
    assert "sandbox-job-done" in finished["output"]
