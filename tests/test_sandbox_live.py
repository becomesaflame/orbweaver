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
