"""Run bubblewrap the way production Bash does. Skip only when bwrap is missing."""

from __future__ import annotations

import os

import pytest

from orbweaver.config import settings
from orbweaver.sandbox.bwrap import is_containerized, run_sandboxed, sandbox_available
from orbweaver.workspace import LocalWorkspace


def test_ci_installs_bubblewrap():
    if not os.environ.get("GITHUB_ACTIONS"):
        return
    assert sandbox_available(), "CI must apt-install bubblewrap so live sandbox tests run"


@pytest.fixture
def live_root(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "orbweaver_sandbox", True)
    monkeypatch.setattr(settings, "orbweaver_sandbox_fail_if_unavailable", True)
    if not sandbox_available() or is_containerized():
        pytest.skip("bubblewrap not available")
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
