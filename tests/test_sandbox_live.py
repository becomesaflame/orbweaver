"""Run bubblewrap the way production Bash does. Skip when bwrap cannot start."""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path

import pytest

from orbweaver.config import settings
from orbweaver.sandbox.bwrap import (
    SandboxUnavailable,
    is_containerized,
    run_sandboxed,
    sandbox_available,
    seccomp_enabled,
)
from orbweaver.workspace import LocalWorkspace


def test_ci_installs_bubblewrap():
    if not os.environ.get("GITHUB_ACTIONS"):
        return
    assert sandbox_available(), "CI must apt-install bubblewrap so live sandbox tests can run where netns allows"


def _loopback_blocked(text: object) -> bool:
    lowered = str(text).lower()
    return "rtm_newaddr" in lowered or "loopback:" in lowered


def _proc_status_field(name: str) -> str:
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith(name + ":"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return ""


def _nested_in_seccomp_sandbox() -> bool:
    """pytest itself is under a seccomp filter (e.g. inside an Orbweaver sandbox): unshare is EPERM."""
    return _proc_status_field("Seccomp") == "2"


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
        if _nested_in_seccomp_sandbox() and "not permitted" in str(e).lower():
            pytest.skip(f"nested bwrap blocked by the outer seccomp filter: {e}")
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


def _bind_identities_policy(root):
    """Operator opt-in (sandbox.json ``ssh.bindIdentities``): host IdentityFile keys are bound."""
    from dataclasses import replace

    from orbweaver.sandbox.policy import load_sandbox_policy
    from orbweaver.sandbox.ssh import SshPolicy

    return replace(load_sandbox_policy(root), ssh=SshPolicy(bind_identities=True))


def test_sandboxed_ssh_uses_host_identity(live_root):
    from orbweaver.sandbox.ssh import ssh_private_identity_files

    policy = _bind_identities_policy(live_root)
    keys = ssh_private_identity_files(ssh_policy=policy.ssh)
    if not keys:
        pytest.skip("no host SSH identity file")
    out = run_sandboxed("ssh -G github.com", live_root, timeout=15, policy=policy)
    assert any(key.name in out for key in keys)


def test_sandboxed_ssh_default_policy_hides_private_keys(live_root):
    """Issue #95: without bindIdentities no private key is visible; public files are."""
    import os

    from orbweaver.sandbox.ssh import SshPolicy, ssh_private_identity_files, ssh_public_files

    keys = ssh_private_identity_files(ssh_policy=SshPolicy(bind_identities=True))
    if not keys:
        pytest.skip("no host SSH identity file")
    ssh_dir = os.path.expanduser("~/.ssh")
    out = run_sandboxed(f"ls -A {ssh_dir}", live_root, timeout=15)
    listed = set(out.split())
    for key in keys:
        assert key.name not in listed, out
    for public in ssh_public_files():
        assert public.name in listed, out


def test_sandboxed_git_ls_remote_github(live_root):
    """Production orbweaver2: git clone git@github.com:… over sandboxed ssh."""
    from orbweaver.sandbox.ssh import ssh_private_identity_files

    policy = _bind_identities_policy(live_root)
    if not ssh_private_identity_files(ssh_policy=policy.ssh):
        pytest.skip("no host SSH identity file")
    out = run_sandboxed(
        "git ls-remote git@github.com:becomesaflame/orbweaver.git",
        live_root,
        timeout=45,
        policy=policy,
    )
    assert "Could not resolve hostname" not in out
    assert "Bad owner or permissions" not in out
    assert "Permission denied (publickey)" not in out
    assert "refs/heads" in out or "HEAD" in out


def test_sandboxed_git_init_then_config_protected(live_root):
    """Issue #94 (revises #23): git init works and creates .git/config + hooks. A
    later turn cannot rewrite .git/config (config-execution escape) unless
    allowGitConfig is set. .git init and commit/add still work."""
    from orbweaver.sandbox.policy import SandboxPolicy

    init = run_sandboxed("git init", live_root, timeout=15)
    assert (live_root / ".git" / "config").is_file()
    assert (live_root / ".git" / "hooks").is_dir()
    assert "Read-only file system" not in init

    # A later turn sees the now-existing config/hooks mounted read-only.
    denied = run_sandboxed(
        "git remote add origin git@github.com:becomesaflame/orbweaver.git",
        live_root,
        timeout=15,
    )
    assert (
        "could not" in denied.lower()
        or "read-only" in denied.lower()
        or "busy" in denied.lower()
    )
    assert "origin" not in run_sandboxed("git remote", live_root, timeout=10)

    # allowGitConfig re-enables the git remote set-url workflow.
    allow_pol = SandboxPolicy(allow_git_config=True)
    allowed = run_sandboxed(
        "git remote add origin git@github.com:becomesaflame/orbweaver.git",
        live_root,
        timeout=15,
        policy=allow_pol,
    )
    assert "could not write config" not in allowed.lower()
    assert "origin" in run_sandboxed("git remote", live_root, timeout=10)


def test_sandboxed_git_config_is_readonly_but_commit_works(live_root):
    """Issue #94: a pre-existing repo's .git/config and .git/hooks are read-only
    inside the sandbox, but git add/commit still work on the rest of .git."""
    import subprocess

    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@e",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@e",
    }

    def git(*args):
        subprocess.run(["git", "-C", str(live_root), *args], check=True, env=env)

    git("init", "-b", "main")
    git("config", "user.name", "t")
    git("config", "user.email", "t@e")
    (live_root / "a.txt").write_text("a\n", encoding="utf-8")
    git("add", "a.txt")
    git("commit", "-m", "base")

    append = run_sandboxed("echo escaped >> .git/config", live_root, timeout=15)
    assert "Read-only file system" in append
    hook = run_sandboxed("echo x > .git/hooks/pre-commit", live_root, timeout=15)
    assert "Read-only file system" in hook

    commit = run_sandboxed(
        "printf b > b.txt && git add b.txt && "
        "git -c user.name=t -c user.email=t@e commit -m two && git log --oneline",
        live_root,
        timeout=20,
    )
    assert "Read-only file system" not in commit
    assert "two" in commit


def test_sandboxed_env_excludes_gateway_secrets(live_root, monkeypatch):
    """#93: `env` inside bubblewrap printed ANTHROPIC_API_KEY and ORBWEAVER_JWT_SECRET."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-FAKE")
    monkeypatch.setenv("ORBWEAVER_JWT_SECRET", "x")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123:FAKE")
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@localhost/db")
    out = run_sandboxed("env", live_root, timeout=15)
    assert "sandbox_denied" not in out
    assert "sandbox_unavailable" not in out
    assert "ANTHROPIC_API_KEY" not in out
    assert "sk-ant-FAKE" not in out
    assert "ORBWEAVER_JWT_SECRET" not in out
    assert "TELEGRAM_BOT_TOKEN" not in out
    assert "DATABASE_URL" not in out
    assert "PATH=" in out
    assert "HOME=" in out
    # The proxy wrapper still exports its own proxy settings inside the sandbox.
    assert "https_proxy=http://127.0.0.1:" in out
    assert "ORBWEAVER_SSH_PROXY_HOST=127.0.0.1" in out


def test_sandboxed_background_job_excludes_gateway_secrets(live_root, monkeypatch):
    import json

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-FAKE")
    monkeypatch.setenv("ORBWEAVER_JWT_SECRET", "x")
    ws = LocalWorkspace("workspace:default", str(live_root))
    started = json.loads(ws.bash("env", background=True, timeout=15))
    assert started["status"] == "running"
    finished = json.loads(ws.collect_job(started["job_id"], wait_s=10))
    assert finished["status"] == "exited"
    output = finished["output"]
    assert "ANTHROPIC_API_KEY" not in output
    assert "sk-ant-FAKE" not in output
    assert "ORBWEAVER_JWT_SECRET" not in output
    assert "PATH=" in output


def test_sandboxed_bash_timeout_expiry(live_root):
    ws = LocalWorkspace("workspace:default", str(live_root))
    out = ws.bash("sleep 5", timeout=1)
    assert "timeout: command exceeded 1s" in out


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


# --- issue #114 hardening: run bubblewrap and look at the sandboxed process itself ---


def _status_lines(out: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in out.splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            fields[key.strip()] = value.strip()
    return fields


def test_hardened_sandbox_has_no_capabilities_and_no_new_privs(live_root):
    out = run_sandboxed("grep -E '^(CapEff|CapBnd|NoNewPrivs|Seccomp)' /proc/self/status", live_root, timeout=15)
    fields = _status_lines(out)
    assert fields.get("CapEff") == "0000000000000000", out
    assert fields.get("CapBnd") == "0000000000000000", out
    assert fields.get("NoNewPrivs") == "1", out


def test_hardened_sandbox_seccomp_filter_mode(live_root):
    if not seccomp_enabled():
        pytest.skip("seccomp resolved to off on this host")
    out = run_sandboxed("grep -E '^(NoNewPrivs|Seccomp)' /proc/self/status", live_root, timeout=15)
    fields = _status_lines(out)
    assert fields.get("Seccomp") == "2", out
    assert fields.get("NoNewPrivs") == "1", out


def test_hardened_sandbox_seccomp_denies_unshare_with_eperm(live_root):
    """The filter returns EPERM (not SIGSYS) so probing tools keep running."""
    if not seccomp_enabled():
        pytest.skip("seccomp resolved to off on this host")
    probe = (
        "python3 -c \"import ctypes, os; libc = ctypes.CDLL(None, use_errno=True); "
        "r = libc.unshare(0x10000000); print('unshare', r, os.strerror(ctypes.get_errno()))\""
    )
    out = run_sandboxed(probe, live_root, timeout=15)
    assert "unshare -1 Operation not permitted" in out, out
    assert "sandbox_denied: syscall" in out


def test_hardened_sandbox_seccomp_blocks_ptrace(live_root):
    if not seccomp_enabled():
        pytest.skip("seccomp resolved to off on this host")
    if shutil.which("strace") is None:
        pytest.skip("strace not installed")
    out = run_sandboxed("strace -o /dev/null true; echo rc=$?", live_root, timeout=15)
    assert "Operation not permitted" in out, out
    assert "rc=0" not in out


def test_hardened_sandbox_seccomp_off_setting_disables_filter(live_root, monkeypatch):
    monkeypatch.setattr(settings, "orbweaver_sandbox_seccomp", "off")
    out = run_sandboxed("grep -E '^Seccomp:' /proc/self/status", live_root, timeout=15)
    assert _status_lines(out).get("Seccomp") == "0", out


def test_hardened_sandbox_ulimits_are_bounded(live_root, monkeypatch):
    monkeypatch.setattr(settings, "orbweaver_sandbox_max_procs", 300)
    monkeypatch.setattr(settings, "orbweaver_sandbox_max_mem_mb", 1536)
    monkeypatch.setattr(settings, "orbweaver_sandbox_max_open_files", 1024)
    out = run_sandboxed(
        "echo nproc=$(ulimit -u) nproc_hard=$(ulimit -Hu) as=$(ulimit -v) nofile=$(ulimit -n); "
        "ulimit -u 100000 2>/dev/null && echo RAISED",
        live_root,
        timeout=15,
    )
    assert "nproc=300 nproc_hard=300" in out, out
    assert "as=1572864" in out, out
    assert "nofile=1024" in out, out
    assert "RAISED" not in out, "hard limit must not be raisable inside the sandbox"


def test_hardened_sandbox_zero_disables_memory_limit(live_root, monkeypatch):
    monkeypatch.setattr(settings, "orbweaver_sandbox_max_mem_mb", 0)
    out = run_sandboxed("ulimit -v", live_root, timeout=15)
    assert "unlimited" in out, out


def test_hardened_sandbox_hides_sys(live_root):
    out = run_sandboxed("echo sys=[$(ls /sys/class 2>/dev/null)] top=[$(ls /sys 2>/dev/null)]", live_root, timeout=15)
    assert "sys=[] top=[]" in out, out


def test_hardened_sandbox_common_tools_still_run_under_limits(live_root):
    if shutil.which("rg") is None:
        pytest.skip("ripgrep not installed on the host")
    out = run_sandboxed(
        "echo ok; rg --version | head -1; python3 -c \"import json, hashlib, ssl; print('py', json.dumps([1]))\"",
        live_root,
        timeout=20,
    )
    assert "sandbox_denied" not in out, out
    assert "\nok\n" in "\n" + out
    assert "ripgrep" in out, out
    assert "py [1]" in out, out


def test_hardened_sandbox_proxy_relay_survives_new_session(live_root):
    """The in-sandbox relay (background python job) must still answer through the domain proxy."""
    out = run_sandboxed(
        'python3 -c "import urllib.request, urllib.error\n'
        "try:\n"
        "    urllib.request.urlopen('http://denied.example.invalid/', timeout=10)\n"
        "except urllib.error.HTTPError as e:\n"
        "    print('relay', e.code, e.read().decode())\n"
        '"',
        live_root,
        timeout=20,
    )
    assert "relay 403" in out, out
    assert "denied.example.invalid" in out


def test_hardened_sandbox_background_job_runs_with_hardening(live_root):
    import json

    ws = LocalWorkspace("workspace:default", str(live_root))
    started = json.loads(
        ws.bash("grep -E '^NoNewPrivs' /proc/self/status; echo hardened-job-done", background=True, timeout=15)
    )
    assert started["status"] == "running"
    finished = json.loads(ws.collect_job(started["job_id"], wait_s=10))
    assert finished["status"] == "exited"
    assert "hardened-job-done" in finished["output"]
    assert "NoNewPrivs:\t1" in finished["output"]


@pytest.mark.skipif(os.environ.get("ORBWEAVER_TEST_FORKBOMB") != "1", reason="set ORBWEAVER_TEST_FORKBOMB=1")
def test_hardened_sandbox_contains_fork_bomb(live_root, monkeypatch):
    monkeypatch.setattr(settings, "orbweaver_sandbox_max_procs", 64)
    before = len(os.listdir("/proc"))
    t0 = time.monotonic()
    out = run_sandboxed(":(){ :|:& };:; wait", live_root, timeout=5)
    elapsed = time.monotonic() - t0
    assert elapsed < 20, f"fork bomb was not contained in time: {elapsed:.1f}s"
    assert "Resource temporarily unavailable" in out or "timeout: command exceeded" in out, out[-500:]
    time.sleep(1)
    after = len(os.listdir("/proc"))
    assert after < before + 20, "sandboxed processes leaked past the timeout"
