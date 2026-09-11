"""Issue #105: cwd persists between Bash calls; one sandbox per session; 120 s default."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from uuid import uuid4

import pytest

from orbweaver.config import settings
from orbweaver.git_ritual import RITUAL_MARK, repo_for_command
from orbweaver.sandbox import shell as shell_mod
from orbweaver.sandbox.bwrap import (
    SandboxUnavailable,
    is_containerized,
    run_sandboxed,
    sandbox_available,
)
from orbweaver.workspace import (
    DEFAULT_BASH_TIMEOUT_S,
    MAX_BASH_TIMEOUT_S,
    LocalWorkspace,
    parse_cwd_sentinel,
    resolve_bash_timeout,
    wrap_cwd_tracking,
)

# -- no bubblewrap needed ------------------------------------------------------


def test_default_timeout_is_120_and_cap_600():
    assert DEFAULT_BASH_TIMEOUT_S == 120
    assert MAX_BASH_TIMEOUT_S == 600
    assert resolve_bash_timeout() == 120
    assert resolve_bash_timeout(9999) == 600


def test_bash_tool_description_says_cwd_persists():
    from orbweaver.agent import TOOL_SPEC

    bash = next(t for t in TOOL_SPEC if t["name"] == "Bash")
    desc = bash["description"]
    assert "working directory persists between commands" in desc
    assert "120" in desc
    assert "120" in bash["input_schema"]["properties"]["timeout"]["description"]


def test_parse_cwd_sentinel_strips_marker_and_reads_exit():
    out = "hello\n\n__OW_CWD__abc=/tmp/x\n__OW_RC__abc=3\nstderr text\n"
    cleaned, cwd, rc = parse_cwd_sentinel(out, "abc")
    assert cleaned == "hello\nstderr text\n"
    assert cwd == "/tmp/x"
    assert rc == 3
    assert parse_cwd_sentinel("no marker", "abc") == ("no marker", None, None)


def test_wrap_cwd_tracking_runs_in_saved_cwd_and_reports_exit(tmp_path: Path):
    """The sentinel survives `exit N` because it is printed from an EXIT trap."""
    sub = tmp_path / "sub"
    sub.mkdir()
    script = wrap_cwd_tracking("pwd; cd ..; exit 4", str(sub), tmp_path, "tok")
    proc = subprocess.run(["bash", "-c", script], capture_output=True, text=True, check=False)
    cleaned, cwd, rc = parse_cwd_sentinel(proc.stdout, "tok")
    assert cleaned.strip() == str(sub.resolve())
    assert Path(cwd or "").resolve() == tmp_path.resolve()
    assert rc == 4
    assert proc.returncode == 4


def test_raw_bash_cwd_persists_between_calls(tmp_path: Path, monkeypatch):
    """Cheap path (sandbox off / permissions all): cd, then pwd in a second call."""
    monkeypatch.setattr(settings, "orbweaver_sandbox", False)
    ws = LocalWorkspace("workspace:default", str(tmp_path))
    first = ws.bash("cd /tmp")
    assert first.startswith("[cwd /tmp]")
    second = ws.bash("pwd")
    assert second.splitlines()[0] == "[cwd /tmp]"
    assert second.splitlines()[1] == "/tmp"
    assert ws.current_cwd() == "/tmp"
    assert ws.last_command_cwd == "/tmp"
    failed = ws.bash("false")
    assert failed.startswith("[cwd /tmp | exit 1]")


def test_raw_bash_cwd_is_keyed_by_session_across_instances(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(settings, "orbweaver_sandbox", False)
    key = f"test-{uuid4().hex}"
    LocalWorkspace("workspace:default", str(tmp_path)).bash("cd /tmp", session_key=key)
    out = LocalWorkspace("workspace:default", str(tmp_path)).bash("pwd", session_key=key)
    assert "\n/tmp" in out
    other = LocalWorkspace("workspace:default", str(tmp_path)).bash("pwd")
    assert "/tmp\n" not in other.split("\n", 1)[1]


def test_repo_for_command_uses_persisted_cwd(tmp_path: Path):
    nested = tmp_path / "proj"
    nested.mkdir()
    subprocess.run(["git", "init", "-q", str(nested)], check=True)
    assert repo_for_command(tmp_path, "git status") is None
    assert repo_for_command(tmp_path, "git status", cwd=nested) == nested.resolve()
    (nested / "pkg").mkdir()
    assert repo_for_command(tmp_path, "cd pkg && git status", cwd=nested) == nested.resolve()


def test_reap_idle_shells_closes_dead_and_stale():
    class Fake:
        def __init__(self, alive: bool, last_used: float):
            self.alive = alive
            self.last_used = last_used
            self.closed = False

        def close(self) -> None:
            self.closed = True

    now = time.monotonic()
    stale = Fake(True, now - 4000)
    fresh = Fake(True, now)
    dead = Fake(False, now)
    keys = [f"reap-{uuid4().hex}" for _ in range(3)]
    shell_mod._SHELLS[keys[0]] = stale  # type: ignore[assignment]
    shell_mod._SHELLS[keys[1]] = fresh  # type: ignore[assignment]
    shell_mod._SHELLS[keys[2]] = dead  # type: ignore[assignment]
    try:
        reaped = shell_mod.reap_idle_shells(now=now, idle_s=1800)
        assert set(reaped) == {keys[0], keys[2]}
        assert stale.closed and dead.closed and not fresh.closed
        assert keys[1] in shell_mod._SHELLS
    finally:
        shell_mod._SHELLS.pop(keys[1], None)


# -- bubblewrap-executing --------------------------------------------------------


def _loopback_blocked(text: object) -> bool:
    lowered = str(text).lower()
    return "rtm_newaddr" in lowered or "loopback:" in lowered


@pytest.fixture
def live_root(tmp_path_factory, monkeypatch):
    # Short path: the proxy's AF_UNIX socket lives under <root>/.orbweaver-tmp (108-char cap).
    tmp_path = tmp_path_factory.mktemp("ow")
    monkeypatch.setattr(settings, "orbweaver_sandbox", True)
    monkeypatch.setattr(settings, "orbweaver_sandbox_fail_if_unavailable", True)
    monkeypatch.setattr(settings, "orbweaver_persistent_shell", True)
    if not sandbox_available() or is_containerized():
        pytest.skip("bubblewrap not available")
    try:
        out = run_sandboxed("true", tmp_path, timeout=10)
    except SandboxUnavailable as e:
        if _loopback_blocked(e):
            pytest.skip(f"bwrap netns loopback not permitted: {e}")
        raise
    if _loopback_blocked(out) or "sandbox_unavailable" in out:
        pytest.skip(out[-200:])
    return tmp_path


def _ws(root: Path) -> LocalWorkspace:
    return LocalWorkspace("workspace:default", str(root), session_key=f"live-{uuid4().hex}")


def test_persistent_shell_cwd_and_env_persist(live_root):
    ws = _ws(live_root)
    assert "[cwd /tmp]" in ws.bash("cd /tmp && export OW_TEST_VAR=kept")
    out = ws.bash("pwd; echo var=$OW_TEST_VAR")
    assert out.splitlines() == ["[cwd /tmp]", "/tmp", "var=kept"]
    shell = shell_mod.peek_session_shell(ws.session_key)
    assert shell is not None and shell.alive


def test_persistent_shell_background_server_reachable_from_next_call(live_root):
    ws = _ws(live_root)
    started = json.loads(ws.bash("python3 -m http.server 8123", background=True, timeout=60))
    assert started["status"] == "running"
    assert started["job_id"].startswith("bj_")
    code = ""
    for _ in range(20):
        out = ws.bash("curl -s -o /dev/null -w '%{http_code}' localhost:8123", timeout=10)
        code = out.splitlines()[-1].strip()
        if code == "200":
            break
        time.sleep(0.25)
    assert code == "200"
    running = json.loads(ws.bash(job_id=started["job_id"]))
    assert running["status"] == "running"
    # A later turn builds a new LocalWorkspace for the same session and still sees the job.
    again = LocalWorkspace("workspace:default", str(live_root), session_key=ws.session_key)
    assert json.loads(again.bash(job_id=started["job_id"]))["status"] == "running"


def test_persistent_shell_timeout_kills_foreground_only(live_root):
    ws = _ws(live_root)
    ws.bash("cd /tmp")
    before = shell_mod.peek_session_shell(ws.session_key)
    assert before is not None
    started = time.monotonic()
    out = ws.bash("sleep 30", timeout=1)
    assert "timeout: command exceeded 1s" in out
    assert time.monotonic() - started < 10
    after_out = ws.bash("echo still-here; pwd")
    assert "still-here" in after_out
    assert "[cwd /tmp]" in after_out
    assert shell_mod.peek_session_shell(ws.session_key) is before
    assert before.alive


def test_persistent_shell_survives_exit_and_reports_code(live_root):
    ws = _ws(live_root)
    out = ws.bash("echo bye; exit 7")
    assert out.startswith(f"[cwd {live_root.resolve()} | exit 7]")
    assert "bye" in out
    assert "ok" in ws.bash("echo ok")


def test_persistent_shell_disabled_falls_back_to_oneshot_with_cwd(live_root, monkeypatch):
    monkeypatch.setattr(settings, "orbweaver_persistent_shell", False)
    ws = _ws(live_root)
    ws.bash("cd /tmp")
    out = ws.bash("pwd")
    assert out.splitlines() == ["[cwd /tmp]", "/tmp"]
    assert shell_mod.peek_session_shell(ws.session_key) is None


def test_full_network_oneshot_shares_cwd_with_persistent_shell(live_root):
    ws = _ws(live_root)
    ws.bash("cd /tmp")
    out = ws.bash("pwd", permissions=["full_network"])
    assert out.splitlines() == ["[cwd /tmp]", "/tmp"]
    ws.bash("cd /usr", permissions=["full_network"])
    assert ws.bash("pwd").splitlines() == ["[cwd /usr]", "/usr"]


def test_close_session_shell_reaps_sandbox_and_next_call_restarts(live_root):
    ws = _ws(live_root)
    started = json.loads(ws.bash("sleep 600", background=True, timeout=600))
    shell = shell_mod.peek_session_shell(ws.session_key)
    assert shell is not None and shell.proc is not None
    proc = shell.proc
    assert ws.close_shell()
    assert proc.poll() is not None
    assert shell_mod.peek_session_shell(ws.session_key) is None
    unknown = json.loads(ws.bash(job_id=started["job_id"]))
    assert unknown["status"] == "unknown"
    assert "fresh" in ws.bash("echo fresh")
    fresh = shell_mod.peek_session_shell(ws.session_key)
    assert fresh is not None and fresh is not shell


def test_idle_reaper_closes_shell(live_root):
    ws = _ws(live_root)
    ws.bash("true")
    shell = shell_mod.peek_session_shell(ws.session_key)
    assert shell is not None
    reaped = shell_mod.reap_idle_shells(now=time.monotonic() + 10_000, idle_s=1800)
    assert ws.session_key in reaped
    assert not shell.alive


@pytest.mark.asyncio
async def test_run_tools_bash_cwd_persists_across_workspace_instances(live_root):
    """Each turn binds a new LocalWorkspace; the session id keys the shell."""
    from orbweaver.agent import run_tools
    from orbweaver.store import reset_store_for_tests

    store = reset_store_for_tests()
    sid = uuid4()
    first = LocalWorkspace("workspace:default", str(live_root))
    out = await run_tools("Bash", {"command": "mkdir -p proj && cd proj && git init -q"}, {
        "workspace": first,
        "store": store,
        "session_id": sid,
    })
    assert "[cwd" in out
    second = LocalWorkspace("workspace:default", str(live_root))
    ctx = {"workspace": second, "store": store, "session_id": sid}
    out = await run_tools("Bash", {"command": "pwd"}, ctx)
    assert out.splitlines()[1] == str((live_root / "proj").resolve())
    ritual = await run_tools("Bash", {"command": "git status --short"}, ctx)
    assert RITUAL_MARK in ritual
    assert f"repo={(live_root / 'proj').resolve()}" in ritual
    assert (await second.abash("echo async-ok", session_key=str(sid))).endswith("async-ok\n")
    shell_mod.close_session_shell(str(sid))
