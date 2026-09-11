"""Persistent per-session sandbox shell.

One long-lived ``bwrap`` per session (same policy, binds, network namespace and
proxy relay as a one-shot ``run_sandboxed``) runs a small Python shell server.
The gateway talks to it over the bwrap process's stdin/stdout pipes with one
JSON object per line. Each ``run`` request executes the command as a fresh
``bash -c`` child in its own process group, so a per-call timeout kills the
foreground job without killing the server. The working directory and exported
environment carry from one command to the next; ``spawn`` starts a background
job inside the *same* sandbox, so ``curl localhost:PORT`` from a later command
reaches it.

Sessions are reaped on idle (``ORBWEAVER_SHELL_IDLE_TIMEOUT_S``), on turn
cancel, and on gateway shutdown. Set ``ORBWEAVER_PERSISTENT_SHELL=0`` to use the
one-shot path for every command.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import shlex
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, NoReturn
from uuid import uuid4

from orbweaver.config import settings
from orbweaver.sandbox import bwrap as _bwrap
from orbweaver.sandbox.bwrap import SandboxUnavailable, terminate_process
from orbweaver.sandbox.policy import SandboxPolicy, load_sandbox_policy

if TYPE_CHECKING:
    from orbweaver.sandbox.proxy import DomainProxy

log = logging.getLogger(__name__)

DEFAULT_IDLE_TIMEOUT_S = 30 * 60
START_TIMEOUT_S = 20.0
RESPONSE_GRACE_S = 20.0
OUTPUT_CAP = 200_000
REAPER_INTERVAL_S = 30.0

# Runs inside the sandbox as ``python3 -c SOURCE TOKEN``. Keep it dependency-free.
SHELL_SERVER_SOURCE = r'''
import json, os, signal, subprocess, sys, tempfile, threading, time, uuid

token = sys.argv[1]
state = {"cwd": os.getcwd(), "env": dict(os.environ)}
work = tempfile.mkdtemp(prefix="ow-shell-", dir="/tmp")
out = sys.stdout
lock = threading.Lock()
jobs = {}
counter = [0]
CAP = 200000
DROP_ENV = ("_", "SHLVL", "PWD", "OLDPWD")


def reply(obj):
    with lock:
        out.write(json.dumps(obj) + "\n")
        out.flush()


def fresh(prefix):
    counter[0] += 1
    return os.path.join(work, "%s%d" % (prefix, counter[0]))


def kill_group(proc):
    for sig, wait in ((signal.SIGTERM, 2.0), (signal.SIGKILL, 5.0)):
        if proc.poll() is not None:
            return
        try:
            os.killpg(proc.pid, sig)
        except OSError:
            pass
        try:
            proc.wait(timeout=wait)
            return
        except subprocess.TimeoutExpired:
            continue


def tail(path):
    try:
        with open(path, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            fh.seek(max(0, size - CAP))
            return fh.read().decode("utf-8", "replace")
    except OSError:
        return ""


def pick_cwd(req):
    cwd = req.get("cwd") or state["cwd"]
    if not os.path.isdir(cwd):
        cwd = state["cwd"] if os.path.isdir(state["cwd"]) else "/"
    return cwd


def start(cmd, cwd, outp, script=None):
    fh = open(outp, "wb")
    try:
        return subprocess.Popen(
            ["bash", "-c", script if script is not None else cmd],
            cwd=cwd,
            env=state["env"],
            stdin=subprocess.DEVNULL,
            stdout=fh,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    finally:
        fh.close()


def absorb_state(cwdf, envf):
    try:
        with open(cwdf, "rb") as fh:
            cwd = fh.read().decode("utf-8", "replace")
        if cwd and os.path.isdir(cwd):
            state["cwd"] = cwd
    except OSError:
        pass
    try:
        with open(envf, "rb") as fh:
            raw = fh.read()
    except OSError:
        return
    env = {}
    for item in raw.split(b"\0"):
        if b"=" not in item:
            continue
        k, v = item.split(b"=", 1)
        env[k.decode("utf-8", "replace")] = v.decode("utf-8", "replace")
    for k in DROP_ENV:
        env.pop(k, None)
    if env.get("PATH"):
        state["env"] = env


def run(req):
    cmd = str(req.get("command") or "")
    timeout = float(req.get("timeout") or 120)
    cwd = pick_cwd(req)
    outp, cwdf, envf = fresh("out"), fresh("cwd"), fresh("env")
    script = (
        "trap '__ow_rc=$?; printf %%s \"$PWD\" > %s; env -0 > %s; exit $__ow_rc' EXIT\n%s\n"
        % (cwdf, envf, cmd)
    )
    proc = start(cmd, cwd, outp, script)
    status = "exited"
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        status = "timeout"
        kill_group(proc)
    data = tail(outp)
    if status == "exited":
        absorb_state(cwdf, envf)
    for p in (outp, cwdf, envf):
        try:
            os.unlink(p)
        except OSError:
            pass
    return {"status": status, "returncode": proc.returncode, "output": data, "cwd": state["cwd"]}


class Job:
    def __init__(self, job_id, cmd, timeout, proc, outp):
        self.job_id = job_id
        self.command = cmd
        self.timeout = timeout
        self.proc = proc
        self.outp = outp
        self.started = time.monotonic()
        self.status = "running"
        self.done = threading.Event()
        threading.Thread(target=self.watch, daemon=True).start()

    def watch(self):
        try:
            self.proc.wait(timeout=self.timeout)
            self.status = "exited"
        except subprocess.TimeoutExpired:
            kill_group(self.proc)
            self.status = "timeout"
        self.done.set()

    def snapshot(self):
        payload = {
            "job_id": self.job_id,
            "status": self.status,
            "command": self.command,
            "timeout": self.timeout,
        }
        if self.status == "running":
            payload["elapsed_s"] = round(time.monotonic() - self.started, 3)
        else:
            payload["returncode"] = self.proc.returncode
            body = tail(self.outp)
            if self.status == "timeout":
                prefix = "timeout: command exceeded %ss" % self.timeout
                body = prefix + "\n" + body if body else prefix
            payload["output"] = body
        return payload


def spawn(req):
    cmd = str(req.get("command") or "")
    timeout = int(req.get("timeout") or 120)
    cwd = pick_cwd(req)
    outp = fresh("bg")
    job_id = "bj_" + uuid.uuid4().hex[:12]
    proc = start(cmd, cwd, outp)
    jobs[job_id] = Job(job_id, cmd, timeout, proc, outp)
    return jobs[job_id].snapshot()


def poll(req):
    job = jobs.get(str(req.get("job_id") or ""))
    if job is None:
        return {"known": False}
    job.done.wait(max(0.0, float(req.get("wait") or 0)))
    payload = job.snapshot()
    payload["known"] = True
    return payload


def kill_all():
    for job in list(jobs.values()):
        if job.proc.poll() is None:
            kill_group(job.proc)


reply({"ready": True, "token": token, "pid": os.getpid()})
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    req = None
    try:
        req = json.loads(line)
        op = req.get("op")
        if op == "run":
            resp = run(req)
        elif op == "spawn":
            resp = spawn(req)
        elif op == "poll":
            resp = poll(req)
        elif op == "ping":
            resp = {"ok": True, "cwd": state["cwd"], "jobs": len(jobs)}
        elif op == "close":
            kill_all()
            reply({"ok": True})
            break
        else:
            resp = {"error": "unknown op"}
    except Exception as e:  # noqa: BLE001 - the protocol must never die on one request
        resp = {"error": str(e)}
    resp["seq"] = req.get("seq") if isinstance(req, dict) else None
    reply(resp)
'''


class ShellStartError(Exception):
    """The persistent shell server did not come up; callers fall back to one-shot Bash."""


class ShellDead(Exception):
    """The shell stopped answering; the caller should retry on the one-shot path."""


@dataclass
class ShellResult:
    status: str
    returncode: int | None
    output: str
    cwd: str | None


def _server_command(token: str) -> str:
    return f"exec python3 -c {shlex.quote(SHELL_SERVER_SOURCE.strip())} {shlex.quote(token)}"


class SessionShell:
    """One sandboxed shell server. Thread-safe: requests are serialized on a lock."""

    def __init__(
        self,
        workspace_root: Path,
        *,
        policy: SandboxPolicy | None = None,
        full_network: bool = False,
        start_timeout: float = START_TIMEOUT_S,
    ) -> None:
        self.root = Path(workspace_root).resolve()
        self.policy = policy or load_sandbox_policy(self.root)
        self.full_network = full_network
        self.start_timeout = start_timeout
        self.proc: subprocess.Popen[str] | None = None
        self.proxy: DomainProxy | None = None
        self.last_used = time.monotonic()
        self.started_at: float | None = None
        self._lock = threading.Lock()
        self._lines: queue.Queue[str | None] = queue.Queue()
        self._stderr: deque[str] = deque(maxlen=200)
        self._seq = 0
        self._dead = False

    # -- lifecycle ---------------------------------------------------------

    @property
    def alive(self) -> bool:
        return self.proc is not None and not self._dead and self.proc.poll() is None

    def start(self) -> None:
        # Attribute access (not a from-import) so tests that monkeypatch bwrap apply here.
        if _bwrap.is_containerized():
            raise SandboxUnavailable("containerized hosts use unsandboxed bash")
        if not _bwrap.bwrap_path():
            raise SandboxUnavailable("bwrap is not installed")
        token = uuid4().hex
        # Same proxy relay, seccomp filter, cleared environment and rlimits as a
        # one-shot run_sandboxed: the persistent shell is not a weaker sandbox.
        argv, proxy, seccomp_fd = _bwrap._prepare_sandbox(
            _server_command(token),
            self.root,
            policy=self.policy,
            full_network=self.full_network,
        )
        self.proxy = proxy
        try:
            self.proc = subprocess.Popen(
                argv,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                start_new_session=True,
                pass_fds=(seccomp_fd,) if seccomp_fd is not None else (),
            )
        except FileNotFoundError as e:
            self.close()
            raise SandboxUnavailable(f"bwrap not found: {e}") from e
        except OSError as e:
            self.close()
            raise SandboxUnavailable(f"bwrap failed to start: {e}") from e
        finally:
            # bwrap inherited its own copy; the program is fully read before exec.
            if seccomp_fd is not None:
                os.close(seccomp_fd)
        threading.Thread(target=self._pump_stdout, daemon=True, name="ow-shell-out").start()
        threading.Thread(target=self._pump_stderr, daemon=True, name="ow-shell-err").start()
        deadline = time.monotonic() + self.start_timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self._fail_start("shell server handshake timed out")
            try:
                line = self._lines.get(timeout=remaining)
            except queue.Empty:
                self._fail_start("shell server handshake timed out")
            if line is None:
                self._fail_start("shell server exited before handshake")
            try:
                obj = json.loads(line)
            except ValueError:
                continue
            if isinstance(obj, dict) and obj.get("ready") and obj.get("token") == token:
                break
        self.started_at = time.monotonic()
        self.last_used = self.started_at

    def _fail_start(self, reason: str) -> NoReturn:
        err = "".join(self._stderr)[-800:]
        self.close()
        lowered = err.lower()
        if "operation not permitted" in lowered and "bwrap" in lowered:
            raise SandboxUnavailable(err)
        raise ShellStartError(f"{reason}: {err.strip()}" if err.strip() else reason)

    def _pump_stdout(self) -> None:
        proc = self.proc
        if proc is None or proc.stdout is None:
            self._lines.put(None)
            return
        try:
            for line in proc.stdout:
                self._lines.put(line)
        except (OSError, ValueError):
            pass
        self._lines.put(None)

    def _pump_stderr(self) -> None:
        proc = self.proc
        if proc is None or proc.stderr is None:
            return
        try:
            for line in proc.stderr:
                self._stderr.append(line)
        except (OSError, ValueError):
            pass

    def close(self) -> None:
        self._dead = True
        proc = self.proc
        if proc is not None:
            try:
                if proc.stdin is not None and proc.poll() is None:
                    proc.stdin.write(json.dumps({"op": "close"}) + "\n")
                    proc.stdin.flush()
            except (OSError, ValueError):
                pass
            try:
                proc.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                pass
            terminate_process(proc)
            for stream in (proc.stdin, proc.stdout, proc.stderr):
                try:
                    if stream is not None:
                        stream.close()
                except OSError:
                    pass
        if self.proxy is not None:
            try:
                self.proxy.close()
            except Exception:
                log.exception("failed to close session shell proxy")
            self.proxy = None

    # -- protocol ----------------------------------------------------------

    def _request(self, payload: dict, wait_s: float) -> dict:
        with self._lock:
            if not self.alive or self.proc is None or self.proc.stdin is None:
                raise ShellDead("session shell is not running")
            self._seq += 1
            payload = {**payload, "seq": self._seq}
            try:
                self.proc.stdin.write(json.dumps(payload) + "\n")
                self.proc.stdin.flush()
            except (OSError, ValueError) as e:
                self._dead = True
                raise ShellDead(f"session shell write failed: {e}") from e
            deadline = time.monotonic() + wait_s + RESPONSE_GRACE_S
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self._dead = True
                    raise ShellDead("session shell stopped responding")
                try:
                    line = self._lines.get(timeout=remaining)
                except queue.Empty:
                    self._dead = True
                    raise ShellDead("session shell stopped responding") from None
                if line is None:
                    self._dead = True
                    raise ShellDead("session shell exited")
                try:
                    obj = json.loads(line)
                except ValueError:
                    continue
                if isinstance(obj, dict) and obj.get("seq") == self._seq:
                    self.last_used = time.monotonic()
                    return obj

    def run(self, command: str, timeout: int, *, cwd: str | None = None) -> ShellResult:
        self.last_used = time.monotonic()
        resp = self._request(
            {"op": "run", "command": command, "timeout": timeout, "cwd": cwd}, float(timeout)
        )
        if "error" in resp:
            return ShellResult("error", None, str(resp["error"]), resp.get("cwd"))
        return ShellResult(
            str(resp.get("status") or "exited"),
            resp.get("returncode"),
            str(resp.get("output") or "")[-OUTPUT_CAP:],
            resp.get("cwd"),
        )

    def spawn(self, command: str, timeout: int, *, cwd: str | None = None) -> str:
        self.last_used = time.monotonic()
        resp = self._request(
            {"op": "spawn", "command": command, "timeout": timeout, "cwd": cwd}, 5.0
        )
        if "error" in resp:
            return json.dumps({"status": "error", "error": str(resp["error"])})
        resp.pop("seq", None)
        return json.dumps(resp)

    def poll(self, job_id: str, wait_s: float = 0) -> str | None:
        """Snapshot JSON for a job started by spawn, or None if this shell does not know it."""
        resp = self._request({"op": "poll", "job_id": job_id, "wait": wait_s}, float(wait_s))
        if not resp.get("known"):
            return None
        for key in ("seq", "known"):
            resp.pop(key, None)
        return json.dumps(resp)


# -- per-session registry ----------------------------------------------------

_REGISTRY_LOCK = threading.Lock()
_SHELLS: dict[str, SessionShell] = {}
_KEY_LOCKS: dict[str, threading.Lock] = {}
_CWD: dict[str, str] = {}
_REAPER: threading.Thread | None = None
_REAPER_STOP = threading.Event()


def persistent_shell_enabled() -> bool:
    return bool(getattr(settings, "orbweaver_persistent_shell", True))


def idle_timeout_s() -> int:
    raw = getattr(settings, "orbweaver_shell_idle_timeout_s", DEFAULT_IDLE_TIMEOUT_S)
    try:
        return max(1, int(raw))
    except (TypeError, ValueError):
        return DEFAULT_IDLE_TIMEOUT_S


def session_cwd(key: str) -> str | None:
    return _CWD.get(key)


def set_session_cwd(key: str, cwd: str | None) -> None:
    if cwd:
        _CWD[key] = cwd


def _key_lock(key: str) -> threading.Lock:
    with _REGISTRY_LOCK:
        lock = _KEY_LOCKS.get(key)
        if lock is None:
            lock = _KEY_LOCKS[key] = threading.Lock()
        return lock


def peek_session_shell(key: str) -> SessionShell | None:
    shell = _SHELLS.get(key)
    return shell if shell is not None and shell.alive else None


def get_session_shell(
    key: str,
    workspace_root: Path,
    *,
    policy: SandboxPolicy | None = None,
) -> SessionShell:
    """Return the live shell for ``key``; start one (or restart on policy change)."""
    pol = policy or load_sandbox_policy(workspace_root)
    root = Path(workspace_root).resolve()
    with _key_lock(key):
        shell = _SHELLS.get(key)
        if shell is not None and (not shell.alive or shell.policy != pol or shell.root != root):
            shell.close()
            _SHELLS.pop(key, None)
            shell = None
        if shell is None:
            shell = SessionShell(root, policy=pol)
            shell.start()
            _SHELLS[key] = shell
            _ensure_reaper()
        return shell


def close_session_shell(key: str) -> bool:
    with _REGISTRY_LOCK:
        shell = _SHELLS.pop(key, None)
    if shell is None:
        return False
    shell.close()
    return True


def close_all_session_shells() -> int:
    with _REGISTRY_LOCK:
        shells = list(_SHELLS.items())
        _SHELLS.clear()
    for _key, shell in shells:
        try:
            shell.close()
        except Exception:
            log.exception("failed to close session shell")
    return len(shells)


def reap_idle_shells(*, now: float | None = None, idle_s: int | None = None) -> list[str]:
    """Close shells idle longer than ``idle_s`` (or dead ones). Returns the reaped keys."""
    limit = idle_s if idle_s is not None else idle_timeout_s()
    clock = now if now is not None else time.monotonic()
    victims: list[tuple[str, SessionShell]] = []
    with _REGISTRY_LOCK:
        for key, shell in list(_SHELLS.items()):
            if not shell.alive or clock - shell.last_used >= limit:
                _SHELLS.pop(key, None)
                victims.append((key, shell))
    for key, shell in victims:
        try:
            shell.close()
        except Exception:
            log.exception("failed to reap session shell %s", key)
    return [key for key, _shell in victims]


def _reaper_loop() -> None:
    while not _REAPER_STOP.wait(REAPER_INTERVAL_S):
        try:
            reap_idle_shells()
        except Exception:
            log.exception("session shell reaper failed")


def _ensure_reaper() -> None:
    global _REAPER
    with _REGISTRY_LOCK:
        if _REAPER is not None and _REAPER.is_alive():
            return
        _REAPER_STOP.clear()
        _REAPER = threading.Thread(target=_reaper_loop, daemon=True, name="ow-shell-reaper")
        _REAPER.start()
