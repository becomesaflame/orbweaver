"""Linux bubblewrap sandbox for LocalWorkspace bash."""

from __future__ import annotations

import asyncio
import logging
import os
import re
import resource
import shutil
import signal
import subprocess
import time
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any
from uuid import uuid4

from orbweaver.config import settings
from orbweaver.procs import (
    STATUS_INTERRUPTED,
    STATUS_TIMEOUT,
    BashInterrupted,
    communicate_async,
    start_shell,
)
from orbweaver.sandbox.environment import build_sandbox_env, setenv_args
from orbweaver.sandbox.errors import label_sandbox_output
from orbweaver.sandbox.policy import (
    PROTECTED_WRITE_REL,
    SandboxPolicy,
    load_sandbox_policy,
)
from orbweaver.sandbox.seccomp import (
    open_seccomp_fd,
    resolve_seccomp_mode,
    seccomp_filter_for_host,
)
from orbweaver.sandbox.ssh import (
    ensure_ssh_sandbox,
    resolv_conf_overlay_args,
    ssh_config_overlay_args,
    ssh_identity_bind_args,
    ssh_private_identity_files,
)
from orbweaver.tooltext import BASH_OUTPUT_CAP, format_bash_result

log = logging.getLogger(__name__)

FALLBACK_RO_BINDS: tuple[str, ...] = (
    "/usr",
    "/bin",
    "/lib",
    "/lib64",
    "/sbin",
    "/etc",
    "/var",
    "/home",
    "/opt",
    "/var/log",
)


class SandboxUnavailable(Exception):
    """bwrap missing, AppArmor blocked, or nested userns failed."""


def _decode_captured(data) -> str:
    if data is None:
        return ""
    if isinstance(data, bytes):
        return data.decode("utf-8", "replace")
    return str(data)


def _timeout_message(timeout: int, output: str, *, returncode: int | None = None) -> str:
    return format_bash_result(
        output, returncode=returncode, elapsed_s=float(timeout), timed_out_after=timeout
    )


def terminate_process(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            proc.terminate()
        except OSError:
            return
    try:
        proc.wait(timeout=2)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            try:
                proc.kill()
            except OSError:
                pass
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            pass


def is_containerized() -> bool:
    return Path("/.dockerenv").exists() or Path("/run/.containerenv").exists()


def bwrap_path() -> str | None:
    override = os.environ.get("ORBWEAVER_BWRAP_PATH", "").strip()
    if override:
        return override if Path(override).exists() else None
    return shutil.which("bwrap")


def sandbox_available() -> bool:
    if not settings.orbweaver_sandbox:
        return False
    if is_containerized():
        return True
    return bwrap_path() is not None


# Isolation flags beyond the baseline user/pid/net namespaces. Each is passed
# only when this bwrap advertises it (see bwrap_supported_flags), so an older
# bubblewrap degrades to the baseline instead of failing to start.
HARDENING_FLAGS: tuple[tuple[str, ...], ...] = (
    ("--unshare-ipc",),
    ("--unshare-uts",),
    ("--unshare-cgroup-try",),
    ("--new-session",),
    ("--cap-drop", "ALL"),
)


@lru_cache(maxsize=4)
def _bwrap_help_flags(exe: str) -> frozenset[str] | None:
    """Flags listed by `bwrap --help`, or None when help cannot be read."""
    try:
        proc = subprocess.run(
            [exe, "--help"], capture_output=True, text=True, timeout=5, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    text = (proc.stdout or "") + (proc.stderr or "")
    flags = frozenset(re.findall(r"--[a-z][a-z0-9-]*", text))
    return flags or None


def bwrap_supported_flags(exe: str | None = None) -> frozenset[str] | None:
    return _bwrap_help_flags(exe or bwrap_path() or "bwrap")


def bwrap_supports(flag: str, exe: str | None = None) -> bool:
    """True when the flag is advertised, or when `--help` could not be parsed."""
    flags = bwrap_supported_flags(exe)
    return True if flags is None else flag in flags


def hardening_args(exe: str | None = None) -> list[str]:
    out: list[str] = []
    for group in HARDENING_FLAGS:
        if bwrap_supports(group[0], exe):
            out.extend(group)
    return out


def _clamp_to_hard_limit(requested: int, which: int) -> int:
    """ulimit cannot raise a hard limit; never ask for more than the gateway has."""
    try:
        _soft, hard = resource.getrlimit(which)
    except (OSError, ValueError):
        return requested
    if hard == resource.RLIM_INFINITY or hard <= 0:
        return requested
    return min(requested, int(hard))


def resource_limits() -> dict[str, int]:
    """Effective limits from settings (0 = disabled), clamped to the host's hard limits."""
    out: dict[str, int] = {}
    procs = int(getattr(settings, "orbweaver_sandbox_max_procs", 0) or 0)
    if procs > 0:
        out["nproc"] = _clamp_to_hard_limit(procs, resource.RLIMIT_NPROC)
    mem_mb = int(getattr(settings, "orbweaver_sandbox_max_mem_mb", 0) or 0)
    if mem_mb > 0:
        out["as_kb"] = _clamp_to_hard_limit(mem_mb * 1024, resource.RLIMIT_AS)
    files = int(getattr(settings, "orbweaver_sandbox_max_open_files", 0) or 0)
    if files > 0:
        out["nofile"] = _clamp_to_hard_limit(files, resource.RLIMIT_NOFILE)
    return out


def rlimit_prologue(limits: dict[str, int] | None = None) -> str:
    """bash lines that lower soft+hard rlimits before the user command runs.

    `ulimit` is a bash builtin, so this works wherever `bash -lc` does and the
    limits are inherited by every child (node, rg, the proxy relay). Each limit
    is set separately so an unsupported one does not stop the others.
    """
    lim = resource_limits() if limits is None else limits
    parts: list[str] = []
    if lim.get("nproc"):
        parts.append(f"ulimit -u {int(lim['nproc'])} 2>/dev/null")
    if lim.get("as_kb"):
        parts.append(f"ulimit -v {int(lim['as_kb'])} 2>/dev/null")
    if lim.get("nofile"):
        parts.append(f"ulimit -n {int(lim['nofile'])} 2>/dev/null")
    if not parts:
        return ""
    return "; ".join(parts) + "\n"


def seccomp_enabled(exe: str | None = None) -> bool:
    setting = str(getattr(settings, "orbweaver_sandbox_seccomp", "auto") or "auto")
    return resolve_seccomp_mode(setting, bwrap_has_flag=bwrap_supports("--seccomp", exe))


def _deny_read_overlay_args(hidden: Path) -> list[str]:
    """Hide an existing denyRead path. Skip missing ones: bwrap cannot mkdir on a ro-bind.

    Directories get an empty ``--tmpfs``; files get ``/dev/null`` bound over them. bwrap
    refuses to mount on a symlink ("Can't create file"), so a symlinked entry (dotfiles
    manager) masks its resolved target instead; the link then points at the mask.
    """
    try:
        if not hidden.exists():
            return []
        target = hidden.resolve() if hidden.is_symlink() else hidden
        dest = str(target)
        if target.is_dir():
            return ["--tmpfs", dest]
        if target.is_file():
            return ["--ro-bind", "/dev/null", dest]
    except OSError:
        return []
    return []


def deny_read_overlay_args(policy: SandboxPolicy) -> list[str]:
    """bwrap mount args that mask every ``policy.deny_read`` entry that exists."""
    args: list[str] = []
    seen: set[str] = set()
    for hidden in policy.deny_read:
        chunk = _deny_read_overlay_args(hidden)
        if not chunk:
            continue
        dest = chunk[-1]
        if dest in seen:
            continue
        seen.add(dest)
        args.extend(chunk)
    return args
# A writable root is `--bind` rw, but files that execute host commands when git
# runs must stay read-only: `.git/config` (core.fsmonitor, core.sshCommand,
# aliases, …), `.git/hooks/*`, and `.gitmodules`. The rest of `.git` stays
# writable so `git add`/`git commit` still work in the sandbox (issue #94).
# Discovery is a bounded walk; repos nested deeper than this are not protected
# (matches sandbox-runtime's --max-depth cap and keeps per-command cost cheap).
GIT_SCAN_MAX_DEPTH = 4
_GIT_SCAN_PRUNE = {
    ".git",
    "node_modules",
    ".orbweaver-tmp",
    ".venv",
    "venv",
    "__pycache__",
}


def git_protected_paths(
    root: Path,
    max_depth: int = GIT_SCAN_MAX_DEPTH,
    *,
    allow_git_config: bool = False,
) -> list[Path]:
    """`.git/config`, `.git/hooks`, and `.gitmodules` for every repo at or under
    `root`, up to `max_depth` directories deep. Missing paths are fine: the caller
    uses `--ro-bind-try`, which skips them (and skips `.git/hooks` for worktrees
    where `.git` is a file). With `allow_git_config`, `.git/config` stays writable
    (hooks and .gitmodules do not)."""
    out: list[Path] = []
    try:
        base = root.resolve()
    except OSError:
        return out
    stack: list[tuple[Path, int]] = [(base, 0)]
    while stack:
        directory, depth = stack.pop()
        git_dir = directory / ".git"
        try:
            if git_dir.is_dir():
                if not allow_git_config:
                    out.append(git_dir / "config")
                out.append(git_dir / "hooks")
        except OSError:
            pass
        gitmodules = directory / ".gitmodules"
        try:
            if gitmodules.is_file():
                out.append(gitmodules)
        except OSError:
            pass
        if depth >= max_depth:
            continue
        try:
            for child in os.scandir(directory):
                if child.name in _GIT_SCAN_PRUNE:
                    continue
                try:
                    is_dir = child.is_dir(follow_symlinks=False)
                except OSError:
                    continue
                if is_dir:
                    stack.append((Path(child.path), depth + 1))
        except OSError:
            continue
    return out


def _dir_chain(path: Path) -> list[str]:
    posix = path.as_posix()
    if not posix.startswith("/"):
        return []
    parts = [p for p in posix.split("/") if p]
    acc: list[str] = []
    out: list[str] = []
    for part in parts[:-1]:
        acc.append(part)
        out.append("/" + "/".join(acc))
    return out


def _is_run_symlink() -> bool:
    try:
        return Path("/var/run").is_symlink()
    except OSError:
        return False


def build_bwrap_argv(
    command: str,
    workspace_root: Path,
    tmp_dir: Path,
    *,
    policy: SandboxPolicy | None = None,
    full_network: bool = False,
    host_root: bool = True,
    seccomp_fd: int | None = None,
    limits: dict[str, int] | None = None,
    environ: Mapping[str, str] | None = None,
) -> list[str]:
    """Argv for one sandboxed `bash -lc command`.

    `seccomp_fd` is a readable fd holding a raw BPF program (see
    `orbweaver.sandbox.seccomp`); the caller must keep it open and pass it to
    Popen via `pass_fds`. `limits` overrides the settings-derived rlimits
    (`resource_limits()`); pass `{}` for none.
    """
    exe = bwrap_path() or "bwrap"
    root = workspace_root.resolve()
    tmp = tmp_dir.resolve()
    pol = policy or SandboxPolicy()
    argv: list[str] = [
        exe,
        "--unshare-user",
        "--unshare-pid",
        "--die-with-parent",
    ]
    argv.extend(hardening_args(exe))
    if not full_network:
        argv.extend(["--unshare-net"])
    if seccomp_fd is not None:
        argv.extend(["--seccomp", str(int(seccomp_fd))])
    if host_root:
        argv.extend(["--ro-bind", "/", "/"])
        if settings.orbweaver_sandbox_hide_sys:
            # Host hardware, firmware, and cgroup layout stay hidden.
            argv.extend(["--tmpfs", "/sys"])
    else:
        for mount in FALLBACK_RO_BINDS:
            argv.extend(["--ro-bind-try", mount, mount])
        argv.extend(
            [
                "--ro-bind-try",
                "/etc/passwd",
                "/etc/passwd",
                "--ro-bind-try",
                "/etc/group",
                "/etc/group",
            ]
        )
    # After the host bind: a private /dev (so /dev/null is writable), /proc, and /tmp.
    argv.extend(["--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp"])
    private_keys = ssh_private_identity_files(ssh_policy=pol.ssh)
    ensure_ssh_sandbox(tmp, proxied=not full_network, identity_files=private_keys)
    argv.extend(ssh_config_overlay_args(tmp))
    argv.extend(["--tmpfs", "/run"])
    if not _is_run_symlink():
        argv.extend(["--tmpfs", "/var/run"])
    argv.extend(resolv_conf_overlay_args(tmp))
    # Credential masks first, then the allowlisted sockets and the ~/.ssh public files go
    # back on top: an SSH_AUTH_SOCK under a hidden directory must survive the tmpfs.
    argv.extend(deny_read_overlay_args(pol))
    seen_dirs: set[str] = set()
    for sock in pol.allow_unix_sockets:
        sock_p = Path(sock)
        for d in _dir_chain(sock_p):
            if d in {"/run", "/var/run"} or d in seen_dirs:
                continue
            seen_dirs.add(d)
            argv.extend(["--dir", d])
        argv.extend(["--ro-bind-try", str(sock_p), str(sock_p)])
    argv.extend(ssh_identity_bind_args(ssh_policy=pol.ssh))
    writable_roots = pol.readwrite_roots(root, tmp)
    for rw in writable_roots:
        argv.extend(["--bind", str(rw), str(rw)])
    for rel in PROTECTED_WRITE_REL:
        protected = root / rel
        argv.extend(["--ro-bind-try", str(protected), str(protected)])
    seen_git: set[str] = set()
    for rw in writable_roots:
        for git_path in git_protected_paths(rw, allow_git_config=pol.allow_git_config):
            dest = str(git_path)
            if dest in seen_git:
                continue
            seen_git.add(dest)
            argv.extend(["--ro-bind-try", dest, dest])
    # Do not inherit the gateway environment (API keys, JWT secret, bot token,
    # DATABASE_URL). Clear it and set an explicit allowlist (#93).
    env = build_sandbox_env(
        environ if environ is not None else os.environ,
        allow=pol.env_allow,
        granted_sockets=pol.allow_unix_sockets,
    )
    env["TMPDIR"] = str(tmp)
    argv.extend(setenv_args(env))
    script = rlimit_prologue(limits) + command
    argv.extend(["--chdir", str(root), "--", "bash", "-lc", script])
    return argv


class SandboxSession:
    """A running bwrap process plus its domain proxy (kept alive until close)."""

    def __init__(self, proc: subprocess.Popen, proxy) -> None:
        self.proc = proc
        self.proxy = proxy

    def close(self) -> None:
        if self.proxy is not None:
            try:
                self.proxy.close()
            except Exception:
                log.exception("failed to close sandbox proxy")
            self.proxy = None


class AsyncSandboxSession:
    """A running bwrap asyncio process plus its domain proxy."""

    def __init__(self, proc: asyncio.subprocess.Process, proxy) -> None:
        self.proc = proc
        self.proxy = proxy

    def close(self) -> None:
        if self.proxy is not None:
            try:
                self.proxy.close()
            except Exception:
                log.exception("failed to close sandbox proxy")
            self.proxy = None


def _prepare_sandbox(
    command: str,
    workspace_root: Path,
    *,
    policy: SandboxPolicy | None,
    full_network: bool,
) -> tuple[list[str], Any, int | None]:
    """Start the domain proxy, open the seccomp fd, and build the bwrap argv.

    Returns (argv, proxy, seccomp_fd); the caller must pass the fd to the child and close it."""
    if is_containerized():
        raise SandboxUnavailable("containerized hosts use unsandboxed bash")
    exe = bwrap_path()
    if not exe:
        raise SandboxUnavailable("bwrap is not installed")
    tmp = workspace_root / ".orbweaver-tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    pol = policy or load_sandbox_policy(workspace_root)
    inner = command
    proxy = None
    if not full_network:
        from orbweaver.sandbox.proxy import DomainProxy, wrap_command_with_proxy

        sock = tmp / f"ow-proxy-{uuid4().hex[:12]}.sock"
        proxy = DomainProxy(sock, pol.network)
        proxy.start()
        inner = wrap_command_with_proxy(command, str(sock))
    seccomp_fd: int | None = None
    try:
        if seccomp_enabled(exe):
            program = seccomp_filter_for_host()
            if program is None:
                raise SandboxUnavailable(
                    "sandbox seccomp is forced on but no syscall table exists for this architecture"
                )
            seccomp_fd = open_seccomp_fd(program)
        argv = build_bwrap_argv(
            inner, workspace_root, tmp, policy=pol, full_network=full_network, seccomp_fd=seccomp_fd
        )
    except Exception:
        if proxy is not None:
            proxy.close()
        if seccomp_fd is not None:
            os.close(seccomp_fd)
        raise
    return argv, proxy, seccomp_fd


def spawn_sandboxed(
    command: str,
    workspace_root: Path,
    *,
    policy: SandboxPolicy | None = None,
    full_network: bool = False,
) -> SandboxSession:
    argv, proxy, seccomp_fd = _prepare_sandbox(
        command, workspace_root, policy=policy, full_network=full_network
    )
    try:
        proc = subprocess.Popen(
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
            pass_fds=(seccomp_fd,) if seccomp_fd is not None else (),
        )
    except FileNotFoundError as e:
        if proxy is not None:
            proxy.close()
        raise SandboxUnavailable(f"bwrap not found: {e}") from e
    except OSError as e:
        if proxy is not None:
            proxy.close()
        raise SandboxUnavailable(f"bwrap failed to start: {e}") from e
    finally:
        # bwrap inherited its own copy; the program is fully read before exec.
        if seccomp_fd is not None:
            os.close(seccomp_fd)
    return SandboxSession(proc, proxy)


async def spawn_sandboxed_async(
    command: str,
    workspace_root: Path,
    *,
    policy: SandboxPolicy | None = None,
    full_network: bool = False,
) -> AsyncSandboxSession:
    """spawn_sandboxed for the event loop: the process is an asyncio subprocess."""
    argv, proxy, seccomp_fd = await asyncio.to_thread(
        _prepare_sandbox, command, workspace_root, policy=policy, full_network=full_network
    )
    try:
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
            pass_fds=(seccomp_fd,) if seccomp_fd is not None else (),
        )
    except FileNotFoundError as e:
        if proxy is not None:
            proxy.close()
        raise SandboxUnavailable(f"bwrap not found: {e}") from e
    except OSError as e:
        if proxy is not None:
            proxy.close()
        raise SandboxUnavailable(f"bwrap failed to start: {e}") from e
    finally:
        if seccomp_fd is not None:
            os.close(seccomp_fd)
    return AsyncSandboxSession(proc, proxy)


def run_sandboxed(
    command: str,
    workspace_root: Path,
    timeout: int = 30,
    *,
    policy: SandboxPolicy | None = None,
    full_network: bool = False,
) -> str:
    if is_containerized():
        return _raw(command, workspace_root, timeout)
    started = time.monotonic()
    session = spawn_sandboxed(command, workspace_root, policy=policy, full_network=full_network)
    try:
        try:
            stdout, stderr = session.proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            terminate_process(session.proc)
            stdout, stderr = session.proc.communicate(timeout=5)
            out = (stdout or "") + (stderr or "")
            return _timeout_message(
                timeout,
                label_sandbox_output(out[-BASH_OUTPUT_CAP:]),
                returncode=session.proc.returncode,
            )
    finally:
        session.close()
    out = (stdout or "") + (stderr or "")
    _raise_if_bwrap_eperm(session.proc.returncode, out)
    return format_bash_result(
        label_sandbox_output(out[-BASH_OUTPUT_CAP:]),
        returncode=session.proc.returncode,
        elapsed_s=time.monotonic() - started,
    )


def _raise_if_bwrap_eperm(returncode: int | None, out: str) -> None:
    if returncode != 0 and "operation not permitted" in out.lower() and "bwrap" in out.lower():
        log.warning("bwrap operation not permitted: %s", out[-500:])
        raise SandboxUnavailable(out[-800:])


async def run_sandboxed_async(
    command: str,
    workspace_root: Path,
    timeout: int = 30,
    *,
    policy: SandboxPolicy | None = None,
    full_network: bool = False,
    cancel: asyncio.Event | None = None,
) -> str:
    """run_sandboxed without blocking the event loop.

    ``cancel`` (the turn's stop event) kills the process group and raises
    BashInterrupted so the caller can record an error tool_result.
    """
    if is_containerized():
        return await _raw_async(command, workspace_root, timeout, cancel=cancel)
    started = time.monotonic()
    session = await spawn_sandboxed_async(
        command, workspace_root, policy=policy, full_network=full_network
    )
    try:
        out, status = await communicate_async(session.proc, timeout, cancel=cancel)
    finally:
        session.close()
    if status == STATUS_INTERRUPTED:
        raise BashInterrupted(label_sandbox_output(out[-BASH_OUTPUT_CAP:]))
    if status == STATUS_TIMEOUT:
        return _timeout_message(
            timeout,
            label_sandbox_output(out[-BASH_OUTPUT_CAP:]),
            returncode=session.proc.returncode,
        )
    _raise_if_bwrap_eperm(session.proc.returncode, out)
    return format_bash_result(
        label_sandbox_output(out[-BASH_OUTPUT_CAP:]),
        returncode=session.proc.returncode,
        elapsed_s=time.monotonic() - started,
    )


async def _raw_async(
    command: str,
    workspace_root: Path,
    timeout: int,
    *,
    cancel: asyncio.Event | None = None,
) -> str:
    started = time.monotonic()
    proc = await start_shell(command, workspace_root)
    out, status = await communicate_async(proc, timeout, cancel=cancel)
    if status == STATUS_INTERRUPTED:
        raise BashInterrupted(out[-BASH_OUTPUT_CAP:])
    if status == STATUS_TIMEOUT:
        return _timeout_message(timeout, out, returncode=proc.returncode)
    return format_bash_result(
        out, returncode=proc.returncode, elapsed_s=time.monotonic() - started
    )


def _raw(command: str, workspace_root: Path, timeout: int) -> str:
    started = time.monotonic()
    try:
        proc = subprocess.run(
            command,
            shell=True,
            cwd=str(workspace_root),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as e:
        return _timeout_message(timeout, _decode_captured(e.stdout) + _decode_captured(e.stderr))
    return format_bash_result(
        (proc.stdout or "") + (proc.stderr or ""),
        returncode=proc.returncode,
        elapsed_s=time.monotonic() - started,
    )
