"""Linux bubblewrap sandbox for LocalWorkspace bash."""

from __future__ import annotations

import logging
import os
import shutil
import signal
import subprocess
from pathlib import Path
from uuid import uuid4

from orbweaver.config import settings
from orbweaver.sandbox.errors import label_sandbox_output
from orbweaver.sandbox.policy import (
    PROTECTED_WRITE_REL,
    SandboxPolicy,
    load_sandbox_policy,
)
from orbweaver.sandbox.ssh import (
    ensure_ssh_sandbox,
    resolv_conf_overlay_args,
    ssh_config_overlay_args,
    ssh_identity_bind_args,
)

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


def _timeout_message(timeout: int, output: str) -> str:
    body = (output or "")[-200_000:]
    prefix = f"timeout: command exceeded {timeout}s"
    return f"{prefix}\n{body}" if body else prefix


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


def _deny_read_overlay_args(hidden: Path) -> list[str]:
    """Hide an existing denyRead path. Skip missing ones: bwrap cannot mkdir on a ro-bind."""
    try:
        if not hidden.exists():
            return []
        dest = str(hidden)
        if hidden.is_dir():
            return ["--tmpfs", dest]
        if hidden.is_file():
            return ["--ro-bind", "/dev/null", dest]
    except OSError:
        return []
    return []


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
) -> list[str]:
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
    if not full_network:
        argv.extend(["--unshare-net"])
    if host_root:
        argv.extend(["--ro-bind", "/", "/"])
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
    ensure_ssh_sandbox(tmp, proxied=not full_network)
    argv.extend(ssh_config_overlay_args(tmp))
    argv.extend(["--tmpfs", "/run"])
    if not _is_run_symlink():
        argv.extend(["--tmpfs", "/var/run"])
    argv.extend(resolv_conf_overlay_args(tmp))
    seen_dirs: set[str] = set()
    for sock in pol.allow_unix_sockets:
        sock_p = Path(sock)
        for d in _dir_chain(sock_p):
            if d in {"/run", "/var/run"} or d in seen_dirs:
                continue
            seen_dirs.add(d)
            argv.extend(["--dir", d])
        argv.extend(["--ro-bind-try", str(sock_p), str(sock_p)])
    for hidden in pol.deny_read:
        argv.extend(_deny_read_overlay_args(hidden))
    argv.extend(ssh_identity_bind_args())
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
    argv.extend(["--setenv", "TMPDIR", str(tmp), "--chdir", str(root), "--", "bash", "-lc", command])
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


def spawn_sandboxed(
    command: str,
    workspace_root: Path,
    *,
    policy: SandboxPolicy | None = None,
    full_network: bool = False,
) -> SandboxSession:
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
    argv = build_bwrap_argv(
        inner, workspace_root, tmp, policy=pol, full_network=full_network
    )
    try:
        proc = subprocess.Popen(
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
    except FileNotFoundError as e:
        if proxy is not None:
            proxy.close()
        raise SandboxUnavailable(f"bwrap not found: {e}") from e
    except OSError as e:
        if proxy is not None:
            proxy.close()
        raise SandboxUnavailable(f"bwrap failed to start: {e}") from e
    return SandboxSession(proc, proxy)


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
    session = spawn_sandboxed(command, workspace_root, policy=policy, full_network=full_network)
    try:
        try:
            stdout, stderr = session.proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            terminate_process(session.proc)
            stdout, stderr = session.proc.communicate(timeout=5)
            out = (stdout or "") + (stderr or "")
            return _timeout_message(timeout, label_sandbox_output(out[-200_000:]))
    finally:
        session.close()
    out = (stdout or "") + (stderr or "")
    if session.proc.returncode != 0 and "operation not permitted" in out.lower() and "bwrap" in out.lower():
        log.warning("bwrap operation not permitted: %s", out[-500:])
        raise SandboxUnavailable(out[-800:])
    return label_sandbox_output(out[-200_000:])


def _raw(command: str, workspace_root: Path, timeout: int) -> str:
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
    return ((proc.stdout or "") + (proc.stderr or ""))[-200_000:]
