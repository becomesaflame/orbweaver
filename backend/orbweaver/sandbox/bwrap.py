"""Linux bubblewrap sandbox for LocalWorkspace bash."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path

from orbweaver.config import settings
from orbweaver.sandbox.errors import label_sandbox_output
from orbweaver.sandbox.policy import (
    PROTECTED_WRITE_REL,
    SandboxPolicy,
    load_sandbox_policy,
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
    argv.extend(["--tmpfs", "/run"])
    if not _is_run_symlink():
        argv.extend(["--tmpfs", "/var/run"])
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
    for rw in pol.readwrite_roots(root, tmp):
        argv.extend(["--bind", str(rw), str(rw)])
    for rel in PROTECTED_WRITE_REL:
        protected = root / rel
        argv.extend(["--ro-bind-try", str(protected), str(protected)])
    argv.extend(["--setenv", "TMPDIR", str(tmp), "--chdir", str(root), "--", "bash", "-lc", command])
    return argv


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

        sock = tmp / "ow-proxy.sock"
        proxy = DomainProxy(sock, pol.network)
        proxy.start()
        inner = wrap_command_with_proxy(command, str(sock))
    argv = build_bwrap_argv(
        inner, workspace_root, tmp, policy=pol, full_network=full_network
    )
    try:
        proc = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError as e:
        raise SandboxUnavailable(f"bwrap not found: {e}") from e
    except OSError as e:
        raise SandboxUnavailable(f"bwrap failed to start: {e}") from e
    finally:
        if proxy is not None:
            proxy.close()
    out = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode != 0 and "operation not permitted" in out.lower() and "bwrap" in out.lower():
        log.warning("bwrap operation not permitted: %s", out[-500:])
        raise SandboxUnavailable(out[-800:])
    return label_sandbox_output(out[-200_000:])


def _raw(command: str, workspace_root: Path, timeout: int) -> str:
    proc = subprocess.run(
        command,
        shell=True,
        cwd=str(workspace_root),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    return ((proc.stdout or "") + (proc.stderr or ""))[-200_000:]
