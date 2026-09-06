"""Linux bubblewrap sandbox for LocalWorkspace bash."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path

from orbweaver.config import settings

log = logging.getLogger(__name__)


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


def build_bwrap_argv(command: str, workspace_root: Path, tmp_dir: Path) -> list[str]:
    exe = bwrap_path() or "bwrap"
    root = str(workspace_root.resolve())
    tmp = str(tmp_dir.resolve())
    return [
        exe,
        "--unshare-user",
        "--unshare-pid",
        "--unshare-net",
        "--die-with-parent",
        "--proc",
        "/proc",
        "--dev",
        "/dev",
        "--tmpfs",
        "/tmp",
        "--ro-bind",
        "/usr",
        "/usr",
        "--ro-bind",
        "/bin",
        "/bin",
        "--ro-bind-try",
        "/lib",
        "/lib",
        "--ro-bind-try",
        "/lib64",
        "/lib64",
        "--ro-bind-try",
        "/sbin",
        "/sbin",
        "--ro-bind-try",
        "/etc/passwd",
        "/etc/passwd",
        "--ro-bind-try",
        "/etc/group",
        "/etc/group",
        "--bind",
        root,
        root,
        "--bind",
        tmp,
        tmp,
        "--setenv",
        "TMPDIR",
        tmp,
        "--chdir",
        root,
        "--",
        "bash",
        "-lc",
        command,
    ]


def run_sandboxed(command: str, workspace_root: Path, timeout: int = 30) -> str:
    if is_containerized():
        return _raw(command, workspace_root, timeout)
    exe = bwrap_path()
    if not exe:
        raise SandboxUnavailable("bwrap is not installed")
    tmp = workspace_root / ".orbweaver-tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    argv = build_bwrap_argv(command, workspace_root, tmp)
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
    out = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode != 0 and "operation not permitted" in out.lower():
        log.warning("bwrap operation not permitted: %s", out[-500:])
        raise SandboxUnavailable(out[-800:])
    return out[-200_000:]


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
