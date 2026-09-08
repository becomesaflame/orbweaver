"""Session workspaces (LocalWorkspace plus bubblewrap Bash)."""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import signal
import subprocess
import threading
import time
from pathlib import Path
from uuid import uuid4

from orbweaver.config import settings
from orbweaver.sandbox.policy import in_roots, load_sandbox_policy
from orbweaver.tooltext import normalize_grep_pattern
from orbweaver.uris import resolve_workspace_uri, validate_workspace_uri

log = logging.getLogger(__name__)

DEFAULT_BASH_TIMEOUT_S = 30
MAX_BASH_TIMEOUT_S = 600
GREP_HIT_CAP = 50
GLOB_HIT_CAP = 200
GREP_LINE_CAP = 200
RG_TIMEOUT_SEC = 30
MAX_READ_SIZE = 10 * 1024 * 1024
BINARY_PROBE_BYTES = 8192
_RG_MISSING = (
    "ripgrep (rg) is required for Grep and Glob. "
    "Install ripgrep (Debian/Ubuntu: apt install ripgrep) and ensure rg is on PATH."
)


def ripgrep_path() -> str:
    path = shutil.which("rg")
    if not path:
        raise FileNotFoundError(_RG_MISSING)
    return path


def _is_all_files_glob(spec: str) -> bool:
    return spec in {"", "**/*", "**", "*"}


def _ctx_int(value: object) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return max(0, value)
    if isinstance(value, str) and value.strip():
        try:
            return max(0, int(value))
        except ValueError:
            return 0
    return 0


def _trim_rg_line(line: str, content_cap: int = GREP_LINE_CAP) -> str:
    parts = line.split(":", 2)
    if len(parts) == 3 and parts[1].isdigit():
        return f"{parts[0]}:{parts[1]}:{parts[2][:content_cap]}"
    if len(line) > content_cap + 80:
        return line[: content_cap + 80]
    return line


def _cap_grep_lines(lines: list[str], max_hits: int) -> tuple[list[str], bool]:
    """Keep context, but stop after max_hits match lines (`path:lineno:`)."""
    out: list[str] = []
    matches = 0
    for line in lines:
        parts = line.split(":", 2)
        is_match = len(parts) == 3 and parts[1].isdigit()
        if is_match:
            matches += 1
            if matches > max_hits:
                return out, True
        out.append(_trim_rg_line(line))
    return out, matches > max_hits


def _bash_permissions(permissions=None, unsandboxed: bool = False) -> frozenset[str]:
    out: set[str] = set()
    if unsandboxed:
        out.add("all")
    if isinstance(permissions, str):
        permissions = [permissions]
    for item in permissions or []:
        val = str(item).strip().lower()
        if val in {"all", "full_network"}:
            out.add(val)
    return frozenset(out)


def _as_int(value, default: int | None = None) -> int | None:
    if value is None or value == "":
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return default


def resolve_bash_timeout(
    timeout=None,
    block_until_ms=None,
    *,
    default: int = DEFAULT_BASH_TIMEOUT_S,
) -> int:
    """Seconds to wait. Default 30s; cap at 600s (10 minutes)."""
    seconds = _as_int(timeout)
    if seconds is None:
        ms = _as_int(block_until_ms)
        if ms is None:
            seconds = default
        else:
            seconds = max(1, (ms + 999) // 1000) if ms > 0 else 0
    return max(0, min(int(seconds), MAX_BASH_TIMEOUT_S))


def _decode_captured(data) -> str:
    if data is None:
        return ""
    if isinstance(data, bytes):
        return data.decode("utf-8", "replace")
    return str(data)


def _terminate_process(proc: subprocess.Popen) -> None:
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


def _timeout_message(timeout: int, output: str) -> str:
    body = (output or "")[-200_000:]
    prefix = f"timeout: command exceeded {timeout}s"
    return f"{prefix}\n{body}" if body else prefix


class _BashJob:
    def __init__(self, job_id: str, command: str, timeout: int, proc: subprocess.Popen, cleanup=None):
        self.job_id = job_id
        self.command = command
        self.timeout = timeout
        self.proc = proc
        self.cleanup = cleanup
        self.started_at = time.monotonic()
        self.status = "running"
        self.returncode: int | None = None
        self.output = ""
        self._done = threading.Event()
        self._thread = threading.Thread(target=self._wait, daemon=True)
        self._thread.start()

    def _wait(self) -> None:
        try:
            try:
                stdout, stderr = self.proc.communicate(timeout=self.timeout)
                self.status = "exited"
                self.returncode = self.proc.returncode
                self.output = (_decode_captured(stdout) + _decode_captured(stderr))[-200_000:]
            except subprocess.TimeoutExpired:
                _terminate_process(self.proc)
                stdout, stderr = self.proc.communicate(timeout=5)
                self.status = "timeout"
                self.returncode = self.proc.returncode
                self.output = _timeout_message(
                    self.timeout, _decode_captured(stdout) + _decode_captured(stderr)
                )
        except Exception as e:
            self.status = "error"
            self.output = str(e)
        finally:
            if self.cleanup is not None:
                try:
                    self.cleanup()
                except Exception:
                    log.exception("failed to clean up background bash job")
            self._done.set()

    def wait(self, seconds: float) -> bool:
        return self._done.wait(timeout=max(0.0, seconds))

    def snapshot(self) -> str:
        payload: dict = {
            "job_id": self.job_id,
            "status": self.status,
            "command": self.command,
            "timeout": self.timeout,
        }
        if self.status == "running":
            payload["elapsed_s"] = round(time.monotonic() - self.started_at, 3)
        else:
            payload["returncode"] = self.returncode
            payload["output"] = self.output
        return json.dumps(payload)


class LocalWorkspace:
    host_reads = True

    def __init__(
        self,
        workspace_uri: str,
        workspace_root: str,
        *,
        host_reads: bool = True,
    ) -> None:
        self.uri = validate_workspace_uri(workspace_uri)
        self.root = resolve_workspace_uri(self.uri, workspace_root)
        self.host_reads = host_reads
        self._jobs: dict[str, _BashJob] = {}

    def _policy(self):
        return load_sandbox_policy(self.root)

    def _resolve(self, raw: str, *, write: bool = False) -> Path:
        from orbweaver.permissions.rules import path_is_always_denied

        if path_is_always_denied(raw):
            raise PermissionError(f"path denied: {raw}")
        spec = (raw or "").strip()
        if not spec:
            raise PermissionError("empty path")
        if spec.startswith(("/", "~")):
            path = Path(spec).expanduser().resolve()
            absolute = True
        else:
            path = (self.root / spec).resolve()
            absolute = False
        policy = self._policy()
        if write:
            if not in_roots(path, policy.readwrite_roots(self.root)):
                raise PermissionError(f"path escapes working set: {raw}")
            return path
        if in_roots(path, policy.working_set_roots(self.root)):
            return path
        if self.host_reads and absolute:
            return path
        raise PermissionError(f"path escapes workspace: {raw}")

    def _display(self, path: Path) -> str:
        try:
            return str(path.relative_to(self.root))
        except ValueError:
            return str(path)

    def read(self, path: str) -> str:
        return self._resolve(path).read_text(encoding="utf-8")

    def stat_size(self, path: str) -> int:
        """Byte size after workspace path resolution. Directories raise IsADirectoryError."""
        target = self._resolve(path)
        if target.is_dir():
            raise IsADirectoryError(f"[Errno 21] Is a directory: '{target}'")
        return target.stat().st_size

    def read_prefix(self, path: str, n: int = BINARY_PROBE_BYTES) -> bytes:
        """First n bytes after workspace path resolution."""
        target = self._resolve(path)
        if target.is_dir():
            raise IsADirectoryError(f"[Errno 21] Is a directory: '{target}'")
        with target.open("rb") as handle:
            return handle.read(n)

    def read_text_for_tool(self, path: str, *, max_size: int = MAX_READ_SIZE) -> str:
        """Stat, reject oversized or NUL-prefixed binary files, then decode UTF-8."""
        size = self.stat_size(path)
        if size > max_size:
            raise OSError(f"file too large ({size} bytes; max {max_size})")
        probe = self.read_prefix(path, BINARY_PROBE_BYTES)
        if b"\x00" in probe:
            raise OSError("appears to be binary")
        return self.read(path)

    def write(self, path: str, content: str) -> None:
        p = self._resolve(path, write=True)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")

    def str_replace(
        self, path: str, old: str, new: str, *, replace_all: bool = False
    ) -> str:
        if not old:
            return "error: old_string is required and must be non-empty"
        target = self._resolve(path, write=True)
        if not target.is_file():
            return f"error: file not found: {path}"
        current = target.read_text(encoding="utf-8")
        n = current.count(old)
        if n == 0:
            return f"error: old_string not found in {path}"
        if n > 1 and not replace_all:
            return (
                f"error: old_string matched {n} times in {path}; "
                "include more surrounding context for a unique match, or set replace_all true"
            )
        updated = current.replace(old, new) if replace_all else current.replace(old, new, 1)
        target.write_text(updated, encoding="utf-8")
        noun = "occurrence" if n == 1 else "occurrences"
        return f"updated {path} ({n} {noun})"

    def write_bytes(self, path: str, data: bytes) -> str:
        p = self._resolve(path, write=True)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        return self._display(p)

    def delete(self, path: str) -> str:
        p = self._resolve(path, write=True)
        if p == self.root.resolve():
            raise PermissionError(f"refusing to delete workspace root: {path}")
        if not p.exists():
            return f"not found: {path}"
        if p.is_symlink() or p.is_file():
            p.unlink()
            return f"deleted {self._display(p)}"
        if p.is_dir():
            shutil.rmtree(p)
            return f"deleted {self._display(p)}"
        raise PermissionError(f"cannot delete {path}")

    def read_bytes(self, path: str) -> bytes:
        return self._resolve(path).read_bytes()

    def _rg_base(self) -> list[str]:
        return [
            ripgrep_path(),
            "--no-config",
            "--color=never",
            "--hidden",
            "--glob",
            "!.git/**",
        ]

    def _run_rg(self, args: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
        try:
            return subprocess.run(
                args,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=RG_TIMEOUT_SEC,
                check=False,
            )
        except FileNotFoundError as e:
            raise FileNotFoundError(_RG_MISSING) from e
        except subprocess.TimeoutExpired as e:
            raise TimeoutError("ripgrep timed out") from e

    def _rg_files(self, search_root: Path, glob: str, *, relative: bool) -> list[str]:
        cmd = self._rg_base() + ["--files"]
        if not _is_all_files_glob(glob):
            cmd.extend(["--glob", glob])
        proc = self._run_rg(cmd, cwd=search_root)
        if proc.returncode not in (0, 1):
            err = (proc.stderr or proc.stdout or "ripgrep failed").strip()
            raise RuntimeError(err)
        hits: list[str] = []
        for raw in (proc.stdout or "").splitlines():
            if not raw:
                continue
            path = (search_root / raw).resolve() if not Path(raw).is_absolute() else Path(raw)
            try:
                self._resolve(str(path))
            except PermissionError:
                continue
            if not path.is_file():
                continue
            hits.append(raw if relative else str(path))
            if len(hits) >= GLOB_HIT_CAP:
                break
        return hits

    def _absolute_glob_parts(self, spec: str) -> tuple[Path, str]:
        expanded = Path(spec).expanduser()
        acc = Path(expanded.anchor or "/")
        rest: list[str] = []
        seen_meta = False
        for part in expanded.parts[1:]:
            if seen_meta or any(ch in part for ch in "*?["):
                seen_meta = True
                rest.append(part)
            else:
                acc = acc / part
        while not acc.exists() and acc != acc.parent:
            rest.insert(0, acc.name)
            acc = acc.parent
        return acc, "/".join(rest) if rest else "*"

    def glob(self, pattern: str) -> list[str]:
        spec = (pattern or "").strip() or "**/*"
        if spec.startswith(("/", "~")):
            root, gpat = self._absolute_glob_parts(spec)
            try:
                self._resolve(str(root))
            except PermissionError:
                return []
            return self._rg_files(root, gpat, relative=False)
        return self._rg_files(self.root, spec, relative=True)

    def grep(
        self,
        pattern: str,
        glob: str = "**/*",
        *,
        file_type: str | None = None,
        after: int = 0,
        before: int = 0,
        context: int = 0,
        max_hits: int = GREP_HIT_CAP,
    ) -> list[str]:
        spec = normalize_grep_pattern(pattern)
        if not spec:
            return []
        cmd = self._rg_base() + [
            "--no-heading",
            "--line-number",
            "--max-columns",
            str(GREP_LINE_CAP),
            "--max-columns-preview",
        ]
        if not _is_all_files_glob((glob or "").strip()):
            cmd.extend(["--glob", glob.strip()])
        rg_type = (file_type or "").strip()
        if rg_type:
            cmd.extend(["--type", rg_type])
        ctx = _ctx_int(context)
        after_n = _ctx_int(after)
        before_n = _ctx_int(before)
        if ctx:
            cmd.extend(["-C", str(ctx)])
        else:
            if after_n:
                cmd.extend(["-A", str(after_n)])
            if before_n:
                cmd.extend(["-B", str(before_n)])
        try:
            re.compile(spec)
        except re.error:
            cmd.append("--fixed-strings")
            spec = (pattern or "").strip()
        cmd.extend(["--", spec, "."])
        proc = self._run_rg(cmd, cwd=self.root)
        if proc.returncode == 1:
            return []
        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout or "ripgrep failed").strip()
            return [f"error: {err}"]
        raw_lines = [ln for ln in (proc.stdout or "").splitlines() if ln != ""]
        hits, truncated = _cap_grep_lines(raw_lines, max_hits)
        if truncated:
            hits.append(f"[truncated at {max_hits} hits]")
        return hits

    def _raw_bash(self, command: str, timeout: int) -> str:
        try:
            proc = subprocess.run(
                command,
                shell=True,
                cwd=self.root,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as e:
            return _timeout_message(timeout, _decode_captured(e.stdout) + _decode_captured(e.stderr))
        return ((proc.stdout or "") + (proc.stderr or ""))[-200_000:]

    def _spawn_raw(self, command: str) -> subprocess.Popen:
        return subprocess.Popen(
            command,
            shell=True,
            cwd=self.root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )

    def collect_job(self, job_id: str, wait_s: float = 0) -> str:
        job = self._jobs.get(job_id)
        if job is None:
            return json.dumps({"job_id": job_id, "status": "unknown", "error": "no such job"})
        job.wait(wait_s)
        return job.snapshot()

    def _start_job(self, command: str, timeout: int, proc: subprocess.Popen, cleanup=None) -> str:
        job_id = f"bj_{uuid4().hex[:12]}"
        self._jobs[job_id] = _BashJob(job_id, command, timeout, proc, cleanup=cleanup)
        return self._jobs[job_id].snapshot()

    def bash(
        self,
        command: str = "",
        timeout: int | None = None,
        sandbox: bool = True,
        unsandboxed: bool = False,
        permissions: list[str] | tuple[str, ...] | None = None,
        background: bool = False,
        job_id: str | None = None,
        block_until_ms: int | None = None,
    ) -> str:
        if job_id:
            wait_default = 0 if timeout is None and block_until_ms is None else DEFAULT_BASH_TIMEOUT_S
            wait_s = resolve_bash_timeout(timeout, block_until_ms, default=wait_default)
            return self.collect_job(str(job_id), wait_s=wait_s)
        seconds = resolve_bash_timeout(timeout, block_until_ms)
        cmd = command if isinstance(command, str) else str(command or "")
        if not cmd.strip():
            return "error: command is required unless job_id is set"
        perms = _bash_permissions(permissions, unsandboxed)
        use_raw = "all" in perms or not sandbox or not settings.orbweaver_sandbox
        if use_raw:
            if background:
                return self._start_job(cmd, seconds, self._spawn_raw(cmd))
            return self._raw_bash(cmd, seconds)
        from orbweaver.sandbox.bwrap import (
            SandboxUnavailable,
            is_containerized,
            run_sandboxed,
            spawn_sandboxed,
        )

        if is_containerized():
            if background:
                return self._start_job(cmd, seconds, self._spawn_raw(cmd))
            return self._raw_bash(cmd, seconds)
        try:
            if background:
                session = spawn_sandboxed(
                    cmd,
                    self.root,
                    policy=self._policy(),
                    full_network="full_network" in perms,
                )
                return self._start_job(cmd, seconds, session.proc, cleanup=session.close)
            return run_sandboxed(
                cmd,
                self.root,
                seconds,
                policy=self._policy(),
                full_network="full_network" in perms,
            )
        except SandboxUnavailable as e:
            if settings.orbweaver_sandbox_fail_if_unavailable:
                return f"sandbox_unavailable: {e}"
            if background:
                return self._start_job(cmd, seconds, self._spawn_raw(cmd))
            return self._raw_bash(cmd, seconds)

    def propose_patch(self, path: str, old: str, new: str) -> dict:
        target = self._resolve(path, write=True)
        current = target.read_text(encoding="utf-8") if target.exists() else ""
        if old and old not in current:
            return {"ok": False, "error": "old_string not found", "path": path, "current": current}
        updated = current.replace(old, new, 1) if old else new
        return {"ok": True, "path": path, "old": old, "new": new, "updated": updated}


WORKSPACE_KIND_LOCAL = "local"


def normalize_workspace_kind(kind: str | None) -> str:
    """DockerWorkspace is gone; leftover 'docker' rows become local."""
    del kind
    return WORKSPACE_KIND_LOCAL


def apply_local_workspace_kind(jsonld: dict) -> bool:
    if jsonld.get("workspace_kind") == WORKSPACE_KIND_LOCAL:
        return False
    jsonld["workspace_kind"] = WORKSPACE_KIND_LOCAL
    return True


def bind_workspace(jsonld: dict, workspace_root: str):
    """LocalWorkspace for this session; persist if jsonld.workspace_kind was rewritten."""
    changed = apply_local_workspace_kind(jsonld)
    uri = str(jsonld.get("workspace_uri") or "workspace:default")
    return LocalWorkspace(uri, workspace_root), WORKSPACE_KIND_LOCAL, changed


def make_workspace(kind: str, uri: str, workspace_root: str):
    del kind
    return LocalWorkspace(uri, workspace_root)
