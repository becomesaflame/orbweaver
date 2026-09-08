"""Session workspaces (LocalWorkspace plus bubblewrap Bash)."""

from __future__ import annotations

import glob as globmod
import json
import os
import signal
import subprocess
import threading
import time
from pathlib import Path
from uuid import uuid4

from orbweaver.tooltext import grep_regex

from orbweaver.config import settings
from orbweaver.sandbox.policy import in_roots, load_sandbox_policy
from orbweaver.uris import resolve_workspace_uri, validate_workspace_uri

DEFAULT_BASH_TIMEOUT_S = 30
MAX_BASH_TIMEOUT_S = 600


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
                    pass
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

    def write(self, path: str, content: str) -> None:
        p = self._resolve(path, write=True)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")

    def write_bytes(self, path: str, data: bytes) -> str:
        p = self._resolve(path, write=True)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        return self._display(p)

    def read_bytes(self, path: str) -> bytes:
        return self._resolve(path).read_bytes()

    def glob(self, pattern: str) -> list[str]:
        spec = (pattern or "").strip()
        if spec.startswith(("/", "~")):
            expanded = str(Path(spec).expanduser())
            hits = []
            for match in globmod.glob(expanded, recursive=True):
                p = Path(match)
                if p.is_file():
                    try:
                        self._resolve(str(p))
                    except PermissionError:
                        continue
                    hits.append(str(p))
                if len(hits) >= 200:
                    break
            return hits
        return [str(p.relative_to(self.root)) for p in self.root.glob(pattern) if p.is_file()]

    def grep(self, pattern: str, glob: str = "**/*") -> list[str]:
        rx = grep_regex(pattern)
        if rx is None:
            return []
        hits: list[str] = []
        for rel in self.glob(glob):
            try:
                text = self.read(rel)
            except (OSError, UnicodeDecodeError, PermissionError):
                continue
            for i, line in enumerate(text.splitlines(), 1):
                if rx.search(line):
                    hits.append(f"{rel}:{i}:{line[:200]}")
                    if len(hits) >= 50:
                        return hits
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
        from orbweaver.sandbox.bwrap import SandboxUnavailable, is_containerized, run_sandboxed, spawn_sandboxed

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
