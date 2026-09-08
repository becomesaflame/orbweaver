"""Session workspaces (LocalWorkspace plus bubblewrap Bash)."""

from __future__ import annotations

import glob as globmod
import shutil
import subprocess
from pathlib import Path

from orbweaver.config import settings
from orbweaver.sandbox.policy import in_roots, load_sandbox_policy
from orbweaver.tooltext import grep_regex
from orbweaver.uris import resolve_workspace_uri, validate_workspace_uri


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
        proc = subprocess.run(
            command,
            shell=True,
            cwd=self.root,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return ((proc.stdout or "") + (proc.stderr or ""))[-200_000:]

    def bash(
        self,
        command: str,
        timeout: int = 30,
        sandbox: bool = True,
        unsandboxed: bool = False,
        permissions: list[str] | tuple[str, ...] | None = None,
    ) -> str:
        perms = _bash_permissions(permissions, unsandboxed)
        if "all" in perms or not sandbox or not settings.orbweaver_sandbox:
            return self._raw_bash(command, timeout)
        from orbweaver.sandbox.bwrap import SandboxUnavailable, is_containerized, run_sandboxed

        if is_containerized():
            return self._raw_bash(command, timeout)
        try:
            return run_sandboxed(
                command,
                self.root,
                timeout,
                policy=self._policy(),
                full_network="full_network" in perms,
            )
        except SandboxUnavailable as e:
            if settings.orbweaver_sandbox_fail_if_unavailable:
                return f"sandbox_unavailable: {e}"
            return self._raw_bash(command, timeout)

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
