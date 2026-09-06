"""Local and Docker workspaces."""

from __future__ import annotations

import subprocess
from pathlib import Path

from orbweaver.config import settings
from orbweaver.uris import resolve_workspace_uri, validate_workspace_uri


class LocalWorkspace:
    def __init__(self, workspace_uri: str, workspace_root: str) -> None:
        self.uri = validate_workspace_uri(workspace_uri)
        self.root = resolve_workspace_uri(self.uri, workspace_root)

    def _safe(self, rel: str) -> Path:
        path = (self.root / rel).resolve()
        if self.root not in path.parents and path != self.root:
            raise PermissionError(f"path escapes workspace: {rel}")
        return path

    def read(self, path: str) -> str:
        return self._safe(path).read_text(encoding="utf-8")

    def write(self, path: str, content: str) -> None:
        p = self._safe(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")

    def glob(self, pattern: str) -> list[str]:
        return [str(p.relative_to(self.root)) for p in self.root.glob(pattern) if p.is_file()]

    def grep(self, pattern: str, glob: str = "**/*") -> list[str]:
        hits: list[str] = []
        for rel in self.glob(glob):
            try:
                text = self.read(rel)
            except (OSError, UnicodeDecodeError):
                continue
            for i, line in enumerate(text.splitlines(), 1):
                if pattern in line:
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
    ) -> str:
        if unsandboxed or not sandbox or not settings.orbweaver_sandbox:
            return self._raw_bash(command, timeout)
        from orbweaver.sandbox.bwrap import SandboxUnavailable, is_containerized, run_sandboxed

        if is_containerized():
            return self._raw_bash(command, timeout)
        try:
            return run_sandboxed(command, self.root, timeout)
        except SandboxUnavailable as e:
            if settings.orbweaver_sandbox_fail_if_unavailable:
                return f"sandbox_unavailable: {e}"
            return self._raw_bash(command, timeout)

    def propose_patch(self, path: str, old: str, new: str) -> dict:
        current = self.read(path) if self._safe(path).exists() else ""
        if old and old not in current:
            return {"ok": False, "error": "old_string not found", "path": path, "current": current}
        updated = current.replace(old, new, 1) if old else new
        return {"ok": True, "path": path, "old": old, "new": new, "updated": updated}


class DockerWorkspace:
    """Untrusted sessions: run bash in a throwaway container with the tree mounted."""

    def __init__(self, workspace_uri: str, workspace_root: str, image: str = "python:3.12-slim") -> None:
        self.local = LocalWorkspace(workspace_uri, workspace_root)
        self.image = image

    def read(self, path: str) -> str:
        return self.local.read(path)

    def write(self, path: str, content: str) -> None:
        return self.local.write(path, content)

    def glob(self, pattern: str) -> list[str]:
        return self.local.glob(pattern)

    def grep(self, pattern: str, glob: str = "**/*") -> list[str]:
        return self.local.grep(pattern, glob)

    def propose_patch(self, path: str, old: str, new: str) -> dict:
        return self.local.propose_patch(path, old, new)

    def bash(
        self,
        command: str,
        timeout: int = 30,
        sandbox: bool = True,
        unsandboxed: bool = False,
    ) -> str:
        del sandbox, unsandboxed
        root = str(self.local.root)
        try:
            proc = subprocess.run(
                [
                    "docker",
                    "run",
                    "--rm",
                    "-v",
                    f"{root}:/work:rw",
                    "-w",
                    "/work",
                    self.image,
                    "bash",
                    "-lc",
                    command,
                ],
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except FileNotFoundError:
            return "docker is not available on this host; bash refused for DockerWorkspace"
        return ((proc.stdout or "") + (proc.stderr or ""))[-200_000:]


def make_workspace(kind: str, uri: str, workspace_root: str):
    if kind == "docker":
        return DockerWorkspace(uri, workspace_root)
    return LocalWorkspace(uri, workspace_root)
