"""Deterministic Git state appended after Bash that invoked git.

The model must not treat a success substring (e.g. 'Everything up-to-date')
as proof a rebase finished or a branch moved. This footer is independent of
the LLM: it runs git in the repo the command used.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

GIT_CMD_RE = re.compile(r"(?:^|[\s;|&`])git\s+\w")
CD_RE = re.compile(r"(?:^|&&|\|\||;)\s*cd\s+((?:'[^']+'|\"[^\"]+\"|\S+))")
ADD_ALL_RE = re.compile(r"\bgit\s+add\s+(?:-A|--all)\b")

RITUAL_MARK = "--- git ritual ---"
REBASE_WARNING = (
    "REBASE_IN_PROGRESS: the rebase is not finished. "
    "After resolving conflicts, git add the files and run git rebase --continue "
    "(or git rebase --abort). git commit is not continue. "
    "While HEAD is detached, git push origin <branch> updates the old branch tip, "
    "not this commit. Do not treat 'Everything up-to-date' as a successful push of HEAD."
)
DETACHED_WARNING = (
    "DETACHED_HEAD: git push origin <branch-name> does not publish this commit. "
    "Finish the rebase/merge or check out the branch and move the ref first."
)
ADD_ALL_WARNING = (
    "git add -A/--all stages untracked junk (.venv, .orbweaver-tmp). "
    "Add the files you changed instead."
)

_GIT_ENV = {
    "GIT_OPTIONAL_LOCKS": "1",
    "GIT_TERMINAL_PROMPT": "0",
}


def command_uses_git(command: str) -> bool:
    return bool(GIT_CMD_RE.search(command or ""))


def _unquote(token: str) -> str:
    token = token.strip()
    if len(token) >= 2 and token[0] == token[-1] and token[0] in {'"', "'"}:
        return token[1:-1]
    return token


def cd_targets(command: str) -> list[str]:
    return [_unquote(m.group(1)) for m in CD_RE.finditer(command or "")]


def _run_git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, **_GIT_ENV}
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        timeout=8,
        env=env,
        check=False,
    )


def git_toplevel(cwd: Path) -> Path | None:
    if not cwd.is_dir():
        return None
    proc = _run_git(cwd, "rev-parse", "--show-toplevel")
    if proc.returncode != 0:
        return None
    top = (proc.stdout or "").strip()
    return Path(top) if top else None


def repo_for_command(workspace_root: Path, command: str) -> Path | None:
    root = Path(workspace_root).resolve()
    candidates: list[Path] = []
    for raw in cd_targets(command):
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = (root / path)
        try:
            candidates.append(path.resolve())
        except OSError:
            continue
    candidates.append(root)
    seen: set[Path] = set()
    for cwd in candidates:
        if cwd in seen:
            continue
        seen.add(cwd)
        top = git_toplevel(cwd)
        if top is not None:
            return top
    return None


def git_snapshot(repo: Path) -> str:
    status = _run_git(repo, "status", "--short", "--branch")
    if status.returncode != 0 and not (status.stdout or status.stderr):
        return ""
    head_ab = (_run_git(repo, "rev-parse", "--abbrev-ref", "HEAD").stdout or "").strip()
    head_sha = (_run_git(repo, "rev-parse", "--short", "HEAD").stdout or "").strip()
    git_dir_raw = (_run_git(repo, "rev-parse", "--git-dir").stdout or "").strip()
    git_dir = Path(git_dir_raw) if git_dir_raw else repo / ".git"
    if not git_dir.is_absolute():
        git_dir = repo / git_dir

    lines = [
        f"repo={repo}",
        f"branch={head_ab or '?'}",
        f"HEAD={head_sha or '?'}",
        (status.stdout or status.stderr or "").rstrip() or "(no status)",
    ]
    if (git_dir / "rebase-merge").is_dir() or (git_dir / "rebase-apply").is_dir():
        lines.append(REBASE_WARNING)
    if head_ab == "HEAD":
        lines.append(DETACHED_WARNING)
    return "\n".join(lines)


def annotate_bash_output(workspace_root: Path, command: str, output: str) -> str:
    if not command_uses_git(command):
        return output
    repo = repo_for_command(workspace_root, command)
    if repo is None:
        return output
    try:
        snap = git_snapshot(repo)
    except (OSError, subprocess.TimeoutExpired):
        return output
    if not snap:
        return output
    parts = [str(output or "").rstrip(), "", RITUAL_MARK, snap]
    if ADD_ALL_RE.search(command or ""):
        parts.append(ADD_ALL_WARNING)
    return "\n".join(parts) + "\n"
