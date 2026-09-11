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
COMMIT_OR_PUSH_RE = re.compile(r"\bgit\s+(commit|push)\b")
REBASE_FINISH_RE = re.compile(r"\bgit\s+rebase\s+(?:--continue|--abort|--skip)\b")

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
NOT_DONE_MARK = "NOT_DONE"
NOT_DONE_WARNING = (
    "NOT_DONE: this is not done. git commit / git push did not finish the rebase "
    "and did not publish HEAD. The next Git command must be `git rebase --continue` "
    "(after git add of resolved files) or `git rebase --abort`. "
    "Do not treat 'Everything up-to-date' or a new commit hash as success."
)
STAGED_TMP_WARNING = (
    ".orbweaver-tmp is in git status; do not stage it. "
    "Add only the files you changed."
)

# Config-execution vectors git reads from the repo's own .git/config. A hostile
# workspace can set these to run an arbitrary host command when the ritual runs
# `git status`/`rev-parse` (see issue #94). Command-line `-c` has the highest
# precedence, so these override whatever the repo config says.
_GIT_CONFIG_ARGS: tuple[str, ...] = (
    "-c",
    "core.fsmonitor=",
    "-c",
    "core.hooksPath=/dev/null",
    "-c",
    "core.sshCommand=",
    "-c",
    "core.pager=cat",
    "-c",
    "credential.helper=",
)

_GIT_ENV = {
    "GIT_CONFIG_GLOBAL": "/dev/null",
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_TERMINAL_PROMPT": "0",
    "GIT_OPTIONAL_LOCKS": "0",
}

# Only a minimal environment reaches git, not os.environ (the gateway user's
# secrets). #93 tracks the broader env-inheritance problem for sandboxed Bash.
_GIT_ENV_PASSTHROUGH = ("PATH", "HOME", "LANG")


def _git_env() -> dict[str, str]:
    env = {k: os.environ[k] for k in _GIT_ENV_PASSTHROUGH if k in os.environ}
    env.update(_GIT_ENV)
    return env


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
    return subprocess.run(
        ["git", *_GIT_CONFIG_ARGS, "-C", str(repo), *args],
        capture_output=True,
        text=True,
        timeout=8,
        env=_git_env(),
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


def _working_set_roots(root: Path) -> tuple[Path, ...]:
    """Working-set roots for the ritual: never follow `cd` outside them (#94).

    A hostile `cd /any/attacker/checkout && git log` must not make the host-side
    ritual honor that repo's config. Fall back to the workspace root alone if the
    sandbox policy cannot be loaded.
    """
    try:
        from orbweaver.sandbox.policy import load_sandbox_policy

        return load_sandbox_policy(root).working_set_roots(root)
    except Exception:
        return (root,)


def repo_for_command(workspace_root: Path, command: str) -> Path | None:
    root = Path(workspace_root).resolve()
    from orbweaver.sandbox.policy import in_roots

    roots = _working_set_roots(root)
    candidates: list[Path] = []
    for raw in cd_targets(command):
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = (root / path)
        try:
            resolved = path.resolve()
        except OSError:
            continue
        if not in_roots(resolved, roots):
            continue
        candidates.append(resolved)
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


def command_commits_or_pushes(command: str) -> bool:
    return bool(COMMIT_OR_PUSH_RE.search(command or ""))


def command_finishes_rebase(command: str) -> bool:
    return bool(REBASE_FINISH_RE.search(command or ""))


def ritual_says_not_done(output: str) -> bool:
    blob = str(output or "")
    return f"{NOT_DONE_MARK}:" in blob


def escalate_lines(
    command: str, *, rebase: bool, detached: bool, status: str
) -> list[str]:
    extra: list[str] = []
    if (
        (rebase or detached)
        and command_commits_or_pushes(command)
        and not command_finishes_rebase(command)
    ):
        extra.append(NOT_DONE_WARNING)
    if ADD_ALL_RE.search(command or ""):
        extra.append(ADD_ALL_WARNING)
        if ".orbweaver-tmp" in (status or ""):
            extra.append(STAGED_TMP_WARNING)
    return extra


def git_snapshot(repo: Path) -> tuple[str, bool, bool, str]:
    status = _run_git(repo, "status", "--short", "--branch")
    if status.returncode != 0 and not (status.stdout or status.stderr):
        return "", False, False, ""
    head_ab = (_run_git(repo, "rev-parse", "--abbrev-ref", "HEAD").stdout or "").strip()
    head_sha = (_run_git(repo, "rev-parse", "--short", "HEAD").stdout or "").strip()
    git_dir_raw = (_run_git(repo, "rev-parse", "--git-dir").stdout or "").strip()
    git_dir = Path(git_dir_raw) if git_dir_raw else repo / ".git"
    if not git_dir.is_absolute():
        git_dir = repo / git_dir

    status_text = (status.stdout or status.stderr or "").rstrip() or "(no status)"
    rebase = (git_dir / "rebase-merge").is_dir() or (git_dir / "rebase-apply").is_dir()
    detached = head_ab == "HEAD"
    lines = [
        f"repo={repo}",
        f"branch={head_ab or '?'}",
        f"HEAD={head_sha or '?'}",
        status_text,
    ]
    if rebase:
        lines.append(REBASE_WARNING)
    if detached:
        lines.append(DETACHED_WARNING)
    return "\n".join(lines), rebase, detached, status_text


def annotate_bash_output(workspace_root: Path, command: str, output: str) -> str:
    if not command_uses_git(command):
        return output
    repo = repo_for_command(workspace_root, command)
    if repo is None:
        return output
    try:
        snap, rebase, detached, status_text = git_snapshot(repo)
    except (OSError, subprocess.TimeoutExpired):
        return output
    if not snap:
        return output
    parts = [str(output or "").rstrip(), "", RITUAL_MARK, snap]
    parts.extend(
        escalate_lines(
            command, rebase=rebase, detached=detached, status=status_text
        )
    )
    return "\n".join(parts) + "\n"
