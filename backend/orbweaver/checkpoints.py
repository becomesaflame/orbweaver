"""Per-turn workspace checkpoints so rewind can restore files, not just chat.

A checkpoint is a git tree of the working directory (tracked + untracked,
`.gitignore` respected) built through a *private* index at
``.orbweaver/checkpoints/index`` so the user's real index is never touched.
The tree is wrapped in a parentless commit and kept alive with
``refs/orbweaver/checkpoints/<session>/<seq>`` so ``git gc`` does not drop it;
the ref's committer date drives age-based pruning.

Restore reads the checkpoint tree into the private index, writes back every
path that differs from the working tree, and deletes paths that exist now
but not in the checkpoint. Deletions are limited to paths the private index
tracks at either point, so ignored or otherwise unknown user files survive.

Outside a git repository there is no checkpoint (no fallback copy yet).
"""

from __future__ import annotations

import logging
import os
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from orbweaver.config import settings

log = logging.getLogger(__name__)

CHECKPOINT_DIR = Path(".orbweaver") / "checkpoints"
REF_PREFIX = "refs/orbweaver/checkpoints"
FILE_EDIT_TOOLS = frozenset({"Write", "StrReplace", "Delete", "NotebookEdit", "Bash"})
EXCLUDED_DIRS = (".orbweaver", ".orbweaver-tmp")
NOT_GIT_NOTE = (
    "workspace is not a git repository; file checkpoints are unavailable "
    "(no copy-based fallback yet), only the conversation was rewound"
)
_EXCLUDE_PATHSPECS = tuple(f":(glob)**/{name}/**" for name in EXCLUDED_DIRS)
_ADD_PATHSPECS = ("--", ".") + tuple(
    f":(exclude,glob)**/{name}/**" for name in EXCLUDED_DIRS
)

_GIT_FLAGS = (
    "-c",
    "core.fsmonitor=",
    "-c",
    "core.hooksPath=/dev/null",
    "-c",
    "core.sshCommand=",
    "-c",
    "core.pager=cat",
)
_GIT_ENV = {
    "GIT_CONFIG_GLOBAL": "/dev/null",
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_TERMINAL_PROMPT": "0",
    "GIT_OPTIONAL_LOCKS": "0",
    "GIT_AUTHOR_NAME": "orbweaver",
    "GIT_AUTHOR_EMAIL": "orbweaver@localhost",
    "GIT_COMMITTER_NAME": "orbweaver",
    "GIT_COMMITTER_EMAIL": "orbweaver@localhost",
    "LC_ALL": "C",
}
_GIT_TIMEOUT_S = 120
_INDEX_LOCK = threading.Lock()


class CheckpointError(RuntimeError):
    """A git step failed while taking or restoring a checkpoint."""


class CheckpointHeadMismatch(CheckpointError):
    """The workspace HEAD moved since the checkpoint; pass force to override."""


@dataclass
class Checkpoint:
    tree: str
    head: str | None
    repo: str

    def payload(self) -> dict[str, Any]:
        # Host paths stay out of the event log; restore re-resolves the repo.
        return {"tree": self.tree, "head": self.head}


@dataclass
class RestoreSummary:
    tree: str
    head: str | None
    restored: list[str] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)

    def payload(self) -> dict[str, Any]:
        return {
            "status": "restored",
            "tree": self.tree,
            "head": self.head,
            "restored": list(self.restored),
            "deleted": list(self.deleted),
        }


def _git_env() -> dict[str, str]:
    env = {
        "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
        "HOME": os.environ.get("HOME", "/nonexistent"),
    }
    env.update(_GIT_ENV)
    return env


def _run_git(
    repo: Path,
    *args: str,
    index: Path | None = None,
    stdin: bytes | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[bytes]:
    env = _git_env()
    if index is not None:
        env["GIT_INDEX_FILE"] = str(index)
    try:
        proc = subprocess.run(
            ["git", "-C", str(repo), *_GIT_FLAGS, *args],
            input=stdin,
            capture_output=True,
            timeout=_GIT_TIMEOUT_S,
            env=env,
            check=False,
        )
    except FileNotFoundError as e:
        raise CheckpointError("git is not installed") from e
    except subprocess.TimeoutExpired as e:
        raise CheckpointError(f"git {args[0]} timed out") from e
    if check and proc.returncode != 0:
        err = proc.stderr.decode("utf-8", "replace").strip() or f"exit {proc.returncode}"
        raise CheckpointError(f"git {args[0]} failed: {err}")
    return proc


def _out(proc: subprocess.CompletedProcess[bytes]) -> str:
    return proc.stdout.decode("utf-8", "replace").strip()


def git_toplevel(root: Path) -> Path | None:
    """Top of the git work tree containing ``root``; None outside git."""
    root = Path(root)
    if not root.is_dir():
        return None
    try:
        proc = _run_git(root, "rev-parse", "--show-toplevel", check=False)
    except CheckpointError:
        return None
    if proc.returncode != 0:
        return None
    top = _out(proc)
    return Path(top) if top else None


def workspace_root_of(workspace: Any) -> Path | None:
    root = getattr(workspace, "root", None)
    if root is None:
        return None
    return Path(str(root))


def _head_sha(repo: Path) -> str | None:
    proc = _run_git(repo, "rev-parse", "--verify", "-q", "HEAD^{commit}", check=False)
    if proc.returncode != 0:
        return None
    return _out(proc) or None


def _index_path(repo: Path) -> Path:
    index = repo / CHECKPOINT_DIR / "index"
    index.parent.mkdir(parents=True, exist_ok=True)
    return index


def _excluded(path: str) -> bool:
    return any(part in EXCLUDED_DIRS for part in path.split("/"))


def _fill_private_index(repo: Path, index: Path) -> None:
    """Private index = HEAD, minus our own dirs, plus the current work tree."""
    _run_git(repo, "read-tree", "--empty", index=index)
    if _head_sha(repo) is not None:
        _run_git(repo, "read-tree", "HEAD", index=index)
        _run_git(
            repo,
            "rm",
            "--cached",
            "-r",
            "-q",
            "-f",
            "--ignore-unmatch",
            "--",
            *_EXCLUDE_PATHSPECS,
            index=index,
        )
    _run_git(repo, "add", "-A", *_ADD_PATHSPECS, index=index)


def _write_tree(repo: Path, index: Path) -> str:
    return _out(_run_git(repo, "write-tree", index=index))


def snapshot_tree(repo: Path) -> Checkpoint:
    """Tree id for the current work tree of ``repo`` (a git toplevel)."""
    with _INDEX_LOCK:
        index = _index_path(repo)
        _fill_private_index(repo, index)
        tree = _write_tree(repo, index)
    return Checkpoint(tree=tree, head=_head_sha(repo), repo=str(repo))


def checkpoint_ref(session_id: Any, label: Any) -> str:
    return f"{REF_PREFIX}/{session_id}/{label}"


def pin_checkpoint(checkpoint: Checkpoint, session_id: Any, label: Any) -> str:
    """Wrap the tree in a parentless commit and point a ref at it."""
    repo = Path(checkpoint.repo)
    message = f"orbweaver checkpoint session={session_id} seq={label} head={checkpoint.head or '-'}"
    commit = _out(_run_git(repo, "commit-tree", checkpoint.tree, "-m", message))
    ref = checkpoint_ref(session_id, label)
    _run_git(repo, "update-ref", ref, commit)
    return ref


def take_checkpoint(workspace_root: Path | str) -> Checkpoint | None:
    """Snapshot the workspace; None when it is not inside a git repository.

    Exposed so a hook can call it before destructive tools as well as at the
    start of a user turn.
    """
    if not settings.orbweaver_checkpoints:
        return None
    top = git_toplevel(Path(workspace_root))
    if top is None:
        return None
    return snapshot_tree(top)


def prune_checkpoints(
    repo: Path | str,
    *,
    session_id: Any | None = None,
    max_age_days: float | None = None,
    now: float | None = None,
) -> list[str]:
    """Delete checkpoint refs for a session and/or older than ``max_age_days``."""
    repo = Path(repo)
    prefix = f"{REF_PREFIX}/{session_id}/" if session_id is not None else f"{REF_PREFIX}/"
    proc = _run_git(
        repo,
        "for-each-ref",
        "--format=%(refname)%00%(committerdate:unix)",
        prefix,
        check=False,
    )
    if proc.returncode != 0:
        return []
    cutoff = None
    if max_age_days is not None:
        cutoff = (now if now is not None else time.time()) - max_age_days * 86400
    victims: list[str] = []
    for line in _out(proc).splitlines():
        name, _, stamp = line.partition("\0")
        if not name:
            continue
        if cutoff is None:
            victims.append(name)
            continue
        try:
            when = float(stamp or 0)
        except ValueError:
            when = 0.0
        if when < cutoff:
            victims.append(name)
    for ref in victims:
        _run_git(repo, "update-ref", "-d", ref, check=False)
    return victims


def checkpoint_user_turn(workspace: Any) -> dict[str, Any] | None:
    """``checkpoint`` payload for a ``user`` event; None when disabled. Never raises."""
    if not settings.orbweaver_checkpoints:
        return None
    root = workspace_root_of(workspace)
    if root is None:
        return {"unavailable": "workspace has no local root"}
    try:
        cp = take_checkpoint(root)
    except CheckpointError as e:
        log.warning("checkpoint skipped for %s: %s", root, e)
        return {"unavailable": str(e)}
    if cp is None:
        return {"unavailable": NOT_GIT_NOTE}
    return cp.payload()


def pin_user_turn(workspace: Any, checkpoint: dict[str, Any], session_id: Any, seq: int) -> None:
    """Keep the turn's tree alive under a ref; prune stale refs. Never raises."""
    tree = checkpoint.get("tree")
    root = workspace_root_of(workspace)
    if not tree or root is None:
        return
    try:
        repo = git_toplevel(root)
        if repo is None:
            return
        cp = Checkpoint(tree=str(tree), head=checkpoint.get("head"), repo=str(repo))
        pin_checkpoint(cp, session_id, seq)
        prune_checkpoints(repo, max_age_days=settings.orbweaver_checkpoint_keep_days)
    except CheckpointError as e:
        log.warning("checkpoint ref not written for %s: %s", root, e)


def turn_edits_files(events: list[Any]) -> bool:
    """True when any event is a tool_call to a tool that can change files."""
    for ev in events:
        if getattr(ev, "kind", None) != "tool_call":
            continue
        name = str((getattr(ev, "payload", None) or {}).get("name") or "")
        if name in FILE_EDIT_TOOLS:
            return True
    return False


def find_checkpoint(events: list[Any]) -> dict[str, Any] | None:
    """The first user-event checkpoint among the events being discarded."""
    for ev in events:
        if getattr(ev, "kind", None) != "user":
            continue
        cp = (getattr(ev, "payload", None) or {}).get("checkpoint")
        if isinstance(cp, dict) and cp.get("tree"):
            return cp
    return None


def _diff_paths(repo: Path, old_tree: str, new_tree: str) -> tuple[list[str], list[str]]:
    """(paths to write from new_tree, paths present in old_tree but not new_tree)."""
    proc = _run_git(repo, "diff-tree", "-r", "-z", "--name-status", old_tree, new_tree)
    fields = proc.stdout.split(b"\0")
    write: list[str] = []
    delete: list[str] = []
    i = 0
    while i + 1 < len(fields):
        status = fields[i].decode("utf-8", "replace")
        path = fields[i + 1].decode("utf-8", "surrogateescape")
        i += 2
        if not path or _excluded(path) or ".." in path.split("/"):
            continue
        if status.startswith("D"):
            delete.append(path)
        else:
            write.append(path)
    return write, delete


def _remove_paths(repo: Path, paths: list[str]) -> list[str]:
    removed: list[str] = []
    root = repo.resolve()
    for rel in paths:
        target = root / rel
        try:
            if not target.is_symlink() and not target.is_file():
                continue
            target.unlink()
        except OSError as e:
            log.warning("checkpoint restore could not delete %s: %s", rel, e)
            continue
        removed.append(rel)
        parent = target.parent
        while parent != root:
            try:
                parent.rmdir()
            except OSError:
                break
            parent = parent.parent
    return removed


def restore_checkpoint(
    workspace_root: Path | str,
    checkpoint: dict[str, Any],
    *,
    force: bool = False,
) -> RestoreSummary:
    """Put the work tree back to ``checkpoint['tree']``.

    Raises CheckpointHeadMismatch when HEAD moved since the checkpoint and
    ``force`` is false; CheckpointError for missing trees or git failures.
    """
    tree = str(checkpoint.get("tree") or "")
    if not tree:
        raise CheckpointError("checkpoint has no tree")
    top = git_toplevel(Path(workspace_root))
    if top is None:
        raise CheckpointError(NOT_GIT_NOTE)
    repo = top
    exists = _run_git(repo, "cat-file", "-e", f"{tree}^{{tree}}", check=False)
    if exists.returncode != 0:
        raise CheckpointError(f"checkpoint tree {tree[:12]} is no longer in the repository")
    want_head = checkpoint.get("head")
    have_head = _head_sha(repo)
    if (want_head or None) != have_head and not force:
        raise CheckpointHeadMismatch(
            "workspace HEAD is "
            f"{(have_head or 'unborn')[:12]} but the checkpoint was taken at "
            f"{(str(want_head) if want_head else 'unborn')[:12]}; "
            "commits were made since. Pass force=true to restore files anyway."
        )
    with _INDEX_LOCK:
        index = _index_path(repo)
        _fill_private_index(repo, index)
        now_tree = _write_tree(repo, index)
        write, delete = _diff_paths(repo, now_tree, tree)
        _run_git(repo, "read-tree", tree, index=index)
        if write:
            payload = b"".join(p.encode("utf-8", "surrogateescape") + b"\0" for p in write)
            _run_git(repo, "checkout-index", "-f", "-z", "--stdin", index=index, stdin=payload)
        deleted = _remove_paths(repo, delete)
    return RestoreSummary(
        tree=tree,
        head=str(want_head) if want_head else None,
        restored=write,
        deleted=deleted,
    )
