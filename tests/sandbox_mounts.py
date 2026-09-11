"""Catch bwrap overlay dests that cannot be created after --ro-bind / /."""

from __future__ import annotations

from pathlib import Path

_ZERO = {
    "--unshare-user",
    "--unshare-pid",
    "--unshare-net",
    "--die-with-parent",
    "--clearenv",
}
_ONE = {"--tmpfs", "--dir", "--dev", "--proc", "--chdir"}
_TWO = {"--ro-bind", "--ro-bind-try", "--bind", "--setenv"}


def _under(path: str, root: str) -> bool:
    p, r = Path(path), Path(root)
    try:
        p.relative_to(r)
    except ValueError:
        return False
    return True


def _writable(path: str, roots: list[str]) -> bool:
    return any(_under(path, root) for root in roots)


def assert_ro_bind_dests_creatable(argv: list[str]) -> None:
    """Fail if a non-try --ro-bind dest is a new file on a still-read-only mount.

    Production 0.6.0 stashed /usr/bin/ssh onto workspace .orbweaver-tmp/ow-ssh/openssh
    before the workspace was --bind read-write. bwrap then died with
    "Can't create file ... Read-only file system" on every Bash command.
    """
    writable: list[str] = []
    i = 1
    while i < len(argv):
        flag = argv[i]
        if flag == "--":
            break
        if flag in _ZERO:
            i += 1
            continue
        if flag in _ONE:
            dest = argv[i + 1]
            if flag in {"--tmpfs", "--dev", "--proc"} or flag == "--dir" and _writable(str(Path(dest).parent), writable):
                writable.append(dest)
            i += 2
            continue
        if flag in _TWO:
            dest = argv[i + 2]
            if flag == "--bind":
                writable.append(dest)
            elif flag == "--ro-bind":
                exists = Path(dest).exists()
                if not exists and not _writable(dest, writable):
                    raise AssertionError(
                        f"--ro-bind dest {dest!r} does not exist and is not under a writable "
                        f"mount {writable}. bwrap will fail: Can't create file ... "
                        f"Read-only file system"
                    )
            i += 3
            continue
        i += 1
