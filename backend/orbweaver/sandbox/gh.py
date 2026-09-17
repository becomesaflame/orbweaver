"""Host-side GitHub CLI proxy: credentials stay out of the sandbox.

``~/.config/gh`` is denyRead and ``GH_TOKEN`` is env-excluded (#93, #124), so a
sandboxed ``gh pr create`` always printed ``gh auth login`` even when the
gateway user was logged in. Telegram attached to a desk session hit the same
path: the turn runs in bubblewrap as the gateway user, not as the Cursor
process that already has ``gh``.

This is the SSH_AUTH_SOCK analogue. A thread in the gateway process runs the
real ``gh`` with the host environment (``GH_TOKEN`` / ``GITHUB_TOKEN`` /
``~/.config/gh``). The sandbox only sees a unix socket and a wrapper bound
over ``gh``. ``gh auth token`` / ``login`` / ``logout`` are refused so the
agent cannot print or drop the host credentials.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import stat
import struct
import subprocess
import threading
from pathlib import Path

SANDBOX_GH_DIR = "/tmp/ow-gh"
SANDBOX_GH_SOCK = "/tmp/ow-gh/sock"
MAX_FRAME = 2_000_000
MAX_STDIN = 1_000_000
GH_TIMEOUT_S = 120

_HEADER = struct.Struct(">I")

# Passed to the host gh process. Tokens stay here, never in the sandbox env.
_HOST_ENV_KEEP = (
    "PATH",
    "HOME",
    "USER",
    "LOGNAME",
    "LANG",
    "LANGUAGE",
    "LC_ALL",
    "LC_CTYPE",
    "TZ",
    "SSH_AUTH_SOCK",
    "GH_TOKEN",
    "GITHUB_TOKEN",
    "GH_HOST",
    "GH_ENTERPRISE_TOKEN",
    "GH_CONFIG_DIR",
    "XDG_CONFIG_HOME",
    "http_proxy",
    "https_proxy",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "NO_PROXY",
    "no_proxy",
)

WRAPPER_SOURCE = r'''#!/usr/bin/env python3
"""Forward `gh` argv to the host-side Orbweaver proxy. No credentials here."""
from __future__ import annotations

import json
import os
import socket
import struct
import sys

HEADER = struct.Struct(">I")
MAX_STDIN = 1_000_000
MAX_FRAME = 2_000_000


def _recv_all(sock: socket.socket, n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("gh proxy closed")
        buf += chunk
    return buf


def main() -> int:
    sock_path = os.environ.get("GH_PROXY_SOCK", "").strip()
    if not sock_path:
        sys.stderr.write(
            "gh: not available in this sandbox (no host GitHub CLI proxy)\n"
        )
        return 127
    stdin = ""
    if not sys.stdin.isatty():
        stdin = sys.stdin.read(MAX_STDIN + 1)
        if len(stdin) > MAX_STDIN:
            sys.stderr.write("gh: stdin too large\n")
            return 2
    payload = json.dumps(
        {"argv": sys.argv[1:], "cwd": os.getcwd(), "stdin": stdin}
    ).encode()
    if len(payload) > MAX_FRAME:
        sys.stderr.write("gh: request too large\n")
        return 2
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        sock.connect(sock_path)
        sock.sendall(HEADER.pack(len(payload)) + payload)
        raw_n = _recv_all(sock, HEADER.size)
        (n,) = HEADER.unpack(raw_n)
        if n > MAX_FRAME:
            sys.stderr.write("gh: oversized reply\n")
            return 1
        reply = json.loads(_recv_all(sock, n))
    except OSError as e:
        sys.stderr.write(f"gh: proxy connect failed: {e}\n")
        return 1
    finally:
        sock.close()
    if reply.get("stdout"):
        sys.stdout.write(str(reply["stdout"]))
    if reply.get("stderr"):
        sys.stderr.write(str(reply["stderr"]))
    try:
        return int(reply.get("returncode") or 0)
    except (TypeError, ValueError):
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
'''


def gh_binaries(environ: dict[str, str] | None = None) -> tuple[Path, ...]:
    """Host ``gh`` paths to overlay with the sandbox wrapper."""
    env = environ if environ is not None else os.environ
    found: list[Path] = []
    which = shutil.which("gh", path=env.get("PATH"))
    if which:
        found.append(Path(which).resolve())
    for raw in ("/usr/bin/gh", "/usr/local/bin/gh"):
        path = Path(raw)
        try:
            resolved = path.resolve() if path.exists() else None
        except OSError:
            resolved = None
        if resolved is not None and resolved not in found:
            found.append(resolved)
    return tuple(found)


def host_gh_env(environ: dict[str, str] | None = None) -> dict[str, str]:
    """Minimal environment for the host ``gh`` subprocess."""
    src = environ if environ is not None else os.environ
    out: dict[str, str] = {
        "GH_PROMPT_DISABLED": "1",
        "GH_NO_UPDATE_NOTIFIER": "1",
        "GIT_TERMINAL_PROMPT": "0",
        "PATH": src.get("PATH") or "/usr/local/bin:/usr/bin:/bin",
    }
    for name in _HOST_ENV_KEEP:
        val = src.get(name)
        if val:
            out[name] = val
    return out


def forbidden_gh_argv(argv: list[str]) -> str | None:
    """Refuse commands that would leak or drop host GitHub credentials."""
    positional = [a for a in argv if a and not a.startswith("-")]
    if not positional:
        return None
    cmd = positional[0]
    if cmd == "auth":
        sub = positional[1] if len(positional) > 1 else "status"
        if sub != "status":
            return (
                "gh auth login/token/logout are blocked in the sandbox; "
                "the host holds GitHub credentials. Use `gh auth status`, "
                "or git over SSH (git@github.com:...)."
            )
        return None
    if cmd in {"extension", "+"}:
        return "gh extensions are blocked in the sandbox"
    return None


def _recv_frame(conn: socket.socket) -> bytes:
    hdr = b""
    while len(hdr) < _HEADER.size:
        chunk = conn.recv(_HEADER.size - len(hdr))
        if not chunk:
            raise ConnectionError("closed")
        hdr += chunk
    (n,) = _HEADER.unpack(hdr)
    if n > MAX_FRAME:
        raise ValueError("oversized gh proxy frame")
    buf = b""
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("closed")
        buf += chunk
    return buf


def _send_frame(conn: socket.socket, payload: bytes) -> None:
    conn.sendall(_HEADER.pack(len(payload)) + payload)


def _reply(returncode: int, stdout: str = "", stderr: str = "") -> bytes:
    return json.dumps(
        {"returncode": returncode, "stdout": stdout, "stderr": stderr}
    ).encode()


def _cwd_allowed(cwd: str, roots: tuple[Path, ...]) -> Path | None:
    try:
        resolved = Path(cwd).resolve()
    except OSError:
        return None
    if not resolved.is_dir():
        return None
    for root in roots:
        try:
            resolved.relative_to(root.resolve())
            return resolved
        except ValueError:
            continue
    return None


def _handle(
    conn: socket.socket,
    *,
    gh_bin: Path,
    roots: tuple[Path, ...],
    environ: dict[str, str],
) -> None:
    try:
        raw = _recv_frame(conn)
        req = json.loads(raw)
        argv = [str(a) for a in (req.get("argv") or [])]
        reason = forbidden_gh_argv(argv)
        if reason:
            _send_frame(conn, _reply(2, stderr=reason + "\n"))
            return
        cwd = _cwd_allowed(str(req.get("cwd") or ""), roots)
        if cwd is None:
            _send_frame(
                conn,
                _reply(2, stderr="gh: working directory is outside the workspace\n"),
            )
            return
        stdin = str(req.get("stdin") or "")
        if len(stdin) > MAX_STDIN:
            _send_frame(conn, _reply(2, stderr="gh: stdin too large\n"))
            return
        try:
            completed = subprocess.run(
                [str(gh_bin), *argv],
                cwd=str(cwd),
                input=stdin,
                capture_output=True,
                text=True,
                timeout=GH_TIMEOUT_S,
                env=host_gh_env(environ),
                check=False,
            )
        except FileNotFoundError:
            _send_frame(conn, _reply(127, stderr="gh: GitHub CLI is not installed on the host\n"))
            return
        except subprocess.TimeoutExpired:
            _send_frame(conn, _reply(124, stderr="gh: timed out\n"))
            return
        _send_frame(
            conn,
            _reply(
                int(completed.returncode),
                stdout=completed.stdout or "",
                stderr=completed.stderr or "",
            ),
        )
    except (OSError, ValueError, json.JSONDecodeError, ConnectionError):
        try:
            _send_frame(conn, _reply(1, stderr="gh: proxy error\n"))
        except OSError:
            pass
    finally:
        try:
            conn.close()
        except OSError:
            pass


class GhProxy:
    """Unix-socket server that runs host ``gh`` for one sandbox session."""

    def __init__(
        self,
        sock_path: Path,
        *,
        workspace_root: Path,
        extra_roots: tuple[Path, ...] = (),
        gh_bin: Path | None = None,
        environ: dict[str, str] | None = None,
    ) -> None:
        self.sock_path = Path(sock_path)
        self.workspace_root = Path(workspace_root).resolve()
        self.extra_roots = tuple(Path(p).resolve() for p in extra_roots)
        self.environ = dict(environ) if environ is not None else dict(os.environ)
        which = gh_bin or (gh_binaries(self.environ)[0] if gh_binaries(self.environ) else None)
        self.gh_bin = which
        self._sock: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def start(self) -> Path | None:
        if self.gh_bin is None:
            return None
        if self.sock_path.exists():
            self.sock_path.unlink()
        self.sock_path.parent.mkdir(parents=True, exist_ok=True)
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.bind(str(self.sock_path))
        sock.listen(16)
        sock.settimeout(0.5)
        self._sock = sock
        self._thread = threading.Thread(target=self._serve, name="orbweaver-gh", daemon=True)
        self._thread.start()
        return self.sock_path

    def _roots(self) -> tuple[Path, ...]:
        return (self.workspace_root, *self.extra_roots)

    def _serve(self) -> None:
        assert self._sock is not None
        assert self.gh_bin is not None
        while not self._stop.is_set():
            try:
                conn, _ = self._sock.accept()
            except TimeoutError:
                continue
            except OSError:
                if self._stop.is_set():
                    return
                continue
            threading.Thread(
                target=_handle,
                kwargs={
                    "conn": conn,
                    "gh_bin": self.gh_bin,
                    "roots": self._roots(),
                    "environ": self.environ,
                },
                daemon=True,
            ).start()

    def close(self) -> None:
        self._stop.set()
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
        if self._thread is not None:
            self._thread.join(timeout=2)
        try:
            self.sock_path.unlink(missing_ok=True)
        except OSError:
            pass


def gh_helper_dir(tmp: Path) -> Path:
    return tmp.resolve() / "ow-gh"


def ensure_gh_sandbox(tmp: Path) -> Path:
    """Write the sandbox ``gh`` wrapper under ``tmp/ow-gh``."""
    dest = gh_helper_dir(tmp)
    dest.mkdir(parents=True, exist_ok=True)
    wrapper = dest / "gh"
    wrapper.write_text(WRAPPER_SOURCE, encoding="utf-8")
    wrapper.chmod(wrapper.stat().st_mode | stat.S_IEXEC)
    return dest


def gh_overlay_args(tmp: Path, sock: Path, environ: dict[str, str] | None = None) -> list[str]:
    """Bind the wrapper over host ``gh`` binaries. Call after ensure_gh_sandbox."""
    dest = gh_helper_dir(tmp)
    wrapper = dest / "gh"
    if not wrapper.is_file() or not sock:
        return []
    bins = gh_binaries(environ)
    if not bins:
        return []
    args: list[str] = [
        "--dir",
        SANDBOX_GH_DIR,
        "--ro-bind-try",
        str(sock),
        SANDBOX_GH_SOCK,
    ]
    seen: set[str] = set()
    for path in bins:
        dest_path = str(path)
        if dest_path in seen:
            continue
        seen.add(dest_path)
        args.extend(["--ro-bind", str(wrapper), dest_path])
        # Common argv[0] when PATH hits /bin/gh (merged /usr).
        bin_alias = Path("/bin") / path.name
        if bin_alias.exists() and str(bin_alias) not in seen:
            seen.add(str(bin_alias))
            args.extend(["--ro-bind", str(wrapper), str(bin_alias)])
    return args
