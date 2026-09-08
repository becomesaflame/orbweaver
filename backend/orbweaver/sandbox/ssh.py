"""Sandboxed OpenSSH: clean config, CONNECT ProxyCommand, host identity files."""

from __future__ import annotations

import os
import stat
from pathlib import Path

PROXY_PORT = 19191

SSH_IDENTITY_FILES: tuple[str, ...] = (
    "id_ed25519",
    "id_ed25519.pub",
    "id_rsa",
    "id_rsa.pub",
    "id_ecdsa",
    "id_ecdsa.pub",
    "id_ed25519_sk",
    "id_ed25519_sk.pub",
    "id_ecdsa_sk",
    "id_ecdsa_sk.pub",
    "known_hosts",
)

SYSTEM_SSH_CONFIG = (
    "# Orbweaver sandbox system ssh_config.\n"
    "# Replaces the host file so OpenSSH does not Include ssh_config.d drop-ins\n"
    "# that fail the user-namespace owner check.\n"
)

PROXYCMD_SOURCE = r'''#!/usr/bin/env python3
"""HTTP CONNECT tunnel for sandboxed ssh (stdin/stdout <-> proxy)."""
from __future__ import annotations

import os
import select
import socket
import sys


def main() -> int:
    if len(sys.argv) < 3:
        sys.stderr.write("usage: proxycmd host port\n")
        return 2
    host, port_s = sys.argv[1], sys.argv[2]
    try:
        port = int(port_s)
    except ValueError:
        sys.stderr.write(f"orbweaver-ssh: bad port {port_s!r}\n")
        return 2
    proxy_host = os.environ.get("ORBWEAVER_SSH_PROXY_HOST", "127.0.0.1")
    proxy_port = int(os.environ.get("ORBWEAVER_SSH_PROXY_PORT", "19191"))
    try:
        sock = socket.create_connection((proxy_host, proxy_port), 20)
    except OSError as e:
        sys.stderr.write(f"orbweaver-ssh: proxy connect failed: {e}\n")
        return 1
    req = f"CONNECT {host}:{port} HTTP/1.1\r\nHost: {host}:{port}\r\n\r\n"
    sock.sendall(req.encode("ascii"))
    buf = b""
    while b"\r\n\r\n" not in buf:
        chunk = sock.recv(4096)
        if not chunk:
            sys.stderr.write("orbweaver-ssh: proxy closed before CONNECT response\n")
            return 1
        buf += chunk
        if len(buf) > 65_536:
            sys.stderr.write("orbweaver-ssh: oversized CONNECT response\n")
            return 1
    header, rest = buf.split(b"\r\n\r\n", 1)
    status_line = header.split(b"\r\n", 1)[0].decode("iso-8859-1", "replace")
    parts = status_line.split()
    if len(parts) < 2 or parts[1] != "200":
        sys.stderr.write(f"orbweaver-ssh: CONNECT {host}:{port} failed: {status_line}\n")
        if rest:
            sys.stderr.buffer.write(rest)
        return 1
    if rest:
        os.write(1, rest)
    stdin_fd, stdout_fd, sock_fd = 0, 1, sock.fileno()
    stdin_open = True
    sock_open = True
    while stdin_open or sock_open:
        fds = []
        if stdin_open:
            fds.append(stdin_fd)
        if sock_open:
            fds.append(sock_fd)
        readable, _, _ = select.select(fds, [], [], 60)
        if not readable:
            return 0
        if stdin_open and stdin_fd in readable:
            data = os.read(stdin_fd, 65536)
            if not data:
                stdin_open = False
                try:
                    sock.shutdown(socket.SHUT_WR)
                except OSError:
                    pass
            else:
                sock.sendall(data)
        if sock_open and sock_fd in readable:
            data = sock.recv(65536)
            if not data:
                sock_open = False
            else:
                os.write(stdout_fd, data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


def filter_ssh_argv(argv: list[str]) -> list[str]:
    """Drop -F so the sandbox config (ProxyCommand, identities) always applies."""
    out: list[str] = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "-F":
            i += 2
            continue
        if a.startswith("-F") and len(a) > 2:
            i += 1
            continue
        out.append(a)
        i += 1
    return out


def _wrapper_source(real: Path, config: Path) -> str:
    return (
        "#!/usr/bin/env python3\n"
        "import os, sys\n"
        f"REAL = {str(real)!r}\n"
        f"CONFIG = {str(config)!r}\n"
        "\n"
        "def filter_ssh_argv(argv):\n"
        "    out = []\n"
        "    i = 0\n"
        "    while i < len(argv):\n"
        "        a = argv[i]\n"
        "        if a == '-F':\n"
        "            i += 2\n"
        "            continue\n"
        "        if a.startswith('-F') and len(a) > 2:\n"
        "            i += 1\n"
        "            continue\n"
        "        out.append(a)\n"
        "        i += 1\n"
        "    return out\n"
        "\n"
        "args = filter_ssh_argv(sys.argv[1:])\n"
        "os.execv(REAL, [REAL, '-F', CONFIG, *args])\n"
    )


def _user_ssh_config(*, proxied: bool, proxycmd: Path, known_hosts: Path) -> str:
    lines = [
        "Host *",
        "  AddressFamily inet",
        "  GlobalKnownHostsFile none",
        f"  UserKnownHostsFile {known_hosts}",
        "  StrictHostKeyChecking accept-new",
        "  IdentityFile ~/.ssh/id_ed25519",
        "  IdentityFile ~/.ssh/id_rsa",
        "  IdentityFile ~/.ssh/id_ecdsa",
        "  IdentityFile ~/.ssh/id_ed25519_sk",
        "  IdentityFile ~/.ssh/id_ecdsa_sk",
    ]
    if proxied:
        lines.append(f"  ProxyCommand python3 {proxycmd} %h %p")
    return "\n".join(lines) + "\n"


def ssh_helper_dir(tmp: Path) -> Path:
    return tmp.resolve() / "ow-ssh"


def ensure_ssh_sandbox(tmp: Path, *, proxied: bool) -> Path:
    """Write wrapper, configs, and CONNECT helper under tmp/ow-ssh."""
    dest = ssh_helper_dir(tmp)
    dest.mkdir(parents=True, exist_ok=True)
    real = dest / "openssh"
    config = dest / "config"
    system = dest / "ssh_config.system"
    proxycmd = dest / "proxycmd.py"
    wrapper = dest / "ssh"
    known = dest / "known_hosts"
    known.touch(exist_ok=True)
    system.write_text(SYSTEM_SSH_CONFIG, encoding="utf-8")
    proxycmd.write_text(PROXYCMD_SOURCE, encoding="utf-8")
    config.write_text(
        _user_ssh_config(proxied=proxied, proxycmd=proxycmd, known_hosts=known),
        encoding="utf-8",
    )
    wrapper.write_text(_wrapper_source(real, config), encoding="utf-8")
    mode = wrapper.stat().st_mode
    wrapper.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    proxycmd.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return dest


def ssh_config_overlay_args(tmp: Path) -> list[str]:
    """Replace host ssh_config and wrap /usr/bin/ssh. Call after ensure_ssh_sandbox."""
    dest = ssh_helper_dir(tmp)
    wrapper = dest / "ssh"
    system = dest / "ssh_config.system"
    real = dest / "openssh"
    args: list[str] = [
        "--ro-bind",
        str(system),
        "/etc/ssh/ssh_config",
        "--tmpfs",
        "/etc/ssh/ssh_config.d",
    ]
    host_ssh = Path("/usr/bin/ssh")
    if host_ssh.exists() and wrapper.is_file():
        args.extend(["--ro-bind", str(host_ssh), str(real)])
        args.extend(["--ro-bind", str(wrapper), "/usr/bin/ssh"])
        bin_ssh = Path("/bin/ssh")
        if bin_ssh.exists():
            args.extend(["--ro-bind", str(wrapper), "/bin/ssh"])
    return args


def ssh_identity_bind_args(home: Path | None = None) -> list[str]:
    """Re-expose keys after the ~/.ssh denyRead tmpfs. Skip host ssh_config."""
    home_dir = (home or Path.home()).resolve()
    ssh_dir = home_dir / ".ssh"
    args: list[str] = []
    for name in SSH_IDENTITY_FILES:
        src = ssh_dir / name
        args.extend(["--ro-bind-try", str(src), str(src)])
    return args


def ssh_agent_socket(environ: dict[str, str] | None = None) -> tuple[Path, ...]:
    env = environ if environ is not None else os.environ
    raw = (env.get("SSH_AUTH_SOCK") or "").strip()
    if not raw:
        return ()
    return (Path(raw),)
