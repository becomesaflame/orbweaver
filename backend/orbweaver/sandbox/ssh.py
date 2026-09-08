"""Sandboxed OpenSSH: clean config, CONNECT ProxyCommand, host identity files."""

from __future__ import annotations

import os
import re
import stat
from pathlib import Path

PROXY_PORT = 19191
# Stash the real ssh binary here — sandbox /tmp is tmpfs (writable) after --ro-bind / /.
# Do not use workspace .orbweaver-tmp: that path is still read-only when overlays run.
SANDBOX_SSH_DIR = "/tmp/ow-ssh"
SANDBOX_OPENSSH = "/tmp/ow-ssh/openssh"

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

SSH_SKIP_NAMES: frozenset[str] = frozenset(
    {
        "config",
        "authorized_keys",
        "authorized_keys2",
        "known_hosts.old",
    }
)

_IDENTITYFILE_RE = re.compile(r"^\s*IdentityFile\s+(\S+)", re.IGNORECASE)
_LOOPBACK_NS_PREFIX = "127."
_LOOPBACK_NS = frozenset({"127.0.0.1", "127.0.0.53", "::1"})

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


def _user_ssh_config(
    *,
    proxied: bool,
    proxycmd: Path,
    known_hosts: Path,
    identity_files: tuple[Path, ...] = (),
) -> str:
    lines = [
        "Host *",
        "  AddressFamily inet",
        "  GlobalKnownHostsFile none",
        f"  UserKnownHostsFile {known_hosts}",
        "  StrictHostKeyChecking accept-new",
    ]
    priv = [p for p in identity_files if p.name != "known_hosts" and not p.name.endswith(".pub")]
    if priv:
        for path in priv:
            lines.append(f"  IdentityFile {path}")
    else:
        lines.extend(
            [
                "  IdentityFile ~/.ssh/id_ed25519",
                "  IdentityFile ~/.ssh/id_rsa",
                "  IdentityFile ~/.ssh/id_ecdsa",
                "  IdentityFile ~/.ssh/id_ed25519_sk",
                "  IdentityFile ~/.ssh/id_ecdsa_sk",
            ]
        )
    if proxied:
        lines.append(f"  ProxyCommand python3 {proxycmd} %h %p")
    return "\n".join(lines) + "\n"


def ssh_helper_dir(tmp: Path) -> Path:
    return tmp.resolve() / "ow-ssh"


def ensure_ssh_sandbox(tmp: Path, *, proxied: bool) -> Path:
    """Write wrapper, configs, resolv.conf, and CONNECT helper under tmp/ow-ssh."""
    dest = ssh_helper_dir(tmp)
    dest.mkdir(parents=True, exist_ok=True)
    real = Path(SANDBOX_OPENSSH)
    config = dest / "config"
    system = dest / "ssh_config.system"
    proxycmd = dest / "proxycmd.py"
    wrapper = dest / "ssh"
    known = dest / "known_hosts"
    known.touch(exist_ok=True)
    system.write_text(SYSTEM_SSH_CONFIG, encoding="utf-8")
    proxycmd.write_text(PROXYCMD_SOURCE, encoding="utf-8")
    (dest / "resolv.conf").write_text(sandbox_resolv_conf_text(), encoding="utf-8")
    config.write_text(
        _user_ssh_config(
            proxied=proxied,
            proxycmd=proxycmd,
            known_hosts=known,
            identity_files=ssh_private_identity_files(),
        ),
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
    args: list[str] = [
        "--ro-bind",
        str(system),
        "/etc/ssh/ssh_config",
        "--tmpfs",
        "/etc/ssh/ssh_config.d",
    ]
    host_ssh = Path("/usr/bin/ssh")
    if host_ssh.exists() and wrapper.is_file():
        args.extend(
            [
                "--dir",
                SANDBOX_SSH_DIR,
                "--ro-bind",
                str(host_ssh),
                SANDBOX_OPENSSH,
                "--ro-bind",
                str(wrapper),
                "/usr/bin/ssh",
            ]
        )
        bin_ssh = Path("/bin/ssh")
        if bin_ssh.exists():
            args.extend(["--ro-bind", str(wrapper), "/bin/ssh"])
    return args


def _expand_identity_path(raw: str, ssh_dir: Path) -> Path:
    if raw.startswith("~/"):
        return (ssh_dir.parent / raw[2:]).resolve()
    path = Path(raw)
    if not path.is_absolute():
        return (ssh_dir / path).resolve()
    return path.expanduser().resolve()


def _identity_files_from_host_config(ssh_dir: Path) -> tuple[Path, ...]:
    """Read IdentityFile paths from host config without exposing the file itself."""
    config = ssh_dir / "config"
    out: list[Path] = []
    try:
        text = config.read_text(encoding="utf-8")
    except OSError:
        return ()
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        match = _IDENTITYFILE_RE.match(line)
        if not match:
            continue
        raw = match.group(1).strip().strip('"').strip("'")
        try:
            path = _expand_identity_path(raw, ssh_dir)
            if path.is_file():
                out.append(path)
        except OSError:
            continue
    return tuple(out)


def ssh_identity_files(home: Path | None = None) -> tuple[Path, ...]:
    """Identity files to re-bind. Includes custom names and host-config IdentityFile."""
    home_dir = (home or Path.home()).resolve()
    ssh_dir = home_dir / ".ssh"
    found: list[Path] = []
    seen: set[Path] = set()

    def add(path: Path, *, require_file: bool) -> None:
        try:
            resolved = path.expanduser()
            resolved = resolved.resolve() if resolved.is_absolute() else (ssh_dir / resolved).resolve()
        except OSError:
            return
        if resolved in seen:
            return
        if require_file and not resolved.is_file():
            return
        seen.add(resolved)
        found.append(resolved)

    for name in SSH_IDENTITY_FILES:
        add(ssh_dir / name, require_file=False)
    try:
        for entry in ssh_dir.iterdir():
            if not entry.is_file() or entry.name in SSH_SKIP_NAMES or entry.name.endswith(".old"):
                continue
            add(entry, require_file=True)
    except OSError:
        pass
    for path in _identity_files_from_host_config(ssh_dir):
        add(path, require_file=True)
    return tuple(found)


def ssh_private_identity_files(home: Path | None = None) -> tuple[Path, ...]:
    return tuple(
        p
        for p in ssh_identity_files(home)
        if p.is_file() and p.name != "known_hosts" and not p.name.endswith(".pub")
    )


def sandbox_resolv_conf_text(raw: str | None = None) -> str:
    """resolv.conf that still works after the sandbox hides /run (systemd-resolved stub)."""
    text = raw if raw is not None else _read_host_resolv()
    v4: list[str] = []
    v6: list[str] = []
    extra: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("nameserver"):
            parts = stripped.split()
            if len(parts) < 2:
                continue
            ns = parts[1]
            if ns in _LOOPBACK_NS or ns.startswith(_LOOPBACK_NS_PREFIX):
                continue
            if ":" in ns:
                v6.append(f"nameserver {ns}")
            else:
                v4.append(f"nameserver {ns}")
        elif stripped.startswith(("search ", "domain ", "options ")):
            extra.append(stripped)
    nameservers = v4 + v6
    if not nameservers:
        nameservers = ["nameserver 1.1.1.1", "nameserver 8.8.8.8"]
    return "\n".join(nameservers + extra) + "\n"


def _read_host_resolv() -> str:
    for src in (Path("/run/systemd/resolve/resolv.conf"), Path("/etc/resolv.conf")):
        try:
            if src.exists():
                return src.read_text(encoding="utf-8")
        except OSError:
            continue
    return ""


def _parent_dirs(path: Path) -> list[str]:
    parts = [p for p in path.as_posix().split("/") if p]
    acc: list[str] = []
    out: list[str] = []
    for part in parts[:-1]:
        acc.append(part)
        out.append("/" + "/".join(acc))
    return out


def resolv_conf_overlay_args(tmp: Path) -> list[str]:
    """Put uplink DNS in place after --tmpfs /run (host resolv.conf is a symlink into /run)."""
    src = ssh_helper_dir(tmp) / "resolv.conf"
    if not src.is_file():
        return []
    host = Path("/etc/resolv.conf")
    args: list[str] = []
    try:
        if host.is_symlink():
            target = Path(os.path.normpath(str(Path("/etc") / host.readlink())))
            if str(target).startswith("/run/") or str(target).startswith("/var/run/"):
                for directory in _parent_dirs(target):
                    if directory in {"/", "/run", "/var", "/var/run"}:
                        continue
                    args.extend(["--dir", directory])
                args.extend(["--ro-bind", str(src), str(target)])
                return args
    except OSError:
        pass
    args.extend(["--ro-bind", str(src), "/etc/resolv.conf"])
    return args


def ssh_identity_bind_args(home: Path | None = None) -> list[str]:
    """Re-expose keys after the ~/.ssh denyRead tmpfs. Skip host ssh_config."""
    args: list[str] = []
    seen: set[str] = set()
    for src in ssh_identity_files(home):
        dest = str(src)
        if dest in seen:
            continue
        seen.add(dest)
        args.extend(["--ro-bind-try", dest, dest])
    return args


def ssh_agent_socket(environ: dict[str, str] | None = None) -> tuple[Path, ...]:
    env = environ if environ is not None else os.environ
    raw = (env.get("SSH_AUTH_SOCK") or "").strip()
    if not raw:
        return ()
    return (Path(raw),)
