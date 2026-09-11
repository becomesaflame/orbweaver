"""Sandboxed OpenSSH: clean config, CONNECT ProxyCommand, host identity files."""

from __future__ import annotations

import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path

PROXY_PORT = 19191
# Stash the real ssh binary here — sandbox /tmp is tmpfs (writable) after --ro-bind / /.
# Do not use workspace .orbweaver-tmp: that path is still read-only when overlays run.
SANDBOX_SSH_DIR = "/tmp/ow-ssh"
SANDBOX_OPENSSH = "/tmp/ow-ssh/openssh"

# Standard private identity names OpenSSH tries by default. Bound only with bindIdentities.
SSH_DEFAULT_PRIVATE_KEYS: tuple[str, ...] = (
    "id_ed25519",
    "id_rsa",
    "id_ecdsa",
    "id_ed25519_sk",
    "id_ecdsa_sk",
)

# Non-secret ~/.ssh files re-exposed after the denyRead tmpfs (plus every ``*.pub``).
SSH_PUBLIC_NAMES: frozenset[str] = frozenset(
    {
        "config",
        "known_hosts",
        "known_hosts.old",
        "authorized_keys",
        "authorized_keys2",
    }
)


@dataclass(frozen=True)
class SshPolicy:
    """How private keys reach sandboxed ssh.

    Default: they do not. ``SSH_AUTH_SOCK`` is allowlisted by the sandbox policy, so an
    ssh-agent on the host signs without the key entering the sandbox. ``identities`` binds
    exactly those files read-only; ``bind_identities`` binds the host config's
    ``IdentityFile`` entries and the standard ``id_*`` names. Never "every file in ~/.ssh".
    """

    identities: tuple[Path, ...] = ()
    bind_identities: bool = False

    def binds_private_keys(self) -> bool:
        return self.bind_identities or bool(self.identities)

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


def ensure_ssh_sandbox(
    tmp: Path,
    *,
    proxied: bool,
    identity_files: tuple[Path, ...] | None = None,
) -> Path:
    """Write wrapper, configs, resolv.conf, and CONNECT helper under tmp/ow-ssh.

    ``identity_files`` are the private keys the sandbox will bind (see ``SshPolicy``);
    ``None`` means the default policy, which binds nothing.
    """
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
            identity_files=(
                identity_files if identity_files is not None else ssh_private_identity_files()
            ),
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
    """IdentityFile paths named by the host ~/.ssh/config (bound only with bindIdentities)."""
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


def _ssh_dir(home: Path | None) -> Path:
    return (home or Path.home()).resolve() / ".ssh"


def ssh_public_files(home: Path | None = None) -> tuple[Path, ...]:
    """Non-secret ~/.ssh files: config, known_hosts, authorized_keys, and ``*.pub``."""
    ssh_dir = _ssh_dir(home)
    found: list[Path] = []
    try:
        entries = sorted(ssh_dir.iterdir())
    except OSError:
        return ()
    for entry in entries:
        try:
            if not entry.is_file():
                continue
        except OSError:
            continue
        if entry.name in SSH_PUBLIC_NAMES or entry.name.endswith(".pub"):
            found.append(entry)
    return tuple(found)


def ssh_private_identity_files(
    home: Path | None = None,
    *,
    ssh_policy: SshPolicy | None = None,
) -> tuple[Path, ...]:
    """Private keys the policy lets into the sandbox. Empty for the default policy."""
    pol = ssh_policy or SshPolicy()
    ssh_dir = _ssh_dir(home)
    found: list[Path] = []
    seen: set[Path] = set()

    def add(path: Path) -> None:
        try:
            resolved = path.expanduser()
            resolved = resolved.resolve() if resolved.is_absolute() else (ssh_dir / resolved).resolve()
            if resolved in seen or not resolved.is_file():
                return
        except OSError:
            return
        if resolved.name.endswith(".pub") or resolved.name in SSH_PUBLIC_NAMES:
            return
        seen.add(resolved)
        found.append(resolved)

    for path in pol.identities:
        add(path)
    if pol.bind_identities:
        for path in _identity_files_from_host_config(ssh_dir):
            add(path)
        for name in SSH_DEFAULT_PRIVATE_KEYS:
            add(ssh_dir / name)
    return tuple(found)


def ssh_identity_files(
    home: Path | None = None,
    *,
    ssh_policy: SshPolicy | None = None,
) -> tuple[Path, ...]:
    """Everything re-bound under ~/.ssh: public files plus policy-named private keys."""
    out: list[Path] = []
    seen: set[Path] = set()
    for path in (*ssh_public_files(home), *ssh_private_identity_files(home, ssh_policy=ssh_policy)):
        if path in seen:
            continue
        seen.add(path)
        out.append(path)
    return tuple(out)


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


def ssh_identity_bind_args(
    home: Path | None = None,
    *,
    ssh_policy: SshPolicy | None = None,
) -> list[str]:
    """Re-expose ~/.ssh public files (and only policy-named private keys) after the tmpfs.

    Binds land on the tmpfs that hides ``~/.ssh`` so a private key with any other name
    stays absent inside the sandbox. Symlinked entries bind their resolved target.
    """
    args: list[str] = []
    seen: set[str] = set()
    for src in ssh_identity_files(home, ssh_policy=ssh_policy):
        dest = str(src)
        if dest in seen:
            continue
        seen.add(dest)
        try:
            real = str(src.resolve())
        except OSError:
            continue
        args.extend(["--ro-bind-try", real, dest])
    return args


def ssh_agent_socket(environ: dict[str, str] | None = None) -> tuple[Path, ...]:
    env = environ if environ is not None else os.environ
    raw = (env.get("SSH_AUTH_SOCK") or "").strip()
    if not raw:
        return ()
    return (Path(raw),)
