"""CONNECT/HTTP proxy with domain allowlist. No TLS interception."""

from __future__ import annotations

import ipaddress
import select
import socket
import threading
from pathlib import Path

from orbweaver.sandbox.errors import network_denied
from orbweaver.sandbox.policy import NetworkPolicy
from orbweaver.sandbox.ssh import PROXY_PORT

_METADATA_V4 = ipaddress.ip_address("169.254.169.254")


def host_matches(hostname: str, pattern: str) -> bool:
    host = (hostname or "").lower().split("%")[0].rstrip(".")
    pat = (pattern or "").lower().rstrip(".")
    if not host or not pat:
        return False
    if pat == "*":
        return True
    if pat.startswith("*."):
        base = pat[2:]
        return host == base or host.endswith("." + base)
    return host == pat


def domain_allowed(hostname: str, network: NetworkPolicy) -> bool:
    host = (hostname or "").split(":")[0].strip().lower().rstrip(".")
    if not host:
        return False
    for pat in network.deny:
        if host_matches(host, pat):
            return False
    for pat in network.allowed_hosts():
        if host_matches(host, pat):
            return True
    return network.default == "allow"


def ip_is_blocked(addr: str) -> bool:
    try:
        ip: ipaddress.IPv4Address | ipaddress.IPv6Address = ipaddress.ip_address(
            addr.split("%")[0].strip("[]")
        )
    except ValueError:
        return True
    mapped = getattr(ip, "ipv4_mapped", None)
    if mapped is not None:
        # ::ffff:127.0.0.1 must be judged as 127.0.0.1 (3.12 does not do this itself).
        ip = mapped
    if ip == _METADATA_V4:
        return True
    return bool(
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
        or not ip.is_global
    )


def resolve_unblocked(hostname: str, port: int) -> list[tuple[socket.AddressFamily, tuple]]:
    infos = socket.getaddrinfo(hostname, port, type=socket.SOCK_STREAM)
    ok: list[tuple[socket.AddressFamily, tuple]] = []
    for fam, _kind, _proto, _canon, sockaddr in infos:
        ip = str(sockaddr[0])
        if ip_is_blocked(ip):
            continue
        ok.append((fam, sockaddr))
    return ok


def _read_headers(conn: socket.socket) -> bytes:
    buf = b""
    while b"\r\n\r\n" not in buf and b"\n\n" not in buf:
        chunk = conn.recv(4096)
        if not chunk:
            break
        buf += chunk
        if len(buf) > 64_000:
            break
    return buf


def _header_host(header_blob: bytes) -> str:
    text = header_blob.decode("iso-8859-1", errors="replace")
    first = text.splitlines()[0] if text.splitlines() else ""
    parts = first.split()
    if len(parts) >= 2 and parts[0].upper() == "CONNECT":
        return parts[1]
    for line in text.splitlines():
        if line.lower().startswith("host:"):
            return line.split(":", 1)[1].strip()
    if len(parts) >= 2:
        return parts[1]
    return ""


def _split_host_port(target: str, default_port: int) -> tuple[str, int]:
    target = target.strip()
    if target.startswith("[") and "]" in target:
        host, rest = target[1:].split("]", 1)
        port = default_port
        if rest.startswith(":"):
            port = int(rest[1:] or default_port)
        return host, port
    if target.count(":") == 1:
        host, port_s = target.split(":")
        return host, int(port_s or default_port)
    return target, default_port


def _splice(a: socket.socket, b: socket.socket) -> None:
    sockets = [a, b]
    try:
        while True:
            readable, _, _ = select.select(sockets, [], [], 60)
            if not readable:
                break
            for src in readable:
                dest = b if src is a else a
                data = src.recv(65536)
                if not data:
                    return
                dest.sendall(data)
    except OSError:
        return


def _handle(conn: socket.socket, network: NetworkPolicy) -> None:
    try:
        headers = _read_headers(conn)
        if not headers:
            return
        first = headers.split(b"\n", 1)[0].decode("iso-8859-1", errors="replace")
        method = first.split()[0].upper() if first.split() else ""
        target = _header_host(headers)
        default_port = 443 if method == "CONNECT" else 80
        host, port = _split_host_port(target, default_port)
        if not domain_allowed(host, network):
            body = (network_denied(host) + "\n").encode()
            conn.sendall(
                b"HTTP/1.1 403 Forbidden\r\nContent-Type: text/plain\r\nConnection: close\r\n\r\n" + body
            )
            return
        try:
            addrs = resolve_unblocked(host, port)
        except socket.gaierror:
            body = (network_denied(host) + "\n").encode()
            conn.sendall(
                b"HTTP/1.1 403 Forbidden\r\nContent-Type: text/plain\r\nConnection: close\r\n\r\n" + body
            )
            return
        if not addrs:
            body = (network_denied(f"{host} resolved to a blocked address") + "\n").encode()
            conn.sendall(
                b"HTTP/1.1 403 Forbidden\r\nContent-Type: text/plain\r\nConnection: close\r\n\r\n" + body
            )
            return
        upstream = None
        last_err = None
        for fam, sockaddr in addrs:
            try:
                upstream = socket.socket(fam, socket.SOCK_STREAM)
                upstream.settimeout(20)
                upstream.connect(sockaddr)
                break
            except OSError as e:
                last_err = e
                if upstream is not None:
                    upstream.close()
                    upstream = None
        if upstream is None:
            msg = f"sandbox_denied: network ({host} connect failed: {last_err})\n"
            conn.sendall(
                b"HTTP/1.1 502 Bad Gateway\r\nContent-Type: text/plain\r\nConnection: close\r\n\r\n"
                + msg.encode()
            )
            return
        if method == "CONNECT":
            conn.sendall(b"HTTP/1.1 200 Connection Established\r\n\r\n")
        else:
            upstream.sendall(headers)
        _splice(conn, upstream)
        upstream.close()
    except OSError:
        return
    finally:
        try:
            conn.close()
        except OSError:
            pass


class DomainProxy:
    def __init__(self, sock_path: Path, network: NetworkPolicy):
        self.sock_path = Path(sock_path)
        self.network = network
        self._sock: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def start(self) -> Path:
        if self.sock_path.exists():
            self.sock_path.unlink()
        self.sock_path.parent.mkdir(parents=True, exist_ok=True)
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.bind(str(self.sock_path))
        sock.listen(32)
        sock.settimeout(0.5)
        self._sock = sock
        self._thread = threading.Thread(target=self._serve, name="orbweaver-proxy", daemon=True)
        self._thread.start()
        return self.sock_path

    def _serve(self) -> None:
        assert self._sock is not None
        while not self._stop.is_set():
            try:
                conn, _ = self._sock.accept()
            except TimeoutError:
                continue
            except OSError:
                if self._stop.is_set():
                    return
                continue
            threading.Thread(target=_handle, args=(conn, self.network), daemon=True).start()

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


RELAY_SOURCE = r"""
import socket, select, sys, subprocess, os
sock_path, port = sys.argv[1], int(sys.argv[2])
try:
    subprocess.run(["ip", "link", "set", "lo", "up"], check=False, capture_output=True)
except OSError:
    pass
srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
srv.bind(("127.0.0.1", port))
srv.listen(32)
while True:
    client, _ = srv.accept()
    try:
        upstream = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        upstream.connect(sock_path)
    except OSError:
        client.close()
        continue
    pair = [client, upstream]
    try:
        while True:
            ready, _, _ = select.select(pair, [], [], 60)
            if not ready:
                break
            for src in ready:
                dest = upstream if src is client else client
                data = src.recv(65536)
                if not data:
                    raise OSError("eof")
                dest.sendall(data)
    except OSError:
        pass
    finally:
        client.close()
        upstream.close()
"""


def wrap_command_with_proxy(command: str, unix_sock: str, port: int = PROXY_PORT) -> str:
    import shlex

    inner = (
        f"python3 -c {shlex.quote(RELAY_SOURCE.strip())} {shlex.quote(unix_sock)} {port} &\n"
        "ow_relay_pid=$!\n"
        "ip link set lo up >/dev/null 2>&1 || true\n"
        f"export http_proxy=http://127.0.0.1:{port}\n"
        f"export https_proxy=http://127.0.0.1:{port}\n"
        f"export HTTP_PROXY=http://127.0.0.1:{port}\n"
        f"export HTTPS_PROXY=http://127.0.0.1:{port}\n"
        f"export ALL_PROXY=http://127.0.0.1:{port}\n"
        f"export ORBWEAVER_SSH_PROXY_HOST=127.0.0.1\n"
        f"export ORBWEAVER_SSH_PROXY_PORT={port}\n"
        "export NO_PROXY=localhost,127.0.0.1,::1\n"
        "export no_proxy=localhost,127.0.0.1,::1\n"
        f"{command}\n"
        "ow_status=$?\n"
        "kill $ow_relay_pid >/dev/null 2>&1 || true\n"
        "exit $ow_status\n"
    )
    return inner
