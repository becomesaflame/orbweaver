import socket
from pathlib import Path

from orbweaver.sandbox.errors import (
    label_sandbox_output,
    network_denied,
    unix_socket_denied,
)
from orbweaver.sandbox.policy import NetworkPolicy
from orbweaver.sandbox.proxy import (
    DomainProxy,
    domain_allowed,
    host_matches,
    ip_is_blocked,
)


def test_host_match_and_deny_wins():
    assert host_matches("pypi.org", "pypi.org")
    assert host_matches("files.pythonhosted.org", "*.pythonhosted.org")
    assert host_matches("pythonhosted.org", "*.pythonhosted.org")
    assert not host_matches("evilpythonhosted.org", "*.pythonhosted.org")
    net = NetworkPolicy(default="deny", include_defaults=True, allow=("example.com",), deny=("*.evil.com",))
    assert domain_allowed("pypi.org", net)
    assert domain_allowed("example.com", net)
    assert not domain_allowed("phish.evil.com", NetworkPolicy(default="allow", deny=("*.evil.com",)))
    assert not domain_allowed("blocked.example", NetworkPolicy(default="deny", include_defaults=False))


def test_private_and_metadata_ips_blocked():
    assert ip_is_blocked("127.0.0.1")
    assert ip_is_blocked("10.1.2.3")
    assert ip_is_blocked("192.168.1.1")
    assert ip_is_blocked("169.254.169.254")
    assert ip_is_blocked("::1")
    assert not ip_is_blocked("1.1.1.1")


def test_proxy_connect_denied(tmp_path: Path):
    sock = tmp_path / "proxy.sock"
    proxy = DomainProxy(sock, NetworkPolicy(default="deny", include_defaults=False))
    proxy.start()
    try:
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        client.connect(str(sock))
        client.sendall(b"CONNECT secret.internal:443 HTTP/1.1\r\nHost: secret.internal:443\r\n\r\n")
        data = b""
        while b"\r\n\r\n" not in data:
            chunk = client.recv(4096)
            if not chunk:
                break
            data += chunk
            if len(data) > 8000:
                break
        client.close()
    finally:
        proxy.close()
    text = data.decode("utf-8", errors="replace")
    assert "403" in text
    assert "sandbox_denied: network" in text
    assert "secret.internal" in text
    assert "full_network" in text


def test_label_output_prefixes():
    assert "unix_socket" in unix_socket_denied("/run/postgresql/.s.PGSQL.5432")
    assert "full_network" in network_denied("pypi.org")
    labeled = label_sandbox_output("curl: (7) Failed to connect: Network is unreachable")
    assert labeled.startswith("sandbox_denied: network")
    already = "sandbox_denied: network (x)\nrest"
    assert label_sandbox_output(already) == already
