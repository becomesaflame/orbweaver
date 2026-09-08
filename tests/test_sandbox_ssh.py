import os
import socket
import subprocess
import sys
import threading
from pathlib import Path

from orbweaver.sandbox.bwrap import build_bwrap_argv
from orbweaver.sandbox.policy import load_sandbox_policy
from orbweaver.sandbox.proxy import wrap_command_with_proxy
from orbweaver.sandbox.ssh import (
    ensure_ssh_sandbox,
    filter_ssh_argv,
    ssh_identity_bind_args,
)


def test_filter_ssh_argv_drops_dash_f():
    assert filter_ssh_argv(["-F", "/dev/null", "-o", "StrictHostKeyChecking=no", "git@github.com"]) == [
        "-o",
        "StrictHostKeyChecking=no",
        "git@github.com",
    ]
    assert filter_ssh_argv(["-F/tmp/x", "host"]) == ["host"]
    assert filter_ssh_argv(["-i", "key", "host"]) == ["-i", "key", "host"]


def test_ensure_ssh_sandbox_proxied_config(tmp_path: Path):
    dest = ensure_ssh_sandbox(tmp_path / "tmp", proxied=True)
    text = (dest / "config").read_text(encoding="utf-8")
    assert "ProxyCommand" in text
    assert "proxycmd.py" in text
    assert (dest / "ssh").is_file()
    direct = ensure_ssh_sandbox(tmp_path / "tmp2", proxied=False)
    assert "ProxyCommand" not in (direct / "config").read_text(encoding="utf-8")


def test_bwrap_overlays_ssh_and_hides_config_d(tmp_path: Path):
    argv = build_bwrap_argv("echo hi", tmp_path, tmp_path / "tmp")
    assert "/etc/ssh/ssh_config" in argv
    assert "/etc/ssh/ssh_config.d" in argv
    i = argv.index("/etc/ssh/ssh_config.d")
    assert argv[i - 1] == "--tmpfs"
    if Path("/usr/bin/ssh").exists():
        assert "/usr/bin/ssh" in argv
        wrapper = tmp_path / "tmp" / "ow-ssh" / "ssh"
        assert str(wrapper.resolve()) in argv


def test_bwrap_full_network_skips_proxycommand(tmp_path: Path):
    build_bwrap_argv("true", tmp_path, tmp_path / "tmp", full_network=True)
    config = (tmp_path / "tmp" / "ow-ssh" / "config").read_text(encoding="utf-8")
    assert "ProxyCommand" not in config


def test_identity_binds_after_deny_read_skip_config(tmp_path: Path, monkeypatch):
    home = tmp_path / "home"
    ssh = home / ".ssh"
    ssh.mkdir(parents=True)
    key = ssh / "id_ed25519"
    key.write_text("fake-key", encoding="utf-8")
    (ssh / "config").write_text("Host *\n  ProxyJump evil\n", encoding="utf-8")
    monkeypatch.setattr("orbweaver.sandbox.ssh.Path.home", lambda: home)
    policy = load_sandbox_policy(tmp_path, environ={}, home=home)
    argv = build_bwrap_argv("true", tmp_path, tmp_path / "tmp", policy=policy)
    ssh_dir = str(ssh.resolve())
    key_path = str(key.resolve())
    assert "--tmpfs" in argv
    tmpfs_at = None
    for i, a in enumerate(argv):
        if a == "--tmpfs" and i + 1 < len(argv) and argv[i + 1] == ssh_dir:
            tmpfs_at = i
    assert tmpfs_at is not None
    assert key_path in argv
    assert argv.index(key_path) > tmpfs_at
    assert str((ssh / "config").resolve()) not in argv
    binds = ssh_identity_bind_args(home)
    assert str((ssh / "config").resolve()) not in binds


def test_ssh_agent_socket_allowlisted(tmp_path: Path, monkeypatch):
    sock = tmp_path / "agent.sock"
    sock.write_text("", encoding="utf-8")
    monkeypatch.setenv("SSH_AUTH_SOCK", str(sock))
    policy = load_sandbox_policy(tmp_path, environ={"SSH_AUTH_SOCK": str(sock)}, home=tmp_path)
    assert sock.resolve() in {p.resolve() for p in policy.allow_unix_sockets}


def test_wrap_command_exports_ssh_proxy_port():
    wrapped = wrap_command_with_proxy("true", "/tmp/ow.sock", port=19191)
    assert "ORBWEAVER_SSH_PROXY_PORT=19191" in wrapped
    assert "ORBWEAVER_SSH_PROXY_HOST=127.0.0.1" in wrapped


def test_proxycmd_tunnels_connect(tmp_path: Path):
    dest = ensure_ssh_sandbox(tmp_path, proxied=True)
    got: dict[str, int] = {}
    ready = threading.Event()

    def server() -> None:
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("127.0.0.1", 0))
        srv.listen(1)
        got["port"] = srv.getsockname()[1]
        ready.set()
        conn, _ = srv.accept()
        buf = b""
        while b"\r\n\r\n" not in buf:
            chunk = conn.recv(4096)
            if not chunk:
                break
            buf += chunk
        assert b"CONNECT github.com:22" in buf
        conn.sendall(b"HTTP/1.1 200 Connection Established\r\n\r\n")
        payload = conn.recv(64)
        conn.sendall(b"pong-" + payload)
        conn.close()
        srv.close()

    thread = threading.Thread(target=server, daemon=True)
    thread.start()
    assert ready.wait(timeout=2)
    env = {
        **os.environ,
        "ORBWEAVER_SSH_PROXY_HOST": "127.0.0.1",
        "ORBWEAVER_SSH_PROXY_PORT": str(got["port"]),
    }
    proc = subprocess.run(
        [sys.executable, str(dest / "proxycmd.py"), "github.com", "22"],
        input=b"ping",
        capture_output=True,
        env=env,
        timeout=5,
        check=False,
    )
    thread.join(timeout=5)
    assert proc.returncode == 0, proc.stderr.decode()
    assert proc.stdout == b"pong-ping"
