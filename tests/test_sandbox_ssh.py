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
    SANDBOX_OPENSSH,
    SANDBOX_SSH_DIR,
    ensure_ssh_sandbox,
    filter_ssh_argv,
    sandbox_resolv_conf_text,
    ssh_identity_bind_args,
    ssh_private_identity_files,
)
from sandbox_mounts import assert_ro_bind_dests_creatable


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
        assert SANDBOX_OPENSSH in argv
        assert argv[argv.index(SANDBOX_OPENSSH) - 2 : argv.index(SANDBOX_OPENSSH)] == [
            "--ro-bind",
            "/usr/bin/ssh",
        ]
        assert str((tmp_path / "tmp" / "ow-ssh" / "openssh").resolve()) not in argv
        tmpfs_tmp = None
        for idx, a in enumerate(argv):
            if a == "--tmpfs" and idx + 1 < len(argv) and argv[idx + 1] == "/tmp":
                tmpfs_tmp = idx
                break
        assert tmpfs_tmp is not None
        assert argv[argv.index(SANDBOX_SSH_DIR) - 1] == "--dir"
        assert tmpfs_tmp < argv.index(SANDBOX_SSH_DIR)
        wrapper_text = wrapper.read_text(encoding="utf-8")
        assert SANDBOX_OPENSSH in wrapper_text
    assert_ro_bind_dests_creatable(argv)


def test_bwrap_full_network_skips_proxycommand(tmp_path: Path):
    build_bwrap_argv("true", tmp_path, tmp_path / "tmp", full_network=True)
    config = (tmp_path / "tmp" / "ow-ssh" / "config").read_text(encoding="utf-8")
    assert "ProxyCommand" not in config


def test_sandbox_resolv_conf_drops_stub_resolver():
    text = sandbox_resolv_conf_text(
        "nameserver 127.0.0.53\noptions edns0 trust-ad\nsearch example.test\n"
    )
    assert "127.0.0.53" not in text
    assert "nameserver 1.1.1.1" in text
    assert "search example.test" in text


def test_sandbox_resolv_conf_prefers_uplink_ipv4():
    text = sandbox_resolv_conf_text(
        "nameserver 2a01:4ff:ff00::add:2\nnameserver 185.12.64.1\nsearch tail.ts.net\n"
    )
    assert "185.12.64.1" in text
    assert "127.0.0.53" not in text
    assert text.index("185.12.64.1") < text.index("2a01:")


def test_bwrap_overlays_resolv_conf(tmp_path: Path):
    argv = build_bwrap_argv("true", tmp_path, tmp_path / "tmp")
    resolv = tmp_path / "tmp" / "ow-ssh" / "resolv.conf"
    assert resolv.is_file()
    assert "127.0.0.53" not in resolv.read_text(encoding="utf-8")
    assert str(resolv.resolve()) in argv
    tmpfs_run = None
    for idx, a in enumerate(argv):
        if a == "--tmpfs" and idx + 1 < len(argv) and argv[idx + 1] == "/run":
            tmpfs_run = idx
            break
    assert tmpfs_run is not None
    assert argv.index(str(resolv.resolve())) > tmpfs_run
    dest = argv[argv.index(str(resolv.resolve())) + 1]
    assert dest in {"/etc/resolv.conf", "/run/systemd/resolve/stub-resolv.conf"}


def test_custom_identity_from_dir_and_host_config(tmp_path: Path, monkeypatch):
    """Production github.com SSH uses IdentityFile ~/.ssh/lampropeltis-orbweaver, not id_ed25519."""
    home = tmp_path / "home"
    ssh = home / ".ssh"
    ssh.mkdir(parents=True)
    key = ssh / "lampropeltis-orbweaver"
    key.write_text("fake-key", encoding="utf-8")
    (ssh / "config").write_text(
        "Host github.com\n  IdentityFile ~/.ssh/lampropeltis-orbweaver\n  IdentitiesOnly yes\n  ProxyJump evil\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("orbweaver.sandbox.ssh.Path.home", lambda: home)
    dest = ensure_ssh_sandbox(tmp_path / "tmp", proxied=True)
    cfg = (dest / "config").read_text(encoding="utf-8")
    assert "lampropeltis-orbweaver" in cfg
    assert "ProxyJump" not in cfg
    assert "IdentitiesOnly" not in cfg
    priv = ssh_private_identity_files(home)
    assert key.resolve() in priv
    binds = ssh_identity_bind_args(home)
    assert str(key.resolve()) in binds
    assert str((ssh / "config").resolve()) not in binds
    assert str((ssh / "config").resolve()) not in cfg


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
