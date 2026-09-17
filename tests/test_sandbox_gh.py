"""Sandboxed `gh` must use host credentials without leaking GH_TOKEN (#93 / Telegram).

Production: Telegram (and every LocalWorkspace turn) runs Bash in bubblewrap.
`~/.config/gh` is denyRead and `*TOKEN*` is env-excluded, so `gh pr` / `gh auth
status` printed `gh auth login` even when the gateway process had GH_TOKEN.
The host-side gh proxy is the SSH_AUTH_SOCK analogue.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import stat
import struct
from pathlib import Path

import pytest

from orbweaver.agent import static_system
from orbweaver.sandbox.bwrap import (
    SandboxUnavailable,
    build_bwrap_argv,
    is_containerized,
    run_sandboxed,
    sandbox_available,
)
from orbweaver.sandbox.gh import (
    SANDBOX_GH_SOCK,
    GhProxy,
    ensure_gh_sandbox,
    forbidden_gh_argv,
    gh_overlay_args,
    host_gh_env,
)

_HEADER = struct.Struct(">I")
FAKE_TOKEN = "gho_testSandboxGhTokenValue"


def _fake_gh(bin_dir: Path) -> Path:
    bin_dir.mkdir(parents=True, exist_ok=True)
    path = bin_dir / "gh"
    path.write_text(
        "#!/usr/bin/env python3\n"
        "import os, sys\n"
        "token = os.environ.get('GH_TOKEN') or os.environ.get('GITHUB_TOKEN') or ''\n"
        "args = sys.argv[1:]\n"
        "if not token:\n"
        "    sys.stderr.write('To get started with GitHub CLI, please run:  gh auth login\\n')\n"
        "    raise SystemExit(1)\n"
        "if args[:2] == ['auth', 'status']:\n"
        "    print('github.com')\n"
        "    print('  Logged in to github.com account test')\n"
        "    raise SystemExit(0)\n"
        "if args[:2] == ['auth', 'token']:\n"
        "    print(token)\n"
        "    raise SystemExit(0)\n"
        "print('ok', *args)\n"
        "raise SystemExit(0)\n",
        encoding="utf-8",
    )
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return path


def test_forbidden_gh_argv_blocks_credential_leaks():
    assert forbidden_gh_argv(["pr", "create"]) is None
    assert forbidden_gh_argv(["auth", "status"]) is None
    assert forbidden_gh_argv([]) is None
    for argv in (
        ["auth", "token"],
        ["auth", "login"],
        ["auth", "logout"],
        ["auth", "refresh"],
        ["extension", "install", "x"],
        ["+", "foo"],
    ):
        reason = forbidden_gh_argv(argv)
        assert reason, argv
        assert "token" not in reason.lower() or "blocked" in reason.lower() or "auth" in reason.lower()


def test_host_gh_env_keeps_token_out_of_unrelated_keys():
    env = host_gh_env(
        {
            "PATH": "/bin",
            "HOME": "/home/gw",
            "GH_TOKEN": FAKE_TOKEN,
            "ANTHROPIC_API_KEY": "sk-ant-secret",
            "TELEGRAM_BOT_TOKEN": "123:abc",
        }
    )
    assert env["GH_TOKEN"] == FAKE_TOKEN
    assert "ANTHROPIC_API_KEY" not in env
    assert "TELEGRAM_BOT_TOKEN" not in env


def test_static_system_mentions_host_gh_proxy():
    text = static_system()
    assert "gh auth login" in text
    assert "git@" in text or "SSH" in text


def _call_proxy(sock_path: Path, argv: list[str], cwd: str) -> dict:
    payload = json.dumps({"argv": argv, "cwd": cwd, "stdin": ""}).encode()
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.connect(str(sock_path))
    sock.sendall(_HEADER.pack(len(payload)) + payload)
    hdr = sock.recv(4)
    (n,) = _HEADER.unpack(hdr)
    raw = b""
    while len(raw) < n:
        chunk = sock.recv(n - len(raw))
        assert chunk
        raw += chunk
    sock.close()
    return json.loads(raw)


def test_gh_proxy_uses_host_token_and_blocks_auth_token(tmp_path, monkeypatch):
    """Same error class as production: host has GH_TOKEN, sandboxed gh must not see it
    as an env var, but `gh auth status` must succeed and `gh auth token` must not
    print the secret."""
    fake = _fake_gh(tmp_path / "bin")
    monkeypatch.setenv("PATH", str(fake.parent) + os.pathsep + os.environ.get("PATH", ""))
    monkeypatch.setenv("GH_TOKEN", FAKE_TOKEN)
    sock = tmp_path / "gh.sock"
    proxy = GhProxy(sock, workspace_root=tmp_path, gh_bin=fake, environ=dict(os.environ))
    assert proxy.start() == sock
    try:
        status = _call_proxy(sock, ["auth", "status"], str(tmp_path))
        assert status["returncode"] == 0
        assert "Logged in" in status["stdout"]
        assert FAKE_TOKEN not in status["stdout"]

        leaked = _call_proxy(sock, ["auth", "token"], str(tmp_path))
        assert leaked["returncode"] != 0
        assert FAKE_TOKEN not in leaked["stdout"] + leaked["stderr"]
        assert "blocked" in leaked["stderr"].lower() or "auth" in leaked["stderr"].lower()

        outside = _call_proxy(sock, ["auth", "status"], "/tmp")
        assert outside["returncode"] != 0
        assert "outside" in outside["stderr"].lower()
    finally:
        proxy.close()


def test_bwrap_overlays_gh_and_exports_sock_not_token(tmp_path, monkeypatch):
    fake = _fake_gh(tmp_path / "bin")
    monkeypatch.setenv("PATH", str(fake.parent) + os.pathsep + os.environ.get("PATH", ""))
    environ = {
        "PATH": os.environ["PATH"],
        "HOME": str(tmp_path),
        "GH_TOKEN": FAKE_TOKEN,
    }
    sock = tmp_path / "gh.sock"
    sock.write_text("", encoding="utf-8")
    ensure_gh_sandbox(tmp_path / "tmp")
    argv = build_bwrap_argv(
        "gh auth status",
        tmp_path,
        tmp_path / "tmp",
        environ=environ,
        gh_sock=sock,
    )
    joined = " ".join(argv)
    assert "GH_PROXY_SOCK" in argv
    assert SANDBOX_GH_SOCK in argv
    assert FAKE_TOKEN not in joined
    assert "GH_TOKEN" not in argv
    overlay = gh_overlay_args(tmp_path / "tmp", sock, environ)
    assert str(fake.resolve()) in overlay or str(fake) in overlay


def test_sandboxed_gh_auth_status_uses_host_token(monkeypatch, tmp_path):
    """Execute bubblewrap the way Telegram Bash does (not argv-only).

    Workspace is a short ``/tmp`` path so the CONNECT proxy unix socket stays
    under the AF_UNIX limit (pytest tmp_path under TMPDIR=/var/tmp does not).
    """
    from orbweaver.config import settings

    if not sandbox_available() or is_containerized():
        pytest.skip("bubblewrap not available")
    monkeypatch.setattr(settings, "orbweaver_sandbox", True)
    monkeypatch.setattr(settings, "orbweaver_sandbox_fail_if_unavailable", True)
    fake = _fake_gh(tmp_path / "bin")
    monkeypatch.setenv("PATH", str(fake.parent) + os.pathsep + os.environ.get("PATH", ""))
    monkeypatch.setenv("GH_TOKEN", FAKE_TOKEN)
    root = Path("/tmp/owghtest")
    if root.exists():
        shutil.rmtree(root)
    root.mkdir()
    try:
        try:
            probe = run_sandboxed("true", root, timeout=10)
        except SandboxUnavailable as e:
            pytest.skip(str(e))
        if "sandbox_unavailable" in probe:
            pytest.skip(probe[-200:])
        out = run_sandboxed(
            "gh auth status; echo TOKEN=${GH_TOKEN-unset}",
            root,
            timeout=20,
        )
        assert "gh auth login" not in out.lower()
        assert "Logged in" in out
        assert FAKE_TOKEN not in out
        assert "TOKEN=unset" in out or "TOKEN=\n" in out

        blocked = run_sandboxed("gh auth token", root, timeout=20)
        assert FAKE_TOKEN not in blocked
        assert "please run" not in blocked.lower()
        assert "blocked" in blocked.lower() or "host holds" in blocked.lower()
    finally:
        shutil.rmtree(root, ignore_errors=True)
