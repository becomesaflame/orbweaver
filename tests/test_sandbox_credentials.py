"""Issue #95: credential files must be masked inside the bubblewrap sandbox.

Argv checks cover the policy; the ``live_root`` tests execute bwrap against a fake
HOME and read the files back, the way an agent's ``cat`` would.
"""

from __future__ import annotations

import shlex
import shutil
import tempfile
from pathlib import Path
from uuid import uuid4

import pytest
from test_sandbox_live import live_root  # noqa: F401

from orbweaver.permissions.denial import DenialTrackingState, reset_denial_states
from orbweaver.permissions.pipeline import can_use_tool
from orbweaver.permissions.rules import path_is_always_denied
from orbweaver.sandbox.bwrap import build_bwrap_argv, deny_read_overlay_args, run_sandboxed
from orbweaver.sandbox.policy import (
    DEFAULT_DENY_READ,
    SandboxPolicy,
    default_deny_read_paths,
    gateway_checkout_root,
    gateway_env_files,
    is_default_denied_read,
    is_env_secret_name,
    load_sandbox_policy,
)
from orbweaver.sandbox.ssh import SshPolicy
from orbweaver.store import Event
from orbweaver.workspace import LocalWorkspace

SECRET = "hunter2-do-not-leak"


def _write(path: Path, text: str = SECRET) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _fake_home(base: Path) -> Path:
    home = base.resolve() / "home"
    _write(home / ".config" / "gh" / "hosts.yml", f"github.com:\n  oauth_token: {SECRET}\n")
    _write(home / ".config" / "gh" / "config.yml", "git_protocol: ssh\n")
    _write(home / ".netrc", f"machine github.com login x password {SECRET}\n")
    _write(home / ".git-credentials", f"https://x:{SECRET}@github.com\n")
    _write(home / ".npmrc", f"//registry.npmjs.org/:_authToken={SECRET}\n")
    _write(home / ".pypirc", f"[pypi]\npassword = {SECRET}\n")
    _write(home / ".docker" / "config.json", '{"auths": {"x": {"auth": "' + SECRET + '"}}}')
    _write(home / ".kube" / "config", f"token: {SECRET}\n")
    _write(home / ".config" / "gcloud" / "credentials.db", SECRET)
    _write(home / ".azure" / "accessTokens.json", SECRET)
    _write(home / ".cargo" / "credentials.toml", f'[registry]\ntoken = "{SECRET}"\n')
    _write(home / ".cargo" / "config.toml", "[build]\njobs = 2\n")
    _write(home / ".ssh" / "id_ed25519", f"-----BEGIN OPENSSH PRIVATE KEY-----\n{SECRET}\n")
    _write(home / ".ssh" / "id_ed25519.pub", "ssh-ed25519 AAAAC3 public-key-ok")
    _write(home / ".ssh" / "custom-key", f"-----BEGIN OPENSSH PRIVATE KEY-----\n{SECRET}\n")
    _write(home / ".ssh" / "config", "Host github.com\n  IdentityFile ~/.ssh/id_ed25519\n")
    _write(home / ".ssh" / "known_hosts", "github.com ssh-ed25519 AAAA known-hosts-ok\n")
    _write(home / ".bashrc", "# plain dotfile\n")
    return home


def _fake_gateway(base: Path) -> Path:
    gateway = base.resolve() / "gateway"
    (gateway / "backend" / "orbweaver").mkdir(parents=True)
    _write(gateway / "backend" / ".env", f"ANTHROPIC_API_KEY={SECRET}\n")
    _write(gateway / "web" / ".env.local", f"VITE_TOKEN={SECRET}\n")
    _write(gateway / ".env.example", "ANTHROPIC_API_KEY=\n")
    _write(gateway / "web" / "node_modules" / "pkg" / ".env", "ignored-by-prune\n")
    return gateway


def test_default_deny_list_covers_issue_95_paths(tmp_path: Path, monkeypatch):
    home = _fake_home(tmp_path)
    gateway = _fake_gateway(tmp_path)
    monkeypatch.setattr("orbweaver.sandbox.policy.gateway_checkout_root", lambda: gateway)
    paths = set(default_deny_read_paths(home))
    for rel in (
        ".ssh",
        ".gnupg",
        ".aws",
        ".netrc",
        ".git-credentials",
        ".config/gh",
        ".config/hub",
        ".config/gcloud",
        ".docker/config.json",
        ".kube",
        ".azure",
        ".npmrc",
        ".pypirc",
        ".cargo/credentials.toml",
    ):
        assert home / rel in paths, rel
    assert home / ".cargo" / "config.toml" not in paths
    assert gateway / "backend" / ".env" in paths
    assert gateway / "web" / ".env.local" in paths
    assert gateway / ".env.example" not in paths
    assert not any("node_modules" in p.parts for p in paths)
    policy = load_sandbox_policy(tmp_path / "ws", environ={}, home=home)
    assert paths.issubset(set(policy.deny_read))


def test_env_secret_names():
    assert is_env_secret_name(".env")
    assert is_env_secret_name(".env.local")
    assert is_env_secret_name(".env.production")
    assert is_env_secret_name("prod.env")
    assert not is_env_secret_name(".env.example")
    assert not is_env_secret_name(".env.sample")
    assert not is_env_secret_name("environment.py")


def test_gateway_env_files_scans_running_checkout(tmp_path: Path):
    gateway = _fake_gateway(tmp_path)
    found = gateway_env_files(gateway)
    assert found[0] == gateway / "backend" / ".env"
    assert gateway / "web" / ".env.local" in found
    assert gateway / ".env.example" not in found
    assert not any("node_modules" in p.parts for p in found)
    # The real running package resolves to a checkout (or None when pip-installed).
    root = gateway_checkout_root()
    assert root is None or (root / "backend" / "orbweaver" / "__init__.py").is_file()


def test_allow_read_only_from_operator_sources(tmp_path: Path):
    home = _fake_home(tmp_path)
    root = tmp_path / "ws"
    (root / ".orbweaver").mkdir(parents=True)
    (root / ".orbweaver" / "sandbox.json").write_text(
        '{"allowRead": ["~/.npmrc", "~/.netrc"]}', encoding="utf-8"
    )
    policy = load_sandbox_policy(root, environ={}, home=home)
    assert home / ".npmrc" in policy.deny_read
    assert home / ".netrc" in policy.deny_read
    assert policy.allow_read == ()

    (home / ".orbweaver").mkdir()
    (home / ".orbweaver" / "sandbox.json").write_text('{"allowRead": ["~/.npmrc"]}', encoding="utf-8")
    policy = load_sandbox_policy(root, environ={}, home=home)
    assert home / ".npmrc" not in policy.deny_read
    assert home / ".netrc" in policy.deny_read
    assert home / ".ssh" in policy.deny_read

    policy = load_sandbox_policy(
        root, environ={"ORBWEAVER_SANDBOX_ALLOW_READ": "~/.netrc"}, home=home
    )
    assert home / ".netrc" not in policy.deny_read
    assert home / ".npmrc" not in policy.deny_read


def test_is_default_denied_read_matches_home_paths(monkeypatch):
    assert is_default_denied_read("/home/x/.config/gh/hosts.yml")
    assert is_default_denied_read("/home/x/.config/gh")
    assert is_default_denied_read("/root/.netrc")
    assert is_default_denied_read("~/.docker/config.json")
    assert is_default_denied_read("~/.cargo/credentials.toml")
    assert is_default_denied_read("~/.kube/config")
    assert not is_default_denied_read("~/.cargo/config.toml")
    assert not is_default_denied_read("/home/x/.config/other/hosts.yml")
    assert not is_default_denied_read("/home/x/.docker/daemon.json")
    assert not is_default_denied_read("src/.npmrc")
    assert not is_default_denied_read(".netrc")
    assert not is_default_denied_read("/etc/passwd")
    assert path_is_always_denied("/home/x/.config/gh/hosts.yml")
    assert path_is_always_denied("~/.git-credentials")
    assert not path_is_always_denied("/home/x/.config/nvim/init.lua")
    for raw in DEFAULT_DENY_READ:
        assert raw.startswith("~/"), raw


@pytest.mark.asyncio
async def test_read_tool_denies_gh_hosts(tmp_path: Path, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("classifier should not run")

    monkeypatch.setattr("orbweaver.permissions.pipeline.classify_action", boom)
    reset_denial_states()
    sid = uuid4()
    ctx = {
        "workspace": LocalWorkspace("workspace:default", str(tmp_path)),
        "workspace_kind": "local",
        "headless": False,
        "session_id": sid,
        "denial_state": DenialTrackingState(),
        "events": [Event(id=uuid4(), session_id=sid, seq=1, kind="user", payload={"text": "hi"})],
    }
    decision = await can_use_tool("Read", {"path": "/home/x/.config/gh/hosts.yml"}, ctx)
    assert decision.behavior == "deny"
    decision = await can_use_tool("Read", {"path": "~/.kube/config"}, ctx)
    assert decision.behavior == "deny"


def test_deny_overlay_masks_symlink_target(tmp_path: Path):
    real = _write(tmp_path / "dotfiles" / "netrc")
    link = tmp_path / ".netrc"
    link.symlink_to(real)
    real_dir = tmp_path / "dotfiles" / "kube"
    real_dir.mkdir()
    dir_link = tmp_path / ".kube"
    dir_link.symlink_to(real_dir)
    args = deny_read_overlay_args(
        SandboxPolicy(deny_read=(link, dir_link, tmp_path / "missing", real))
    )
    assert args == ["--ro-bind", "/dev/null", str(real), "--tmpfs", str(real_dir)]


def test_bwrap_argv_masks_every_default_entry(tmp_path: Path, monkeypatch):
    home = _fake_home(tmp_path)
    gateway = _fake_gateway(tmp_path)
    monkeypatch.setattr("orbweaver.sandbox.policy.gateway_checkout_root", lambda: gateway)
    monkeypatch.setattr("orbweaver.sandbox.ssh.Path.home", lambda: home)
    root = tmp_path / "ws"
    root.mkdir()
    policy = load_sandbox_policy(root, environ={}, home=home)
    argv = build_bwrap_argv("true", root, root / ".orbweaver-tmp", policy=policy)

    def mask_for(path: Path) -> str | None:
        dest = str(path)
        for i, a in enumerate(argv):
            if a == dest and i >= 2 and argv[i - 2] == "--ro-bind" and argv[i - 1] == "/dev/null":
                return "file"
            if a == dest and i >= 1 and argv[i - 1] == "--tmpfs":
                return "dir"
        return None

    assert mask_for(home / ".ssh") == "dir"
    assert mask_for(home / ".config" / "gh") == "dir"
    assert mask_for(home / ".kube") == "dir"
    assert mask_for(home / ".netrc") == "file"
    assert mask_for(home / ".docker" / "config.json") == "file"
    assert mask_for(home / ".cargo" / "credentials.toml") == "file"
    assert mask_for(gateway / "backend" / ".env") == "file"
    assert mask_for(gateway / "web" / ".env.local") == "file"
    assert mask_for(home / ".cargo" / "config.toml") is None
    assert mask_for(home / ".bashrc") is None
    assert str(home / ".ssh" / "id_ed25519") not in argv
    assert str(home / ".ssh" / "custom-key") not in argv
    assert str(home / ".ssh" / "id_ed25519.pub") in argv
    assert str(home / ".ssh" / "config") in argv


# --- live bubblewrap -----------------------------------------------------------------


@pytest.fixture
def cred_base(tmp_path_factory):
    """Fake HOME outside the workspace root (the rw --bind would shadow the masks) and
    outside /tmp (the sandbox puts a tmpfs over /tmp)."""
    sibling = tmp_path_factory.mktemp("cred")
    if not str(sibling).startswith("/tmp/"):
        yield sibling
        return
    base = Path(tempfile.mkdtemp(prefix="ow-issue95-", dir="/var/tmp"))
    try:
        yield base
    finally:
        shutil.rmtree(base, ignore_errors=True)


_PROBE = """
probe() {{
  content=$(cat "$1" 2>/dev/null)
  if [ -n "$content" ]; then echo "READ $1"; else echo "DENIED $1"; fi
}}
for f in {files}; do probe "$f"; done
echo "LS_SSH: $(ls -A {ssh} 2>/dev/null | tr '\\n' ' ')"
echo "LS_GH: $(ls -A {gh} 2>/dev/null | tr '\\n' ' ')"
"""


def _probe_command(home: Path, gateway: Path) -> str:
    files = [
        home / ".config" / "gh" / "hosts.yml",
        home / ".netrc",
        home / ".git-credentials",
        home / ".npmrc",
        home / ".pypirc",
        home / ".docker" / "config.json",
        home / ".kube" / "config",
        home / ".config" / "gcloud" / "credentials.db",
        home / ".azure" / "accessTokens.json",
        home / ".cargo" / "credentials.toml",
        home / ".ssh" / "id_ed25519",
        home / ".ssh" / "custom-key",
        gateway / "backend" / ".env",
        gateway / "web" / ".env.local",
        # must stay readable
        home / ".ssh" / "id_ed25519.pub",
        home / ".ssh" / "config",
        home / ".ssh" / "known_hosts",
        home / ".cargo" / "config.toml",
        home / ".bashrc",
        gateway / ".env.example",
    ]
    return _PROBE.format(
        files=" ".join(shlex.quote(str(f)) for f in files),
        ssh=shlex.quote(str(home / ".ssh")),
        gh=shlex.quote(str(home / ".config" / "gh")),
    )


def _lines(out: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in out.splitlines():
        if line.startswith(("READ ", "DENIED ")):
            verdict, path = line.split(" ", 1)
            result[path] = verdict
    return result


def test_live_credentials_masked_in_sandbox(live_root, cred_base, monkeypatch):  # noqa: F811
    """cat inside bwrap: credential files empty/absent, public ssh files still readable."""
    home = _fake_home(cred_base)
    gateway = _fake_gateway(cred_base)
    monkeypatch.setattr("orbweaver.sandbox.policy.gateway_checkout_root", lambda: gateway)
    monkeypatch.setattr("orbweaver.sandbox.ssh.Path.home", lambda: home)
    policy = load_sandbox_policy(live_root, environ={}, home=home)
    out = run_sandboxed(_probe_command(home, gateway), live_root, timeout=20, policy=policy)
    assert "sandbox_unavailable" not in out, out
    assert SECRET not in out, out
    got = _lines(out)
    denied = [
        home / ".config" / "gh" / "hosts.yml",
        home / ".netrc",
        home / ".git-credentials",
        home / ".npmrc",
        home / ".pypirc",
        home / ".docker" / "config.json",
        home / ".kube" / "config",
        home / ".config" / "gcloud" / "credentials.db",
        home / ".azure" / "accessTokens.json",
        home / ".cargo" / "credentials.toml",
        home / ".ssh" / "id_ed25519",
        home / ".ssh" / "custom-key",
        gateway / "backend" / ".env",
        gateway / "web" / ".env.local",
    ]
    for path in denied:
        assert got.get(str(path)) == "DENIED", (path, out)
    readable = [
        home / ".ssh" / "id_ed25519.pub",
        home / ".ssh" / "config",
        home / ".ssh" / "known_hosts",
        home / ".cargo" / "config.toml",
        home / ".bashrc",
        gateway / ".env.example",
    ]
    for path in readable:
        assert got.get(str(path)) == "READ", (path, out)
    ls_ssh = next(line for line in out.splitlines() if line.startswith("LS_SSH:"))
    assert "id_ed25519.pub" in ls_ssh
    assert "config" in ls_ssh
    assert "custom-key" not in ls_ssh
    assert " id_ed25519 " not in ls_ssh + " "
    ls_gh = next(line for line in out.splitlines() if line.startswith("LS_GH:"))
    assert "hosts.yml" not in ls_gh


def test_live_bind_identities_exposes_only_named_key(live_root, cred_base, monkeypatch):  # noqa: F811
    home = _fake_home(cred_base)
    gateway = _fake_gateway(cred_base)
    monkeypatch.setattr("orbweaver.sandbox.policy.gateway_checkout_root", lambda: gateway)
    monkeypatch.setattr("orbweaver.sandbox.ssh.Path.home", lambda: home)
    (home / ".orbweaver").mkdir()
    (home / ".orbweaver" / "sandbox.json").write_text(
        '{"ssh": {"bindIdentities": true}}', encoding="utf-8"
    )
    policy = load_sandbox_policy(live_root, environ={}, home=home)
    assert policy.ssh == SshPolicy(bind_identities=True)
    out = run_sandboxed(_probe_command(home, gateway), live_root, timeout=20, policy=policy)
    got = _lines(out)
    assert got.get(str(home / ".ssh" / "id_ed25519")) == "READ", out
    assert got.get(str(home / ".ssh" / "custom-key")) == "DENIED", out
    assert got.get(str(home / ".config" / "gh" / "hosts.yml")) == "DENIED", out
    assert got.get(str(home / ".netrc")) == "DENIED", out


def test_live_allow_read_restores_one_entry(live_root, cred_base, monkeypatch):  # noqa: F811
    home = _fake_home(cred_base)
    gateway = _fake_gateway(cred_base)
    monkeypatch.setattr("orbweaver.sandbox.policy.gateway_checkout_root", lambda: gateway)
    monkeypatch.setattr("orbweaver.sandbox.ssh.Path.home", lambda: home)
    (home / ".orbweaver").mkdir()
    (home / ".orbweaver" / "sandbox.json").write_text('{"allowRead": ["~/.npmrc"]}', encoding="utf-8")
    policy = load_sandbox_policy(live_root, environ={}, home=home)
    out = run_sandboxed(_probe_command(home, gateway), live_root, timeout=20, policy=policy)
    got = _lines(out)
    assert got.get(str(home / ".npmrc")) == "READ", out
    assert got.get(str(home / ".netrc")) == "DENIED", out
    assert got.get(str(home / ".ssh" / "id_ed25519")) == "DENIED", out
