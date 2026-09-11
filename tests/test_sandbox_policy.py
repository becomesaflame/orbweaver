from pathlib import Path

from orbweaver.sandbox.bwrap import build_bwrap_argv
from orbweaver.sandbox.domains import DEFAULT_ALLOWED_DOMAINS
from orbweaver.sandbox.policy import (
    default_journal_sockets,
    is_hardcoded_socket_deny,
    load_sandbox_policy,
)


def test_hardcoded_denies_and_docker_sock(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("orbweaver.sandbox.policy.Path.home", lambda: tmp_path)
    policy = load_sandbox_policy(tmp_path, environ={}, home=tmp_path)
    homes = {str(p) for p in policy.deny_read}
    assert any(str(p).endswith(".ssh") for p in policy.deny_read), homes
    assert is_hardcoded_socket_deny(Path("/run/docker.sock"))
    assert is_hardcoded_socket_deny(tmp_path / "docker.sock")
    journals = set(default_journal_sockets())
    assert journals.issubset(set(policy.allow_unix_sockets))


def test_loaded_policy_binds_journal_socket(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("orbweaver.sandbox.policy.Path.home", lambda: tmp_path)
    policy = load_sandbox_policy(tmp_path, environ={}, home=tmp_path)
    argv = build_bwrap_argv("true", tmp_path, tmp_path / "tmp", policy=policy)
    sock = "/run/systemd/journal/socket"
    assert sock in argv
    assert argv[argv.index(sock) - 1] == "--ro-bind-try"


def test_merge_unions_paths_and_repo_wins_default(tmp_path: Path, monkeypatch):
    home = tmp_path / "home"
    root = tmp_path / "ws"
    (home / ".orbweaver").mkdir(parents=True)
    (root / ".orbweaver").mkdir(parents=True)
    extra_ro = tmp_path / "notes"
    extra_ro.mkdir()
    extra_rw = tmp_path / "cache"
    extra_rw.mkdir()
    (home / ".orbweaver" / "sandbox.json").write_text(
        '{"additionalReadonlyPaths": ["'
        + str(extra_ro)
        + '"], "allowUnixSockets": ["/run/docker.sock"], "networkPolicy": {"default": "deny"}}',
        encoding="utf-8",
    )
    (root / ".orbweaver" / "sandbox.json").write_text(
        '{"additionalReadwritePaths": ["'
        + str(extra_rw)
        + '"], "networkPolicy": {"default": "allow", "allow": ["example.com"]}}',
        encoding="utf-8",
    )
    policy = load_sandbox_policy(root, environ={}, home=home)
    assert extra_ro.resolve() in policy.additional_readonly
    assert extra_rw.resolve() in policy.additional_readwrite
    assert extra_ro.resolve() in policy.working_set_roots(root)
    assert extra_rw.resolve() in policy.readwrite_roots(root)
    assert extra_ro.resolve() not in policy.readwrite_roots(root)
    assert policy.network.default == "allow"
    assert "example.com" in policy.network.allow
    assert not any(p.name == "docker.sock" for p in policy.allow_unix_sockets)


def test_env_and_config_file_union(tmp_path: Path, monkeypatch):
    root = tmp_path / "ws"
    root.mkdir()
    cfg = tmp_path / "gateway.json"
    sock = tmp_path / "pg.sock"
    cfg.write_text(
        '{"allowUnixSockets": ["' + str(sock) + '"], "networkPolicy": {"includeDefaults": false}}',
        encoding="utf-8",
    )
    policy = load_sandbox_policy(
        root,
        environ={
            "ORBWEAVER_SANDBOX_CONFIG": str(cfg),
            "ORBWEAVER_SANDBOX_ALLOWED_DOMAINS": "crates.io,github.com",
            "ORBWEAVER_SANDBOX_NETWORK_DEFAULT": "deny",
        },
        home=tmp_path / "nohome",
    )
    assert sock.resolve() in policy.allow_unix_sockets
    assert policy.network.include_defaults is False
    assert "crates.io" in policy.network.allow
    assert "github.com" in policy.network.allowed_hosts()
    assert "pypi.org" not in policy.network.allowed_hosts()


def test_env_allow_from_sandbox_json_and_env(tmp_path: Path):
    home = tmp_path / "home"
    root = tmp_path / "ws"
    (home / ".orbweaver").mkdir(parents=True)
    (root / ".orbweaver").mkdir(parents=True)
    (home / ".orbweaver" / "sandbox.json").write_text('{"env": {"allow": ["PGHOST", "PG*"]}}', encoding="utf-8")
    (root / ".orbweaver" / "sandbox.json").write_text('{"env": {"allow": ["CARGO_HOME"]}}', encoding="utf-8")
    policy = load_sandbox_policy(
        root,
        environ={"ORBWEAVER_SANDBOX_ENV_ALLOW": "GOPATH,PGHOST"},
        home=home,
    )
    assert policy.env_allow == ("PGHOST", "PG*", "CARGO_HOME", "GOPATH")


def test_include_defaults_on(tmp_path: Path):
    policy = load_sandbox_policy(tmp_path, environ={}, home=tmp_path / "h")
    hosts = policy.network.allowed_hosts()
    assert "pypi.org" in hosts
    assert "github.com" in hosts
    assert set(DEFAULT_ALLOWED_DOMAINS).issubset(set(hosts))
