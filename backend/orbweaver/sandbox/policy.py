"""Sandbox policy: extra roots, denyRead, Unix sockets, domain allowlist."""

from __future__ import annotations

import fnmatch
import json
import os
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from orbweaver.sandbox.domains import DEFAULT_ALLOWED_DOMAINS
from orbweaver.sandbox.ssh import SshPolicy, ssh_agent_socket

# Credential files and directories hidden from sandboxed Bash by default. Entries
# are home-relative; a trailing glob (``credentials*``) matches several files.
# Operators put an entry back with ``allowRead`` in ``~/.orbweaver/sandbox.json``
# (or ``ORBWEAVER_SANDBOX_CONFIG`` / ``ORBWEAVER_SANDBOX_ALLOW_READ``). The
# workspace ``.orbweaver/sandbox.json`` is agent-writable, so it cannot.
DEFAULT_DENY_READ: tuple[str, ...] = (
    "~/.ssh",
    "~/.gnupg",
    "~/.aws",
    "~/.netrc",
    "~/.git-credentials",
    "~/.config/gh",
    "~/.config/hub",
    "~/.config/gcloud",
    "~/.docker/config.json",
    "~/.kube",
    "~/.azure",
    "~/.npmrc",
    "~/.pypirc",
    "~/.cargo/credentials*",
)
# Backwards-compatible alias; the list is overridable per entry via allowRead.
HARDCODED_DENY_READ = DEFAULT_DENY_READ

# ``.env`` files that are templates, not secrets.
_ENV_TEMPLATE_SUFFIXES: tuple[str, ...] = (".example", ".sample", ".template", ".dist")
_ENV_SCAN_PRUNE: frozenset[str] = frozenset(
    {".git", ".venv", "venv", "node_modules", "__pycache__", ".orbweaver-tmp", "dist", "build"}
)
_ENV_SCAN_MAX_DEPTH = 3

HARDCODED_UNIX_SOCKET_DENY: tuple[str, ...] = (
    "/var/run/docker.sock",
    "/run/docker.sock",
)


def default_journal_sockets(uid: int | None = None) -> tuple[Path, ...]:
    """System journal plus this uid's user journal. `/run` is tmpfs-hidden otherwise."""
    user_id = os.getuid() if uid is None else uid
    return (
        Path("/run/systemd/journal/socket"),
        Path(f"/run/user/{user_id}/systemd/journal/socket"),
    )


PROTECTED_WRITE_REL: tuple[str, ...] = (
    ".orbweaver/sandbox.json",
    ".orbweaver/mcp.json",
)


def _parse_list(raw: str) -> list[str]:
    text = (raw or "").strip()
    if not text:
        return []
    if text.startswith("["):
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return [x.strip() for x in text.split(",") if x.strip()]
        if isinstance(data, list):
            return [str(x).strip() for x in data if str(x).strip()]
        return []
    return [x.strip() for x in text.split(",") if x.strip()]


def expand_path(raw: str, *, relative_to: Path | None = None, home: Path | None = None) -> Path:
    text = (raw or "").strip()
    if not text:
        raise ValueError("empty path")
    if text.startswith("~/"):
        if home is not None:
            return (home / text[2:]).resolve()
        return Path(text).expanduser().resolve()
    if text.startswith("/") or (len(text) > 1 and text[1] == ":"):
        return Path(text).resolve()
    base = relative_to or Path.cwd()
    return (base / text).resolve()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def gateway_checkout_root() -> Path | None:
    """Checkout that contains the running package (``<root>/backend/orbweaver``)."""
    import orbweaver

    file = getattr(orbweaver, "__file__", None)
    if not file:
        return None
    try:
        pkg = Path(str(file)).resolve().parent
    except OSError:
        return None
    root = pkg.parent.parent
    if (root / "backend" / "orbweaver").is_dir():
        return root
    return None


def is_env_secret_name(name: str) -> bool:
    """``.env``, ``.env.local``, ``prod.env`` ... but not ``.env.example``."""
    lowered = name.lower()
    if lowered.endswith(_ENV_TEMPLATE_SUFFIXES):
        return False
    return lowered.startswith(".env") or lowered.endswith(".env")


def gateway_env_files(root: Path | None = None) -> tuple[Path, ...]:
    """``backend/.env`` and every other ``.env*`` file under the gateway checkout."""
    checkout = root if root is not None else gateway_checkout_root()
    if checkout is None:
        return ()
    found: list[Path] = []
    base_depth = len(checkout.parts)
    try:
        for dirpath, dirnames, filenames in os.walk(checkout):
            here = Path(dirpath)
            depth = len(here.parts) - base_depth
            dirnames[:] = sorted(
                d for d in dirnames if d not in _ENV_SCAN_PRUNE and depth < _ENV_SCAN_MAX_DEPTH
            )
            for name in filenames:
                if is_env_secret_name(name):
                    found.append(here / name)
    except OSError:
        pass
    backend_env = checkout / "backend" / ".env"
    if backend_env not in found:
        found.insert(0, backend_env)
    return tuple(_unique_paths(found))


def _split_default_entry(raw: str) -> tuple[str, ...]:
    text = raw[2:] if raw.startswith("~/") else raw.lstrip("/")
    return tuple(p for p in text.split("/") if p)


def expand_default_deny(raw: str, home: Path) -> tuple[Path, ...]:
    """Expand one ``DEFAULT_DENY_READ`` entry (home-relative, optional trailing glob)."""
    parts = _split_default_entry(raw)
    if not parts:
        return ()
    if any(ch in raw for ch in "*?["):
        parent = home.joinpath(*parts[:-1])
        try:
            return tuple(sorted(p for p in parent.glob(parts[-1]) if not p.name.startswith("..")))
        except OSError:
            return ()
    return (home.joinpath(*parts),)


def _strip_home_prefix(path: Path) -> tuple[str, ...] | None:
    """Path parts below a home directory (``~``, ``/home/<user>``, ``/root``) or None."""
    parts = path.parts
    try:
        home_parts = Path.home().resolve().parts
    except (OSError, RuntimeError):
        home_parts = ()
    if home_parts and parts[: len(home_parts)] == home_parts:
        return parts[len(home_parts) :]
    if len(parts) >= 3 and parts[0] == "/" and parts[1] == "home":
        return parts[3:]
    if len(parts) >= 2 and parts[0] == "/" and parts[1] == "root":
        return parts[2:]
    return None


def is_default_denied_read(raw_path: str) -> bool:
    """True when an absolute or ``~`` path names a ``DEFAULT_DENY_READ`` entry or a child.

    Shared by the Read/Grep/Glob permission rules and the bubblewrap deny overlays so the
    two layers agree. Workspace-relative paths never match: the defaults are home-relative.
    """
    text = (raw_path or "").strip()
    if not text:
        return False
    if text.startswith("~"):
        text = os.path.expanduser(text)
    if not text.startswith("/"):
        return False
    rel = _strip_home_prefix(Path(os.path.normpath(text)))
    if not rel:
        return False
    for raw in DEFAULT_DENY_READ:
        entry = _split_default_entry(raw)
        if len(entry) > len(rel):
            continue
        if all(fnmatch.fnmatchcase(got, want) for got, want in zip(rel, entry, strict=False)):
            return True
    return False


@dataclass(frozen=True)
class NetworkPolicy:
    default: str = "deny"
    include_defaults: bool = True
    allow: tuple[str, ...] = ()
    deny: tuple[str, ...] = ()
    # Default for host-side web tools (WebFetch, Browser) when no allow/deny
    # pattern matches. "allow" keeps the public web reachable; the IP blocklist
    # (loopback, LAN, link-local/metadata) and `deny` patterns always apply.
    web_default: str = "allow"

    def allowed_hosts(self) -> tuple[str, ...]:
        hosts = list(self.allow)
        if self.include_defaults:
            hosts.extend(DEFAULT_ALLOWED_DOMAINS)
        # preserve order, drop dupes
        seen: set[str] = set()
        out: list[str] = []
        for h in hosts:
            key = h.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(h)
        return tuple(out)


@dataclass(frozen=True)
class SandboxPolicy:
    additional_readonly: tuple[Path, ...] = ()
    additional_readwrite: tuple[Path, ...] = ()
    deny_read: tuple[Path, ...] = ()
    # Default deny entries an operator explicitly put back (never from the workspace file).
    allow_read: tuple[Path, ...] = ()
    allow_unix_sockets: tuple[Path, ...] = ()
    network: NetworkPolicy = field(default_factory=NetworkPolicy)
    ssh: SshPolicy = field(default_factory=SshPolicy)
    # Keep .git/config writable inside the sandbox for `git remote set-url`
    # workflows. .git/hooks and .gitmodules stay read-only either way (#94).
    allow_git_config: bool = False
    # Extra environment variable names/globs passed into the sandbox on top of
    # DEFAULT_ENV_ALLOW. DEFAULT_ENV_EXCLUDE still wins (sandbox.json `env.allow`).
    env_allow: tuple[str, ...] = ()

    def readwrite_roots(self, workspace_root: Path, tmp_dir: Path | None = None) -> tuple[Path, ...]:
        roots = [workspace_root.resolve()]
        roots.extend(p.resolve() for p in self.additional_readwrite)
        if tmp_dir is not None:
            roots.append(tmp_dir.resolve())
        return tuple(_unique_paths(roots))

    def readonly_roots(self) -> tuple[Path, ...]:
        return tuple(_unique_paths(p.resolve() for p in self.additional_readonly))

    def working_set_roots(self, workspace_root: Path) -> tuple[Path, ...]:
        roots = [workspace_root.resolve()]
        roots.extend(self.additional_readonly)
        roots.extend(self.additional_readwrite)
        return tuple(_unique_paths(p.resolve() for p in roots))


def _unique_paths(paths: Iterable[Path]) -> list[Path]:
    seen: set[Path] = set()
    out: list[Path] = []
    for p in paths:
        if p in seen:
            continue
        seen.add(p)
        out.append(p)
    return out


def _path_under(path: Path, root: Path) -> bool:
    try:
        path = path.resolve()
        root = root.resolve()
    except OSError:
        return False
    return root == path or root in path.parents


def in_roots(path: Path, roots: Iterable[Path]) -> bool:
    return any(_path_under(path, root) for root in roots)


def is_hardcoded_socket_deny(path: Path) -> bool:
    resolved = str(path.expanduser().resolve()) if path.exists() else str(Path(path).expanduser())
    for raw in HARDCODED_UNIX_SOCKET_DENY:
        deny = str(Path(raw))
        if resolved == deny or resolved.endswith(deny):
            return True
        if Path(raw).name and path.name == Path(raw).name and path.name == "docker.sock":
            return True
    return path.name == "docker.sock"


def _merge_network(base: NetworkPolicy, overlay: dict[str, Any]) -> NetworkPolicy:
    default = str(overlay.get("default") or base.default).strip().lower() or base.default
    if default not in {"allow", "deny"}:
        default = base.default
    web_default = (
        str(overlay.get("webDefault") or overlay.get("web_default") or base.web_default).strip().lower()
        or base.web_default
    )
    if web_default not in {"allow", "deny"}:
        web_default = base.web_default
    include = overlay.get("includeDefaults", overlay.get("include_defaults", base.include_defaults))
    if isinstance(include, str):
        include = include.strip().lower() in {"1", "true", "yes"}
    include = bool(include)
    allow = list(base.allow)
    deny = list(base.deny)
    for key, bucket in (("allow", allow), ("deny", deny)):
        extra = overlay.get(key) or []
        if isinstance(extra, str):
            extra = _parse_list(extra)
        for item in extra:
            text = str(item).strip()
            if text and text not in bucket:
                bucket.append(text)
    # deny wins over allow at evaluation time; keep both lists
    return NetworkPolicy(
        default=default,
        include_defaults=include,
        allow=tuple(allow),
        deny=tuple(deny),
        web_default=web_default,
    )


def _as_bool(raw: Any, default: bool) -> bool:
    if raw is None:
        return default
    if isinstance(raw, str):
        return raw.strip().lower() in {"1", "true", "yes"}
    return bool(raw)


def _merge_ssh(
    base: SshPolicy,
    overlay: dict[str, Any],
    *,
    relative_to: Path,
    home: Path | None,
) -> SshPolicy:
    identities = list(base.identities)
    extra = overlay.get("identities") or overlay.get("identityFiles") or []
    if isinstance(extra, str):
        extra = _parse_list(extra)
    for item in extra:
        try:
            identities.append(expand_path(str(item), relative_to=relative_to, home=home))
        except ValueError:
            continue
    bind_raw = overlay.get("bindIdentities", overlay.get("bind_identities"))
    return SshPolicy(
        identities=tuple(_unique_paths(identities)),
        bind_identities=_as_bool(bind_raw, base.bind_identities),
    )


def _merge_file(
    policy: SandboxPolicy,
    data: dict[str, Any],
    *,
    relative_to: Path,
    operator: bool = True,
    home: Path | None = None,
) -> SandboxPolicy:
    """Union one sandbox.json into ``policy``.

    ``allowRead`` and ``ssh`` widen what the sandbox exposes, so they are honoured only
    when ``operator`` is true (user file, ``ORBWEAVER_SANDBOX_CONFIG``) and ignored in the
    agent-writable workspace ``.orbweaver/sandbox.json``.
    """

    def add_paths(current: tuple[Path, ...], key: str) -> tuple[Path, ...]:
        extra = data.get(key) or []
        if isinstance(extra, str):
            extra = _parse_list(extra)
        paths = list(current)
        for item in extra:
            try:
                paths.append(expand_path(str(item), relative_to=relative_to, home=home))
            except ValueError:
                continue
        return tuple(_unique_paths(paths))

    network = policy.network
    net_raw = data.get("networkPolicy") or data.get("network_policy")
    if isinstance(net_raw, dict):
        network = _merge_network(network, net_raw)
    allow_read = policy.allow_read
    ssh = policy.ssh
    if operator:
        allow_read = add_paths(policy.allow_read, "allowRead")
        ssh_raw = data.get("ssh")
        if isinstance(ssh_raw, dict):
            ssh = _merge_ssh(ssh, ssh_raw, relative_to=relative_to, home=home)
    allow_git_config = policy.allow_git_config
    git_raw = data.get("allowGitConfig", data.get("allow_git_config"))
    if git_raw is not None:
        if isinstance(git_raw, str):
            allow_git_config = git_raw.strip().lower() in {"1", "true", "yes"}
        else:
            allow_git_config = bool(git_raw)
    env_allow = list(policy.env_allow)
    env_raw = data.get("env")
    if isinstance(env_raw, dict):
        env_allow = _merge_names(env_allow, env_raw.get("allow"))
    return replace(
        policy,
        additional_readonly=add_paths(policy.additional_readonly, "additionalReadonlyPaths"),
        additional_readwrite=add_paths(policy.additional_readwrite, "additionalReadwritePaths"),
        deny_read=add_paths(policy.deny_read, "denyRead"),
        allow_read=allow_read,
        allow_unix_sockets=add_paths(policy.allow_unix_sockets, "allowUnixSockets"),
        network=network,
        ssh=ssh,
        allow_git_config=allow_git_config,
        env_allow=tuple(env_allow),
    )


def _merge_names(current: list[str], extra: Any) -> list[str]:
    items = extra or []
    if isinstance(items, str):
        items = _parse_list(items)
    if not isinstance(items, list):
        return current
    out = list(current)
    for item in items:
        text = str(item).strip()
        if text and text not in out:
            out.append(text)
    return out


def _filter_sockets(paths: Iterable[Path]) -> tuple[Path, ...]:
    out: list[Path] = []
    for p in paths:
        if is_hardcoded_socket_deny(p):
            continue
        out.append(p)
    return tuple(_unique_paths(out))


def default_deny_read_paths(
    home: Path | None = None,
    *,
    gateway_root: Path | None = None,
) -> tuple[Path, ...]:
    """Every ``DEFAULT_DENY_READ`` entry plus the gateway checkout's ``.env*`` files."""
    home_dir = (home or Path.home()).resolve()
    out: list[Path] = []
    for raw in DEFAULT_DENY_READ:
        out.extend(expand_default_deny(raw, home_dir))
    out.extend(gateway_env_files(gateway_root))
    return tuple(_unique_paths(out))


def _with_hardcoded_denies(
    policy: SandboxPolicy,
    home: Path | None = None,
    environ: dict[str, str] | None = None,
) -> SandboxPolicy:
    deny = list(policy.deny_read)
    home_dir = (home or Path.home()).resolve()
    allowed = {p.resolve() for p in policy.allow_read}
    for path in default_deny_read_paths(home_dir):
        if path.resolve() in allowed:
            continue
        deny.append(path)
    socks = list(policy.allow_unix_sockets)
    socks.extend(default_journal_sockets())
    socks.extend(ssh_agent_socket(environ))
    return replace(
        policy,
        deny_read=tuple(_unique_paths(deny)),
        allow_unix_sockets=_filter_sockets(socks),
    )


def load_sandbox_policy(
    workspace_root: Path | str | None = None,
    *,
    settings=None,
    environ: dict[str, str] | None = None,
    home: Path | None = None,
) -> SandboxPolicy:
    """Merge user, repo, ORBWEAVER_SANDBOX_CONFIG, then env. Hardcoded denies always win."""
    from orbweaver.config import settings as default_settings

    cfg = settings or default_settings
    env = environ if environ is not None else dict(os.environ)
    root = Path(workspace_root or getattr(cfg, "workspace_root", ".") or ".").resolve()
    home_dir = (home or Path.home()).resolve()

    policy = SandboxPolicy()
    sources: list[tuple[Path, Path, bool]] = []
    user_file = home_dir / ".orbweaver" / "sandbox.json"
    repo_file = root / ".orbweaver" / "sandbox.json"
    sources.append((user_file, home_dir / ".orbweaver", True))
    # The workspace file is agent-writable: it may narrow the sandbox, never widen it.
    sources.append((repo_file, root, False))
    extra = (getattr(cfg, "orbweaver_sandbox_config", None) or env.get("ORBWEAVER_SANDBOX_CONFIG") or "").strip()
    if extra:
        extra_path = expand_path(extra, relative_to=root)
        sources.append((extra_path, extra_path.parent, True))

    for path, rel, operator in sources:
        if path.is_file():
            policy = _merge_file(
                policy, _read_json(path), relative_to=rel, operator=operator, home=home_dir
            )

    def env_list(key: str, attr: str) -> list[str]:
        raw = env.get(key, "")
        if not raw:
            raw = str(getattr(cfg, attr, "") or "")
        return _parse_list(raw)

    extra_ro = env_list("ORBWEAVER_SANDBOX_ADDITIONAL_READONLY", "orbweaver_sandbox_additional_readonly")
    extra_rw = env_list("ORBWEAVER_SANDBOX_ADDITIONAL_READWRITE", "orbweaver_sandbox_additional_readwrite")
    extra_deny = env_list("ORBWEAVER_SANDBOX_DENY_READ", "orbweaver_sandbox_deny_read")
    extra_allow_read = env_list("ORBWEAVER_SANDBOX_ALLOW_READ", "orbweaver_sandbox_allow_read")
    extra_socks = env_list("ORBWEAVER_SANDBOX_UNIX_SOCKETS", "orbweaver_sandbox_unix_sockets")
    extra_allow = env_list("ORBWEAVER_SANDBOX_ALLOWED_DOMAINS", "orbweaver_sandbox_allowed_domains")
    extra_deny_dom = env_list("ORBWEAVER_SANDBOX_DENIED_DOMAINS", "orbweaver_sandbox_denied_domains")
    extra_ssh_ids = env_list("ORBWEAVER_SANDBOX_SSH_IDENTITIES", "orbweaver_sandbox_ssh_identities")
    bind_ids_raw = env.get("ORBWEAVER_SANDBOX_SSH_BIND_IDENTITIES")
    if bind_ids_raw is None or bind_ids_raw == "":
        bind_ids_raw = getattr(cfg, "orbweaver_sandbox_ssh_bind_identities", "")
    extra_env_allow = env_list("ORBWEAVER_SANDBOX_ENV_ALLOW", "orbweaver_sandbox_env_allow")

    paths_ro = list(policy.additional_readonly)
    paths_rw = list(policy.additional_readwrite)
    paths_deny = list(policy.deny_read)
    paths_allow_read = list(policy.allow_read)
    paths_sock = list(policy.allow_unix_sockets)
    ssh_ids = list(policy.ssh.identities)
    for item in extra_ro:
        paths_ro.append(expand_path(item, relative_to=root))
    for item in extra_rw:
        paths_rw.append(expand_path(item, relative_to=root))
    for item in extra_deny:
        paths_deny.append(expand_path(item, relative_to=root))
    for item in extra_allow_read:
        paths_allow_read.append(expand_path(item, relative_to=home_dir, home=home_dir))
    for item in extra_socks:
        paths_sock.append(expand_path(item, relative_to=root))
    for item in extra_ssh_ids:
        ssh_ids.append(expand_path(item, relative_to=home_dir, home=home_dir))
    bind_identities = policy.ssh.bind_identities
    if str(bind_ids_raw).strip() != "":
        bind_identities = _as_bool(bind_ids_raw, bind_identities)

    net_default = (
        env.get("ORBWEAVER_SANDBOX_NETWORK_DEFAULT")
        or getattr(cfg, "orbweaver_sandbox_network_default", "")
        or policy.network.default
    ).strip().lower() or "deny"
    web_default = (
        env.get("ORBWEAVER_SANDBOX_WEB_NETWORK_DEFAULT")
        or getattr(cfg, "orbweaver_sandbox_web_network_default", "")
        or policy.network.web_default
    ).strip().lower() or "allow"
    include_raw = env.get("ORBWEAVER_SANDBOX_INCLUDE_DEFAULT_DOMAINS")
    if include_raw is None or include_raw == "":
        include_raw = getattr(cfg, "orbweaver_sandbox_include_default_domains", "")
    if str(include_raw).strip() == "":
        include_defaults = policy.network.include_defaults
    else:
        include_defaults = str(include_raw).strip().lower() in {"1", "true", "yes"}

    allow = list(policy.network.allow)
    deny_dom = list(policy.network.deny)
    for item in extra_allow:
        if item not in allow:
            allow.append(item)
    for item in extra_deny_dom:
        if item not in deny_dom:
            deny_dom.append(item)

    allow_git_config = policy.allow_git_config
    git_env = env.get("ORBWEAVER_SANDBOX_ALLOW_GIT_CONFIG")
    if git_env is None or git_env == "":
        git_env = str(getattr(cfg, "orbweaver_sandbox_allow_git_config", "") or "")
    if str(git_env).strip() != "":
        allow_git_config = str(git_env).strip().lower() in {"1", "true", "yes"}

    policy = SandboxPolicy(
        additional_readonly=tuple(_unique_paths(paths_ro)),
        additional_readwrite=tuple(_unique_paths(paths_rw)),
        deny_read=tuple(_unique_paths(paths_deny)),
        allow_read=tuple(_unique_paths(paths_allow_read)),
        allow_unix_sockets=tuple(_unique_paths(paths_sock)),
        network=NetworkPolicy(
            default=net_default if net_default in {"allow", "deny"} else "deny",
            include_defaults=include_defaults,
            allow=tuple(allow),
            deny=tuple(deny_dom),
            web_default=web_default if web_default in {"allow", "deny"} else "allow",
        ),
        ssh=SshPolicy(identities=tuple(_unique_paths(ssh_ids)), bind_identities=bind_identities),
        allow_git_config=allow_git_config,
        env_allow=tuple(_merge_names(list(policy.env_allow), extra_env_allow)),
    )
    policy = _with_hardcoded_denies(policy, home=home_dir, environ=env)
    deny = list(policy.deny_read)
    deny.append(home_dir / ".orbweaver" / "mcp.json")
    deny.append(root / ".orbweaver" / "mcp.json")
    return replace(policy, deny_read=tuple(_unique_paths(deny)))
