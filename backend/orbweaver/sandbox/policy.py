"""Sandbox policy: extra roots, denyRead, Unix sockets, domain allowlist."""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from orbweaver.sandbox.domains import DEFAULT_ALLOWED_DOMAINS
from orbweaver.sandbox.ssh import ssh_agent_socket

HARDCODED_DENY_READ: tuple[str, ...] = (
    "~/.ssh",
    "~/.gnupg",
    "~/.aws",
)

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


def expand_path(raw: str, *, relative_to: Path | None = None) -> Path:
    text = (raw or "").strip()
    if not text:
        raise ValueError("empty path")
    if text.startswith("~/"):
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


@dataclass(frozen=True)
class NetworkPolicy:
    default: str = "deny"
    include_defaults: bool = True
    allow: tuple[str, ...] = ()
    deny: tuple[str, ...] = ()

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
    allow_unix_sockets: tuple[Path, ...] = ()
    network: NetworkPolicy = field(default_factory=NetworkPolicy)

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
    )


def _merge_file(
    policy: SandboxPolicy,
    data: dict[str, Any],
    *,
    relative_to: Path,
) -> SandboxPolicy:
    def add_paths(current: tuple[Path, ...], key: str) -> tuple[Path, ...]:
        extra = data.get(key) or []
        if isinstance(extra, str):
            extra = _parse_list(extra)
        paths = list(current)
        for item in extra:
            try:
                paths.append(expand_path(str(item), relative_to=relative_to))
            except ValueError:
                continue
        return tuple(_unique_paths(paths))

    network = policy.network
    net_raw = data.get("networkPolicy") or data.get("network_policy")
    if isinstance(net_raw, dict):
        network = _merge_network(network, net_raw)
    return replace(
        policy,
        additional_readonly=add_paths(policy.additional_readonly, "additionalReadonlyPaths"),
        additional_readwrite=add_paths(policy.additional_readwrite, "additionalReadwritePaths"),
        deny_read=add_paths(policy.deny_read, "denyRead"),
        allow_unix_sockets=add_paths(policy.allow_unix_sockets, "allowUnixSockets"),
        network=network,
    )


def _filter_sockets(paths: Iterable[Path]) -> tuple[Path, ...]:
    out: list[Path] = []
    for p in paths:
        if is_hardcoded_socket_deny(p):
            continue
        out.append(p)
    return tuple(_unique_paths(out))


def _with_hardcoded_denies(
    policy: SandboxPolicy,
    home: Path | None = None,
    environ: dict[str, str] | None = None,
) -> SandboxPolicy:
    deny = list(policy.deny_read)
    home_dir = (home or Path.home()).resolve()
    for raw in HARDCODED_DENY_READ:
        if raw.startswith("~/"):
            deny.append((home_dir / raw[2:]).resolve())
        else:
            deny.append(expand_path(raw, relative_to=home_dir))
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
    sources: list[tuple[Path, Path]] = []
    user_file = home_dir / ".orbweaver" / "sandbox.json"
    repo_file = root / ".orbweaver" / "sandbox.json"
    sources.append((user_file, home_dir / ".orbweaver"))
    sources.append((repo_file, root))
    extra = (getattr(cfg, "orbweaver_sandbox_config", None) or env.get("ORBWEAVER_SANDBOX_CONFIG") or "").strip()
    if extra:
        extra_path = expand_path(extra, relative_to=root)
        sources.append((extra_path, extra_path.parent))

    for path, rel in sources:
        if path.is_file():
            policy = _merge_file(policy, _read_json(path), relative_to=rel)

    def env_list(key: str, attr: str) -> list[str]:
        raw = env.get(key, "")
        if not raw:
            raw = str(getattr(cfg, attr, "") or "")
        return _parse_list(raw)

    extra_ro = env_list("ORBWEAVER_SANDBOX_ADDITIONAL_READONLY", "orbweaver_sandbox_additional_readonly")
    extra_rw = env_list("ORBWEAVER_SANDBOX_ADDITIONAL_READWRITE", "orbweaver_sandbox_additional_readwrite")
    extra_deny = env_list("ORBWEAVER_SANDBOX_DENY_READ", "orbweaver_sandbox_deny_read")
    extra_socks = env_list("ORBWEAVER_SANDBOX_UNIX_SOCKETS", "orbweaver_sandbox_unix_sockets")
    extra_allow = env_list("ORBWEAVER_SANDBOX_ALLOWED_DOMAINS", "orbweaver_sandbox_allowed_domains")
    extra_deny_dom = env_list("ORBWEAVER_SANDBOX_DENIED_DOMAINS", "orbweaver_sandbox_denied_domains")

    paths_ro = list(policy.additional_readonly)
    paths_rw = list(policy.additional_readwrite)
    paths_deny = list(policy.deny_read)
    paths_sock = list(policy.allow_unix_sockets)
    for item in extra_ro:
        paths_ro.append(expand_path(item, relative_to=root))
    for item in extra_rw:
        paths_rw.append(expand_path(item, relative_to=root))
    for item in extra_deny:
        paths_deny.append(expand_path(item, relative_to=root))
    for item in extra_socks:
        paths_sock.append(expand_path(item, relative_to=root))

    net_default = (
        env.get("ORBWEAVER_SANDBOX_NETWORK_DEFAULT")
        or getattr(cfg, "orbweaver_sandbox_network_default", "")
        or policy.network.default
    ).strip().lower() or "deny"
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

    policy = SandboxPolicy(
        additional_readonly=tuple(_unique_paths(paths_ro)),
        additional_readwrite=tuple(_unique_paths(paths_rw)),
        deny_read=tuple(_unique_paths(paths_deny)),
        allow_unix_sockets=tuple(_unique_paths(paths_sock)),
        network=NetworkPolicy(
            default=net_default if net_default in {"allow", "deny"} else "deny",
            include_defaults=include_defaults,
            allow=tuple(allow),
            deny=tuple(deny_dom),
        ),
    )
    return _with_hardcoded_denies(policy, home=home_dir, environ=env)
