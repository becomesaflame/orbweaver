"""Egress policy for host-side web tools (WebFetch, Browser).

Sandboxed Bash reaches the network through :mod:`orbweaver.sandbox.proxy`, which
applies the domain allowlist and refuses loopback / LAN / link-local targets.
WebFetch and the Browser tool run on the host, outside bubblewrap, so they need
the same answer from a different entry point. :func:`check_url` gives it: parse
the URL, apply the domain patterns from the session's :class:`NetworkPolicy`,
resolve the host, and keep only addresses that pass :func:`ip_is_blocked`.

Callers connect to the vetted addresses (not to the hostname again) and re-run
the check on every redirect hop so a public URL cannot bounce to
``http://169.254.169.254/`` or ``http://localhost:8080/``.
"""

from __future__ import annotations

import asyncio
import ipaddress
import socket
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

from orbweaver.sandbox.errors import web_egress_denied
from orbweaver.sandbox.policy import NetworkPolicy, load_sandbox_policy
from orbweaver.sandbox.proxy import host_matches, ip_is_blocked

WEB_SCHEMES = frozenset({"http", "https"})
MAX_REDIRECTS = 5

Resolver = Callable[[str, int], list[str]]


class EgressDenied(PermissionError):
    """Raised when a URL fails the egress policy. ``str(exc)`` is the tool-facing text."""


@dataclass(frozen=True)
class EgressDecision:
    allowed: bool
    url: str
    scheme: str = ""
    host: str = ""
    port: int = 0
    addresses: tuple[str, ...] = ()
    reason: str = ""

    @property
    def message(self) -> str:
        return web_egress_denied(self.reason or f"{self.host or self.url} not allowed")

    def raise_if_denied(self) -> EgressDecision:
        if not self.allowed:
            raise EgressDenied(self.message)
        return self


def web_domain_allowed(hostname: str, network: NetworkPolicy) -> bool:
    """Domain check for host-side web tools.

    ``deny`` patterns win, ``allow`` patterns (plus the package-manager defaults
    when ``includeDefaults`` is on) grant, otherwise fall back to ``webDefault``
    (or ``default`` when that is already ``allow``).
    """
    host = (hostname or "").split("%")[0].strip().lower().rstrip(".")
    if not host:
        return False
    for pat in network.deny:
        if host_matches(host, pat):
            return False
    for pat in network.allowed_hosts():
        if host_matches(host, pat):
            return True
    return network.default == "allow" or network.web_default == "allow"


def _literal_ip(host: str) -> str | None:
    try:
        return str(ipaddress.ip_address(host.split("%")[0]))
    except ValueError:
        return None


def _split_url(url: str) -> tuple[str, str, int] | str:
    """Return ``(scheme, host, port)`` or an error string."""
    try:
        parts = urlsplit((url or "").strip())
    except ValueError as e:
        return f"unparseable URL: {e}"
    scheme = (parts.scheme or "").lower()
    if scheme not in WEB_SCHEMES:
        return f"unsupported URL scheme: {scheme or '(none)'}"
    try:
        host = (parts.hostname or "").strip().lower().rstrip(".")
        port = parts.port
    except ValueError as e:
        return f"invalid host or port: {e}"
    if not host:
        return "URL has no host"
    if port is None:
        port = 443 if scheme == "https" else 80
    return scheme, host, port


def default_resolver(host: str, port: int) -> list[str]:
    infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    out: list[str] = []
    for _fam, _kind, _proto, _canon, sockaddr in infos:
        ip = str(sockaddr[0])
        if ip not in out:
            out.append(ip)
    return out


def _decide(
    url: str,
    addresses: list[str] | None,
    resolve_error: str | None,
    parsed: tuple[str, str, int],
) -> EgressDecision:
    scheme, host, port = parsed
    if resolve_error is not None:
        return EgressDecision(
            False, url, scheme, host, port, reason=f"{host} did not resolve: {resolve_error}"
        )
    good = tuple(a for a in (addresses or []) if not ip_is_blocked(a))
    if not good:
        return EgressDecision(
            False, url, scheme, host, port, reason=f"{host} resolved to a blocked address"
        )
    return EgressDecision(True, url, scheme, host, port, addresses=good)


def _pre_resolve(url: str, network: NetworkPolicy) -> EgressDecision | tuple[str, str, int]:
    parsed = _split_url(url)
    if isinstance(parsed, str):
        return EgressDecision(False, url, reason=f"{url[:200]} ({parsed})")
    scheme, host, port = parsed
    literal = _literal_ip(host)
    if literal is not None:
        if ip_is_blocked(literal):
            return EgressDecision(False, url, scheme, host, port, reason=f"{host} is a blocked address")
        if not web_domain_allowed(host, network):
            return EgressDecision(False, url, scheme, host, port, reason=f"{host} not allowed")
        return EgressDecision(True, url, scheme, host, port, addresses=(literal,))
    if not web_domain_allowed(host, network):
        return EgressDecision(False, url, scheme, host, port, reason=f"{host} not allowed")
    return parsed


def check_url(url: str, network: NetworkPolicy, *, resolver: Resolver | None = None) -> EgressDecision:
    """Synchronous allow/deny for ``url`` under ``network``. Resolves DNS."""
    pre = _pre_resolve(url, network)
    if isinstance(pre, EgressDecision):
        return pre
    resolve = resolver or default_resolver
    try:
        addrs = resolve(pre[1], pre[2])
    except (socket.gaierror, OSError, UnicodeError) as e:
        return _decide(url, None, str(e) or "resolve failed", pre)
    return _decide(url, addrs, None, pre)


async def check_url_async(
    url: str, network: NetworkPolicy, *, resolver: Resolver | None = None
) -> EgressDecision:
    """Async variant: DNS runs in the default executor so the event loop stays free."""
    pre = _pre_resolve(url, network)
    if isinstance(pre, EgressDecision):
        return pre
    resolve = resolver or default_resolver
    loop = asyncio.get_running_loop()
    try:
        addrs = await loop.run_in_executor(None, resolve, pre[1], pre[2])
    except (socket.gaierror, OSError, UnicodeError) as e:
        return _decide(url, None, str(e) or "resolve failed", pre)
    return _decide(url, addrs, None, pre)


def network_policy_for(workspace: Any) -> NetworkPolicy:
    """The session's NetworkPolicy, loaded the same way sandboxed Bash loads it."""
    root = getattr(workspace, "root", None)
    try:
        return load_sandbox_policy(root).network
    except Exception:
        return NetworkPolicy()
