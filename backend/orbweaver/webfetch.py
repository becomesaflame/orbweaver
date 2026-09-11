"""WebFetch: policy-checked, address-pinned HTTP GET with body and type caps.

Every hop (the original URL and each ``Location``) goes through
:func:`orbweaver.sandbox.egress.check_url_async`. The request is then sent to the
vetted IP address with the original hostname in ``Host`` and TLS SNI, so a DNS
answer that changes between check and connect cannot land on a blocked target.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import httpx

from orbweaver.sandbox.egress import (
    MAX_REDIRECTS,
    EgressDecision,
    EgressDenied,
    Resolver,
    check_url_async,
    network_policy_for,
)
from orbweaver.sandbox.errors import web_egress_denied
from orbweaver.sandbox.policy import NetworkPolicy
from orbweaver.tooltext import format_webfetch

log = logging.getLogger(__name__)

FETCH_TIMEOUT = 20.0
MAX_BODY_BYTES = 2 * 1024 * 1024
MAX_ADDRESS_ATTEMPTS = 3
USER_AGENT = "orbweaver-webfetch/1.0 (+https://github.com/becomesaflame/orbweaver)"

_TEXT_APPLICATION_TYPES = frozenset(
    {
        "application/json",
        "application/ld+json",
        "application/x-ndjson",
        "application/xml",
        "application/xhtml+xml",
        "application/rss+xml",
        "application/atom+xml",
        "application/javascript",
        "application/x-javascript",
        "application/ecmascript",
        "application/yaml",
        "application/x-yaml",
        "application/toml",
        "application/x-sh",
        "application/x-www-form-urlencoded",
        "application/graphql",
        "application/sql",
    }
)


@dataclass(frozen=True)
class FetchResult:
    url: str
    final_url: str
    status_code: int
    content_type: str
    text: str
    truncated: bool = False
    skipped: str = ""


def fetchable_content_type(content_type: str) -> bool:
    """True for text-like media types WebFetch is willing to download."""
    media = (content_type or "").split(";", 1)[0].strip().lower()
    if not media:
        return True
    if media.startswith("text/"):
        return True
    if media in _TEXT_APPLICATION_TYPES:
        return True
    return media.startswith("application/") and media.endswith(("+json", "+xml"))


def _host_header(decision: EgressDecision) -> str:
    default = 443 if decision.scheme == "https" else 80
    host = decision.host
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    return host if decision.port == default else f"{host}:{decision.port}"


def pinned_request(client: httpx.AsyncClient, decision: EgressDecision, address: str) -> httpx.Request:
    """Build a GET to ``address`` that still speaks to ``decision.host`` (Host + SNI)."""
    url = httpx.URL(decision.url).copy_with(host=address, port=decision.port)
    headers = {"Host": _host_header(decision), "User-Agent": USER_AGENT}
    extensions: dict[str, Any] = {}
    if decision.scheme == "https":
        extensions["sni_hostname"] = decision.host
    return client.build_request("GET", url, headers=headers, extensions=extensions)


def _decode(body: bytes, response: httpx.Response) -> str:
    encoding = response.encoding or "utf-8"
    try:
        return body.decode(encoding, errors="replace")
    except LookupError:
        return body.decode("utf-8", errors="replace")


async def _send_pinned(
    client: httpx.AsyncClient, decision: EgressDecision
) -> httpx.Response:
    last: Exception | None = None
    for address in decision.addresses[:MAX_ADDRESS_ATTEMPTS]:
        request = pinned_request(client, decision, address)
        try:
            return await client.send(request, stream=True)
        except httpx.ConnectError as e:
            last = e
            continue
    assert last is not None
    raise last


async def fetch_url(
    url: str,
    network: NetworkPolicy,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
    resolver: Resolver | None = None,
    timeout: float = FETCH_TIMEOUT,
    max_bytes: int = MAX_BODY_BYTES,
) -> FetchResult:
    """GET ``url`` under ``network``. Raises :class:`EgressDenied` for any blocked hop."""
    current = (url or "").strip()
    async with httpx.AsyncClient(
        transport=transport, timeout=timeout, follow_redirects=False
    ) as client:
        for _hop in range(MAX_REDIRECTS + 1):
            decision = await check_url_async(current, network, resolver=resolver)
            decision.raise_if_denied()
            response = await _send_pinned(client, decision)
            try:
                if response.is_redirect and response.headers.get("location"):
                    location = response.headers["location"]
                    current = str(httpx.URL(current).join(location))
                    continue
                ctype = response.headers.get("content-type") or ""
                if not fetchable_content_type(ctype):
                    length = response.headers.get("content-length") or "unknown"
                    return FetchResult(
                        url,
                        current,
                        response.status_code,
                        ctype,
                        "",
                        skipped=f"content-type {ctype.split(';')[0].strip()!r} is not text "
                        f"({length} bytes not downloaded)",
                    )
                buf = bytearray()
                truncated = False
                async for chunk in response.aiter_bytes():
                    buf += chunk
                    if len(buf) >= max_bytes:
                        truncated = True
                        del buf[max_bytes:]
                        break
                return FetchResult(
                    url,
                    current,
                    response.status_code,
                    ctype,
                    _decode(bytes(buf), response),
                    truncated=truncated,
                )
            finally:
                await response.aclose()
    raise EgressDenied(web_egress_denied(f"{url[:200]} exceeded {MAX_REDIRECTS} redirects"))


async def run_webfetch(inp: dict[str, Any], ctx: dict[str, Any]) -> str:
    url = str(inp.get("url") or "").strip()
    network = network_policy_for(ctx.get("workspace"))
    try:
        result = await fetch_url(url, network)
    except EgressDenied as e:
        return str(e)
    except httpx.HTTPError as e:
        return f"WebFetch failed for {url[:200]}: {type(e).__name__}: {e}"
    body = result.text
    if result.skipped:
        body = f"[skipped: {result.skipped}]"
    elif result.truncated:
        body += f"\n\n[body truncated to {MAX_BODY_BYTES} bytes]"
    return format_webfetch(url, result.status_code, result.content_type, body)
