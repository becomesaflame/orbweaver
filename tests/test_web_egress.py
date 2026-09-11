"""WebFetch and Browser honour the sandbox egress policy (issue #96)."""

from __future__ import annotations

import http.server
import json
import threading
from pathlib import Path
from typing import ClassVar
from uuid import uuid4

import httpx
import pytest

from orbweaver.agent import run_tools
from orbweaver.browser import EgressGate, _Session, pool
from orbweaver.sandbox.egress import (
    EgressDenied,
    check_url,
    check_url_async,
    web_domain_allowed,
)
from orbweaver.sandbox.policy import NetworkPolicy, load_sandbox_policy
from orbweaver.sandbox.proxy import ip_is_blocked
from orbweaver.webfetch import fetch_url, fetchable_content_type, run_webfetch
from orbweaver.workspace import LocalWorkspace

PUBLIC_IP = "93.184.216.34"
PUBLIC_IP6 = "2606:2800:220:1:248:1893:25c8:1946"


class _Hits(http.server.BaseHTTPRequestHandler):
    hits: ClassVar[list[str]] = []

    def do_GET(self):
        type(self).hits.append(self.path)
        body = b"secret-from-loopback"
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_a):
        return


@pytest.fixture
def loopback_server():
    _Hits.hits = []
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Hits)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{srv.server_address[1]}"
    finally:
        srv.shutdown()
        srv.server_close()


def _resolver(table: dict[str, list[str]]):
    calls: list[str] = []

    def resolve(host: str, _port: int) -> list[str]:
        calls.append(host)
        if host not in table:
            raise OSError(f"fake resolver: {host} unknown")
        return list(table[host])

    resolve.calls = calls  # type: ignore[attr-defined]
    return resolve


def _ctx(tmp_path: Path) -> dict:
    return {
        "workspace": LocalWorkspace("workspace:default", str(tmp_path)),
        "workspace_kind": "local",
        "session_id": uuid4(),
        "store": None,
        "events": [],
    }


# --- ip_is_blocked -----------------------------------------------------------


@pytest.mark.parametrize(
    "addr",
    [
        "169.254.169.254",
        "169.254.0.1",
        "127.0.0.1",
        "127.9.8.7",
        "10.0.0.1",
        "10.255.255.255",
        "172.16.0.1",
        "172.31.255.254",
        "192.168.1.1",
        "0.0.0.0",
        "::1",
        "::",
        "fc00::1",
        "fdab::1",
        "fe80::1",
        "fe80::1%eth0",
        "::ffff:127.0.0.1",
        "::ffff:10.1.2.3",
        "100.64.0.1",
        "not-an-ip",
    ],
)
def test_ip_is_blocked_covers_local_ranges(addr):
    assert ip_is_blocked(addr)


@pytest.mark.parametrize("addr", ["1.1.1.1", PUBLIC_IP, "8.8.8.8", PUBLIC_IP6, "2001:4860:4860::8888"])
def test_public_ips_not_blocked(addr):
    assert not ip_is_blocked(addr)


# --- domain policy -----------------------------------------------------------


def test_web_domain_allowed_follows_sandbox_policy():
    default = NetworkPolicy()
    assert default.web_default == "allow"
    assert web_domain_allowed("docs.python.org", default)
    assert web_domain_allowed("pypi.org", default)
    assert not web_domain_allowed("evil.example", NetworkPolicy(deny=("*.example",)))

    strict = NetworkPolicy(default="deny", web_default="deny", include_defaults=False, allow=("*.example.com",))
    assert web_domain_allowed("docs.example.com", strict)
    assert web_domain_allowed("example.com", strict)
    assert not web_domain_allowed("docs.python.org", strict)
    assert not web_domain_allowed("", strict)

    bash_open = NetworkPolicy(default="allow", web_default="deny", include_defaults=False)
    assert web_domain_allowed("anything.example", bash_open)


def test_web_default_configurable_from_sandbox_json_and_env(tmp_path: Path):
    root = tmp_path / "ws"
    (root / ".orbweaver").mkdir(parents=True)
    (root / ".orbweaver" / "sandbox.json").write_text(
        json.dumps({"networkPolicy": {"webDefault": "deny", "allow": ["*.example.com"]}}),
        encoding="utf-8",
    )
    policy = load_sandbox_policy(root, environ={}, home=tmp_path / "h")
    assert policy.network.web_default == "deny"
    assert policy.network.default == "deny"
    assert web_domain_allowed("api.example.com", policy.network)
    assert not web_domain_allowed("docs.python.org", policy.network)

    policy = load_sandbox_policy(
        root, environ={"ORBWEAVER_SANDBOX_WEB_NETWORK_DEFAULT": "allow"}, home=tmp_path / "h"
    )
    assert policy.network.web_default == "allow"
    assert web_domain_allowed("docs.python.org", policy.network)

    policy = load_sandbox_policy(tmp_path / "empty", environ={}, home=tmp_path / "h")
    assert policy.network.web_default == "allow"


# --- check_url ---------------------------------------------------------------


def test_check_url_literal_blocked_ip_denied_without_resolving():
    resolve = _resolver({})
    net = NetworkPolicy()
    for url in (
        "http://169.254.169.254/latest/meta-data/",
        "http://127.0.0.1:8080/health",
        "http://10.3.0.213/",
        "http://[::1]/",
        "http://[fe80::1]/",
        "http://0.0.0.0/",
    ):
        decision = check_url(url, net, resolver=resolve)
        assert not decision.allowed, url
        assert "blocked address" in decision.reason
        assert "sandbox_denied: network" in decision.message
    assert resolve.calls == []


def test_check_url_resolved_blocked_and_allowed():
    resolve = _resolver(
        {
            "internal.example": ["10.1.2.3", "192.168.0.9"],
            "public.example": ["127.0.0.1", PUBLIC_IP],
            "v6.example": [PUBLIC_IP6],
        }
    )
    net = NetworkPolicy()
    denied = check_url("https://internal.example/x", net, resolver=resolve)
    assert not denied.allowed
    assert "resolved to a blocked address" in denied.reason

    ok = check_url("https://public.example/x", net, resolver=resolve)
    assert ok.allowed
    assert ok.addresses == (PUBLIC_IP,)  # loopback answer dropped, public kept
    assert ok.host == "public.example"
    assert ok.port == 443

    ok6 = check_url("http://v6.example:8080/", net, resolver=resolve)
    assert ok6.allowed and ok6.addresses == (PUBLIC_IP6,) and ok6.port == 8080

    nx = check_url("http://nx.example/", net, resolver=resolve)
    assert not nx.allowed and "did not resolve" in nx.reason


def test_check_url_domain_policy_before_dns():
    resolve = _resolver({"docs.python.org": [PUBLIC_IP]})
    strict = NetworkPolicy(web_default="deny", include_defaults=False, allow=("*.example.com",))
    denied = check_url("https://docs.python.org/3/", strict, resolver=resolve)
    assert not denied.allowed and "not allowed" in denied.reason
    assert resolve.calls == []
    denied = check_url("https://docs.python.org/3/", NetworkPolicy(deny=("*.python.org",)), resolver=resolve)
    assert not denied.allowed
    assert resolve.calls == []


def test_check_url_rejects_non_http():
    for url in ("ftp://example.com/", "file:///etc/passwd", "gopher://x", "", "http:///nohost"):
        assert not check_url(url, NetworkPolicy(), resolver=_resolver({})).allowed


@pytest.mark.asyncio
async def test_check_url_async_matches_sync():
    resolve = _resolver({"public.example": [PUBLIC_IP]})
    d = await check_url_async("https://public.example/", NetworkPolicy(), resolver=resolve)
    assert d.allowed and d.addresses == (PUBLIC_IP,)
    d = await check_url_async("http://127.0.0.1/", NetworkPolicy(), resolver=resolve)
    assert not d.allowed


# --- WebFetch ----------------------------------------------------------------


@pytest.mark.asyncio
async def test_webfetch_refuses_loopback_server(loopback_server, tmp_path):
    out = await run_webfetch({"url": f"{loopback_server}/secret"}, _ctx(tmp_path))
    assert out.startswith("sandbox_denied: network")
    assert "secret-from-loopback" not in out
    assert _Hits.hits == []


@pytest.mark.asyncio
async def test_webfetch_run_tools_wiring_denies_metadata(tmp_path):
    out = await run_tools("WebFetch", {"url": "http://169.254.169.254/latest/meta-data/"}, _ctx(tmp_path))
    assert out.startswith("sandbox_denied: network")
    assert "169.254.169.254" in out


@pytest.mark.asyncio
async def test_webfetch_redirect_to_loopback_denied(loopback_server):
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.headers.get("host") == "allowed.example":
            return httpx.Response(302, headers={"location": f"{loopback_server}/secret"})
        return httpx.Response(200, text="leaked")

    resolve = _resolver({"allowed.example": [PUBLIC_IP]})
    with pytest.raises(EgressDenied, match="blocked address"):
        await fetch_url(
            "http://allowed.example/start",
            NetworkPolicy(),
            transport=httpx.MockTransport(handler),
            resolver=resolve,
        )
    assert len(seen) == 1  # only the first hop was ever sent
    assert seen[0].url.host == PUBLIC_IP  # pinned to the vetted address
    assert _Hits.hits == []


@pytest.mark.asyncio
async def test_webfetch_redirect_to_internal_hostname_denied():
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers.get("host", ""))
        if request.headers.get("host") == "allowed.example":
            return httpx.Response(301, headers={"location": "http://intranet.example/admin"})
        return httpx.Response(200, text="leaked")

    resolve = _resolver({"allowed.example": [PUBLIC_IP], "intranet.example": ["10.0.0.5"]})
    with pytest.raises(EgressDenied, match="intranet.example resolved to a blocked address"):
        await fetch_url(
            "http://allowed.example/",
            NetworkPolicy(),
            transport=httpx.MockTransport(handler),
            resolver=resolve,
        )
    assert seen == ["allowed.example"]


@pytest.mark.asyncio
async def test_webfetch_allowed_host_pinned_with_host_and_sni():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.path == "/old":
            return httpx.Response(302, headers={"location": "/new?x=1"})
        return httpx.Response(
            200,
            headers={"content-type": "text/html; charset=utf-8"},
            text="<html><body><p>Hello from docs</p></body></html>",
        )

    resolve = _resolver({"docs.example.com": [PUBLIC_IP]})
    net = NetworkPolicy(web_default="deny", include_defaults=False, allow=("*.example.com",))
    result = await fetch_url(
        "https://docs.example.com/old",
        net,
        transport=httpx.MockTransport(handler),
        resolver=resolve,
    )
    assert result.status_code == 200
    assert "Hello from docs" in result.text
    assert result.final_url == "https://docs.example.com/new?x=1"
    assert [r.url.host for r in seen] == [PUBLIC_IP, PUBLIC_IP]
    assert all(r.headers["host"] == "docs.example.com" for r in seen)
    assert all(r.extensions.get("sni_hostname") == "docs.example.com" for r in seen)
    assert all(r.url.scheme == "https" for r in seen)


@pytest.mark.asyncio
async def test_webfetch_denied_by_domain_policy(tmp_path):
    root = tmp_path / "ws"
    (root / ".orbweaver").mkdir(parents=True)
    (root / ".orbweaver" / "sandbox.json").write_text(
        json.dumps({"networkPolicy": {"webDefault": "deny", "allow": ["*.example.com"]}}),
        encoding="utf-8",
    )
    ctx = _ctx(root)
    out = await run_webfetch({"url": "https://docs.python.org/3/"}, ctx)
    assert out.startswith("sandbox_denied: network")
    assert "docs.python.org not allowed" in out


@pytest.mark.asyncio
async def test_webfetch_body_cap_and_content_type_cap():
    big = b"x" * (64 * 1024)

    async def stream_big():
        for _ in range(8):
            yield big

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/big":
            return httpx.Response(200, headers={"content-type": "text/plain"}, stream=_AsyncIter(stream_big()))
        return httpx.Response(
            200,
            headers={"content-type": "application/octet-stream", "content-length": "1234"},
            content=b"\x00\x01binary",
        )

    resolve = _resolver({"public.example": [PUBLIC_IP]})
    result = await fetch_url(
        "http://public.example/big",
        NetworkPolicy(),
        transport=httpx.MockTransport(handler),
        resolver=resolve,
        max_bytes=100_000,
    )
    assert result.truncated
    assert len(result.text) == 100_000

    result = await fetch_url(
        "http://public.example/blob",
        NetworkPolicy(),
        transport=httpx.MockTransport(handler),
        resolver=resolve,
    )
    assert result.text == ""
    assert "application/octet-stream" in result.skipped
    assert "1234 bytes" in result.skipped


class _AsyncIter(httpx.AsyncByteStream):
    def __init__(self, gen):
        self._gen = gen

    async def __aiter__(self):
        async for chunk in self._gen:
            yield chunk


def test_fetchable_content_type():
    assert fetchable_content_type("")
    assert fetchable_content_type("text/html; charset=utf-8")
    assert fetchable_content_type("application/json")
    assert fetchable_content_type("application/problem+json")
    assert fetchable_content_type("application/atom+xml")
    assert not fetchable_content_type("application/octet-stream")
    assert not fetchable_content_type("image/png")
    assert not fetchable_content_type("application/pdf")


@pytest.mark.asyncio
async def test_webfetch_too_many_redirects():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"location": "http://public.example/loop"})

    resolve = _resolver({"public.example": [PUBLIC_IP]})
    with pytest.raises(EgressDenied, match="redirects"):
        await fetch_url(
            "http://public.example/",
            NetworkPolicy(),
            transport=httpx.MockTransport(handler),
            resolver=resolve,
        )


# --- Browser -----------------------------------------------------------------


class _FakeRoute:
    def __init__(self, url: str) -> None:
        self.request = type("Req", (), {"url": url})()
        self.outcome: str | None = None

    async def continue_(self) -> None:
        self.outcome = "continue"

    async def abort(self, reason: str = "") -> None:
        self.outcome = f"abort:{reason}"


@pytest.mark.asyncio
async def test_browser_gate_blocks_loopback_and_lan(monkeypatch):
    resolve = _resolver({"public.example": [PUBLIC_IP], "lan.example": ["192.168.4.4"]})
    monkeypatch.setattr("orbweaver.sandbox.egress.default_resolver", resolve)
    gate = EgressGate(NetworkPolicy())
    assert not (await gate.check("http://127.0.0.1:5173/")).allowed
    assert not (await gate.check("http://169.254.169.254/")).allowed
    assert not (await gate.check("https://lan.example/")).allowed
    assert (await gate.check("https://public.example/app")).allowed
    # cached per host: a second sub-resource does not resolve again
    await gate.check("https://public.example/app.js")
    assert resolve.calls.count("public.example") == 1

    for url, expect in (
        ("http://127.0.0.1:5173/api", "abort:blockedbyclient"),
        ("https://lan.example/img.png", "abort:blockedbyclient"),
        ("https://public.example/app.css", "continue"),
        ("file:///tmp/ws/page.html", "continue"),
        ("data:text/plain,hi", "continue"),
    ):
        route = _FakeRoute(url)
        await gate.route(route)
        assert route.outcome == expect, url


@pytest.mark.asyncio
async def test_browser_gate_honours_domain_allowlist(monkeypatch):
    resolve = _resolver({"docs.example.com": [PUBLIC_IP], "docs.python.org": [PUBLIC_IP]})
    monkeypatch.setattr("orbweaver.sandbox.egress.default_resolver", resolve)
    gate = EgressGate(NetworkPolicy(web_default="deny", include_defaults=False, allow=("*.example.com",)))
    assert (await gate.check("https://docs.example.com/")).allowed
    denied = await gate.check("https://docs.python.org/")
    assert not denied.allowed and "not allowed" in denied.reason


class _FakePage:
    def __init__(self) -> None:
        self.url = "about:blank"
        self.routes: list[str] = []

    async def goto(self, url, **_k):
        self.url = url

    async def route(self, pattern, handler):
        self.routes.append(pattern)

    async def evaluate(self, _js):
        return {"url": self.url, "title": "t", "text": "body", "nodes": []}


@pytest.mark.asyncio
async def test_browser_navigate_denied_before_goto(tmp_path, monkeypatch):
    page = _FakePage()

    async def ensure(self):
        self._page = page
        return page

    monkeypatch.setattr("orbweaver.browser.playwright_available", lambda: True)
    monkeypatch.setattr("orbweaver.browser._Session._ensure", ensure)
    ctx = _ctx(tmp_path)
    try:
        for url in ("http://127.0.0.1:8080/health", "http://169.254.169.254/latest/", "http://10.3.0.213/"):
            out = await run_tools("Browser", {"action": "navigate", "url": url}, ctx)
            assert out.startswith("Browser navigate failed: sandbox_denied: network"), out
            assert page.url == "about:blank"
        (tmp_path / "page.html").write_text("<html><body>ok</body></html>", encoding="utf-8")
        out = await run_tools("Browser", {"action": "navigate", "url": "page.html"}, ctx)
        assert page.url.startswith("file://")
        assert "body" in out
    finally:
        await pool.close_session(ctx)


@pytest.mark.asyncio
async def test_browser_session_installs_route_and_uses_workspace_policy(tmp_path, monkeypatch):
    resolve = _resolver({"docs.python.org": [PUBLIC_IP]})
    monkeypatch.setattr("orbweaver.sandbox.egress.default_resolver", resolve)
    root = tmp_path / "ws"
    (root / ".orbweaver").mkdir(parents=True)
    (root / ".orbweaver" / "sandbox.json").write_text(
        json.dumps({"networkPolicy": {"webDefault": "deny", "allow": ["*.example.com"]}}),
        encoding="utf-8",
    )
    page = _FakePage()

    async def ensure(self):
        if self._page is None:
            await page.route("**/*", self._route)
            self._page = page
        return page

    monkeypatch.setattr("orbweaver.browser.playwright_available", lambda: True)
    monkeypatch.setattr("orbweaver.browser._Session._ensure", ensure)
    ctx = _ctx(root)
    try:
        out = await run_tools("Browser", {"action": "navigate", "url": "https://docs.python.org/3/"}, ctx)
        assert "sandbox_denied: network" in out
        assert "docs.python.org not allowed" in out
        assert page.url == "about:blank"
        sess = await pool.session_for(ctx)
        assert isinstance(sess, _Session)
        assert sess.gate.network.web_default == "deny"
        assert "*.example.com" in sess.gate.network.allow
        out = await run_tools("Browser", {"action": "snapshot"}, ctx)
        assert page.routes == ["**/*"]
        assert "body" in out
    finally:
        await pool.close_session(ctx)
