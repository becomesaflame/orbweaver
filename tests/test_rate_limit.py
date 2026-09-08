import pytest
from httpx import ASGITransport, AsyncClient
from starlette.requests import Request

from orbweaver import app as app_mod
from orbweaver.app import _hits, app, rate_limit_client_ip
from orbweaver.config import settings


def _request(headers: dict[str, str] | None = None, host: str = "10.0.0.1") -> Request:
    raw = [(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()]
    return Request(
        {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "GET",
            "scheme": "http",
            "path": "/v1/sessions",
            "raw_path": b"/v1/sessions",
            "query_string": b"",
            "headers": raw,
            "client": (host, 12345),
            "server": ("test", 80),
        }
    )


@pytest.fixture(autouse=True)
def _clear_hits():
    _hits.clear()
    yield
    _hits.clear()


def test_trust_proxy_defaults_off():
    assert settings.orbweaver_trust_proxy is False


def test_client_ip_ignores_forwarded_headers_by_default(monkeypatch):
    monkeypatch.setattr(settings, "orbweaver_trust_proxy", False)
    req = _request(
        {"x-forwarded-for": "203.0.113.9", "x-real-ip": "198.51.100.4"},
        host="10.0.0.1",
    )
    assert rate_limit_client_ip(req) == "10.0.0.1"


def test_client_ip_uses_x_forwarded_for_when_trusted(monkeypatch):
    monkeypatch.setattr(settings, "orbweaver_trust_proxy", True)
    req = _request({"x-forwarded-for": "203.0.113.9, 10.0.0.2"}, host="10.0.0.1")
    assert rate_limit_client_ip(req) == "203.0.113.9"


def test_client_ip_uses_x_real_ip_when_trusted(monkeypatch):
    monkeypatch.setattr(settings, "orbweaver_trust_proxy", True)
    req = _request({"x-real-ip": "198.51.100.4"}, host="10.0.0.1")
    assert rate_limit_client_ip(req) == "198.51.100.4"


def test_client_ip_xff_wins_over_x_real_ip(monkeypatch):
    monkeypatch.setattr(settings, "orbweaver_trust_proxy", True)
    req = _request(
        {"x-forwarded-for": "203.0.113.9", "x-real-ip": "198.51.100.4"},
        host="10.0.0.1",
    )
    assert rate_limit_client_ip(req) == "203.0.113.9"


def test_client_ip_falls_back_to_socket_when_headers_invalid(monkeypatch):
    monkeypatch.setattr(settings, "orbweaver_trust_proxy", True)
    req = _request({"x-forwarded-for": "not-an-ip", "x-real-ip": "also-bad"}, host="10.0.0.1")
    assert rate_limit_client_ip(req) == "10.0.0.1"


@pytest.mark.asyncio
async def test_rate_limit_ignores_spoofed_xff_by_default(monkeypatch):
    monkeypatch.setattr(settings, "orbweaver_trust_proxy", False)
    monkeypatch.setattr(app_mod, "_RATE_LIMIT_MAX", 3)
    transport = ASGITransport(app=app, client=("10.0.0.1", 123))
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        for i in range(3):
            r = await client.get("/v1/sessions", headers={"x-forwarded-for": f"203.0.113.{i}"})
            assert r.status_code == 401
        blocked = await client.get("/v1/sessions", headers={"x-forwarded-for": "203.0.113.99"})
        assert blocked.status_code == 429


@pytest.mark.asyncio
async def test_rate_limit_keys_on_forwarded_ip_when_trusted(monkeypatch):
    monkeypatch.setattr(settings, "orbweaver_trust_proxy", True)
    monkeypatch.setattr(app_mod, "_RATE_LIMIT_MAX", 3)
    transport = ASGITransport(app=app, client=("10.0.0.1", 123))
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        for _ in range(3):
            r = await client.get("/v1/sessions", headers={"x-forwarded-for": "203.0.113.9"})
            assert r.status_code == 401
        blocked = await client.get("/v1/sessions", headers={"x-forwarded-for": "203.0.113.9"})
        assert blocked.status_code == 429
        other = await client.get("/v1/sessions", headers={"x-forwarded-for": "198.51.100.4"})
        assert other.status_code == 401
        real_ip = await client.get("/v1/sessions", headers={"x-real-ip": "192.0.2.10"})
        assert real_ip.status_code == 401
