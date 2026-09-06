import sys

import pytest
from httpx import ASGITransport, AsyncClient

from orbweaver.app import app
from orbweaver.cli import main
from orbweaver.config import settings


@pytest.mark.asyncio
async def test_http_mint_disabled_by_default():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.post("/v1/auth/token", json={"sub": "t"})
        assert r.status_code == 404


@pytest.mark.asyncio
async def test_http_mint_when_enabled(monkeypatch):
    monkeypatch.setattr(settings, "orbweaver_allow_http_mint", True)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.post("/v1/auth/token", json={"sub": "t"})
        assert r.status_code == 200
        assert r.json()["token"].count(".") == 2


def test_mint_cli_prints_jwt(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["orbweaver", "mint", "--sub", "t"])
    main()
    out = capsys.readouterr().out.strip()
    assert out.count(".") == 2
