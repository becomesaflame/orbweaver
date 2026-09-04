import pytest

from orbweaver.channels.telegram import session_for_telegram_user, user_allowed
from orbweaver.config import settings
from orbweaver.store import reset_store_for_tests


def test_user_allowed(monkeypatch):
    monkeypatch.setattr(settings, "telegram_allowlist", "")
    assert user_allowed(1)
    monkeypatch.setattr(settings, "telegram_allowlist", "10,20")
    assert user_allowed(10)
    assert not user_allowed(99)


@pytest.mark.asyncio
async def test_telegram_session_reuse():
    store = reset_store_for_tests()
    a = await session_for_telegram_user(store, 42)
    b = await session_for_telegram_user(store, 42)
    c = await session_for_telegram_user(store, 7)
    assert a.id == b.id
    assert a.jsonld["workspace_kind"] == "docker"
    assert c.id != a.id
