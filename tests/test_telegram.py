import io

import httpx
import pytest
from PIL import Image

from orbweaver.channels.telegram import (
    notify_telegram_photo,
    send_session_photo,
    session_for_telegram_user,
    user_allowed,
)
from orbweaver.config import settings
from orbweaver.image import save_inbound_image
from orbweaver.store import reset_store_for_tests
from orbweaver.workspace import LocalWorkspace, apply_local_workspace_kind


def _png_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), (10, 200, 10)).save(buf, format="PNG")
    return buf.getvalue()


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
    assert a.jsonld["workspace_kind"] == "local"
    assert a.jsonld["channel"] == "telegram"
    assert a.jsonld["telegram_chat_id"] == 42
    updated = await session_for_telegram_user(store, 42, chat_id=999)
    assert updated.id == a.id
    assert updated.jsonld["telegram_chat_id"] == 999
    assert c.id != a.id


@pytest.mark.asyncio
async def test_telegram_session_migrates_docker_kind():
    store = reset_store_for_tests()
    ent = await session_for_telegram_user(store, 42)
    ent.jsonld["workspace_kind"] = "docker"
    await store.put_entity(ent)
    assert apply_local_workspace_kind({"workspace_kind": "docker"}) is True
    migrated = await session_for_telegram_user(store, 42)
    assert migrated.id == ent.id
    assert migrated.jsonld["workspace_kind"] == "local"


class _Resp:
    def __init__(self, status=200, text="ok"):
        self.status_code = status
        self.text = text


class _FakeAsyncClient:
    seen: dict | None = None

    def __init__(self, *a, **k):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def post(self, url, data=None, files=None, json=None):
        _FakeAsyncClient.seen = {"url": url, "data": data, "files": files, "json": json}
        return _Resp()


@pytest.mark.asyncio
async def test_notify_telegram_photo_posts(monkeypatch):
    monkeypatch.setattr(settings, "telegram_bot_token", "tok")
    monkeypatch.setattr(httpx, "AsyncClient", _FakeAsyncClient)
    result = await notify_telegram_photo(7, b"abc", "cap", "x.jpg", "image/jpeg")
    assert result == "ok"
    assert _FakeAsyncClient.seen["url"].endswith("/sendPhoto")
    assert _FakeAsyncClient.seen["data"]["chat_id"] == "7"
    assert _FakeAsyncClient.seen["data"]["caption"] == "cap"


@pytest.mark.asyncio
async def test_send_session_photo_posts_when_chat_bound(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "telegram_bot_token", "tok")
    monkeypatch.setattr(httpx, "AsyncClient", _FakeAsyncClient)
    ws = LocalWorkspace("workspace:default", str(tmp_path))
    saved = save_inbound_image(ws, _png_bytes(), "out")
    store = reset_store_for_tests()
    sess = await session_for_telegram_user(store, 42, chat_id=99)
    result = await send_session_photo(
        {"workspace": ws, "store": store, "session_id": sess.id},
        {"path": saved["path"], "caption": "here"},
    )
    assert '"sent": true' in result
    assert _FakeAsyncClient.seen["url"].endswith("/sendPhoto")
    assert _FakeAsyncClient.seen["data"]["chat_id"] == "99"


@pytest.mark.asyncio
async def test_run_turn_replies_when_agent_raises(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from uuid import uuid4

    from orbweaver.channels.telegram import _run_turn

    replies: list[str] = []

    class _Msg:
        async def reply_text(self, text):
            replies.append(text)

    async def boom(*_a, **_k):
        raise RuntimeError("tool_use ids were found without tool_result blocks")

    async def fake_ws(*_a, **_k):
        return (
            reset_store_for_tests(),
            uuid4(),
            LocalWorkspace("workspace:default", str(tmp_path)),
            "local",
        )

    monkeypatch.setattr("orbweaver.channels.telegram.agent_turn", boom)
    monkeypatch.setattr("orbweaver.channels.telegram._session_workspace", fake_ws)
    await _run_turn(SimpleNamespace(message=_Msg()), None, "Go ahead with the push")
    assert replies
    assert "Turn failed" in replies[0]
    assert "tool_use" in replies[0]
