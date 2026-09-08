import io
from uuid import uuid4

import pytest
from PIL import Image

from orbweaver.agent import run_tools
from orbweaver.channels.telegram import generate_and_maybe_send, send_session_photo
from orbweaver.compact.project import events_to_messages
from orbweaver.config import settings
from orbweaver.image import (
    format_image_read,
    hydrate_workspace_images,
    process_image_bytes,
    save_inbound_image,
    sniff_media_type,
)
from orbweaver.store import Event, reset_store_for_tests
from orbweaver.workspace import LocalWorkspace


def _png_bytes(size: int = 8, color: tuple[int, int, int] = (200, 10, 10)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (size, size), color).save(buf, format="PNG")
    return buf.getvalue()


def test_process_image_bytes_jpeg_normalizes():
    jpeg, media = process_image_bytes(_png_bytes(64))
    assert media == "image/jpeg"
    assert jpeg.startswith(b"\xff\xd8\xff")
    assert sniff_media_type(jpeg) == "image/jpeg"


def test_save_inbound_image_writes_attachments(tmp_path):
    ws = LocalWorkspace("workspace:default", str(tmp_path))
    saved = save_inbound_image(ws, _png_bytes(), "photo 1!")
    assert saved["path"] == "attachments/photo_1_.jpg"
    assert saved["media_type"] == "image/jpeg"
    assert (tmp_path / saved["path"]).is_file()


def test_events_to_messages_emits_workspace_image_blocks():
    sid = uuid4()
    events = [
        Event(
            id=uuid4(),
            session_id=sid,
            seq=1,
            kind="user",
            payload={
                "text": "what is this?",
                "images": [{"path": "attachments/photo.jpg", "media_type": "image/jpeg"}],
            },
        )
    ]
    messages = events_to_messages(events)
    content = messages[0]["content"]
    assert isinstance(content, list)
    assert content[0]["type"] == "image"
    assert content[0]["source"]["type"] == "workspace_path"
    assert content[0]["source"]["path"] == "attachments/photo.jpg"
    assert content[1] == {"type": "text", "text": "what is this?"}


def test_hydrate_workspace_images_base64(tmp_path):
    ws = LocalWorkspace("workspace:default", str(tmp_path))
    saved = save_inbound_image(ws, _png_bytes(), "shot")
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "workspace_path",
                        "media_type": "image/jpeg",
                        "path": saved["path"],
                    },
                },
                {"type": "text", "text": "describe"},
            ],
        }
    ]
    out = hydrate_workspace_images(messages, ws)
    src = out[0]["content"][0]["source"]
    assert src["type"] == "base64"
    assert src["media_type"] == "image/jpeg"
    assert len(src["data"]) > 20


def test_hydrate_drops_missing_image(tmp_path):
    ws = LocalWorkspace("workspace:default", str(tmp_path))
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "workspace_path",
                        "media_type": "image/jpeg",
                        "path": "attachments/missing.jpg",
                    },
                },
                {"type": "text", "text": "hi"},
            ],
        }
    ]
    out = hydrate_workspace_images(messages, ws)
    assert out[0]["content"] == [{"type": "text", "text": "hi"}]


@pytest.mark.asyncio
async def test_send_session_photo_without_chat(tmp_path):
    ws = LocalWorkspace("workspace:default", str(tmp_path))
    saved = save_inbound_image(ws, _png_bytes(), "out")
    store = reset_store_for_tests()
    sid = uuid4()
    result = await send_session_photo(
        {"workspace": ws, "store": store, "session_id": sid},
        {"path": saved["path"], "caption": "hi"},
    )
    assert "no telegram chat" in result
    assert saved["path"] in result


@pytest.mark.asyncio
async def test_generate_image_without_api_key(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "orbweaver_image_api_key", "")
    ws = LocalWorkspace("workspace:default", str(tmp_path))
    store = reset_store_for_tests()
    result = await generate_and_maybe_send(
        {"workspace": ws, "store": store, "session_id": uuid4()},
        {"prompt": "a red square"},
    )
    assert "ORBWEAVER_IMAGE_API_KEY" in result
    assert "SendPhoto" in result


@pytest.mark.asyncio
async def test_read_image_path_returns_vision_marker(tmp_path):
    ws = LocalWorkspace("workspace:default", str(tmp_path))
    saved = save_inbound_image(ws, _png_bytes(16), "vision")
    store = reset_store_for_tests()
    result = await run_tools("Read", {"path": saved["path"]}, {"workspace": ws, "store": store, "session_id": uuid4()})
    assert "__orbweaver_image__" in result
    assert saved["path"] in result
    assert "image/jpeg" in result
    ws.write("notes.txt", "hello\n")
    text_result = await run_tools(
        "Read", {"path": "notes.txt"}, {"workspace": ws, "store": store, "session_id": uuid4()}
    )
    assert "hello" in text_result
    assert "__orbweaver_image__" not in text_result


def test_format_image_read_missing_file(tmp_path):
    ws = LocalWorkspace("workspace:default", str(tmp_path))
    result = format_image_read(ws, "attachments/missing.jpg")
    assert result.startswith("error reading")


def test_read_image_events_hydrate_like_telegram(tmp_path):
    ws = LocalWorkspace("workspace:default", str(tmp_path))
    saved = save_inbound_image(ws, _png_bytes(12), "loop")
    marker = format_image_read(ws, saved["path"])
    sid = uuid4()
    events = [
        Event(id=uuid4(), session_id=sid, seq=1, kind="user", payload={"text": "what is this?"}),
        Event(
            id=uuid4(),
            session_id=sid,
            seq=2,
            kind="tool_call",
            payload={"id": "toolu_1", "name": "Read", "input": {"path": saved["path"]}},
        ),
        Event(
            id=uuid4(),
            session_id=sid,
            seq=3,
            kind="tool_result",
            payload={"tool_use_id": "toolu_1", "name": "Read", "content": marker},
        ),
    ]
    messages = events_to_messages(events)
    tool = messages[-1]["content"][0]
    assert tool["type"] == "tool_result"
    blocks = tool["content"]
    assert blocks[0]["type"] == "image"
    assert blocks[0]["source"]["type"] == "workspace_path"
    assert blocks[0]["source"]["path"] == saved["path"]
    assert blocks[1]["type"] == "text"
    assert "Read image" in blocks[1]["text"]

    hydrated = hydrate_workspace_images(messages, ws)
    src = hydrated[-1]["content"][0]["content"][0]["source"]
    assert src["type"] == "base64"
    assert src["media_type"] == "image/jpeg"
    assert len(src["data"]) > 20
