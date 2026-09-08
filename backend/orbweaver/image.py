"""Inbound image resize/normalize and Anthropic workspace-path hydration."""

from __future__ import annotations

import base64
import io
import logging
from pathlib import Path
from typing import Any

from orbweaver.config import settings

log = logging.getLogger(__name__)

MAX_DIMENSION = 1568
MAX_BYTES = 4_500_000
JPEG_QUALITY = 85
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
PHOTO_MEDIA_TYPES = {"image/jpeg", "image/png", "image/webp"}


def is_image_path(path: str) -> bool:
    return Path(path).suffix.lower() in IMAGE_EXTENSIONS


def sniff_media_type(data: bytes) -> str:
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith(b"\x89PNG"):
        return "image/png"
    if data.startswith(b"GIF8"):
        return "image/gif"
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "image/webp"
    return "application/octet-stream"


def process_image_bytes(data: bytes) -> tuple[bytes, str]:
    """Resize and JPEG-normalize inbound bytes for Claude vision."""
    if not data:
        raise ValueError("empty image")
    from PIL import Image

    im = Image.open(io.BytesIO(data)).convert("RGB")
    im.thumbnail((MAX_DIMENSION, MAX_DIMENSION))
    quality = JPEG_QUALITY
    out = b""
    while quality >= 40:
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=quality, optimize=True)
        out = buf.getvalue()
        if len(out) <= MAX_BYTES:
            return out, "image/jpeg"
        quality -= 15
    if len(out) > MAX_BYTES:
        raise ValueError("image still too large after compression")
    return out, "image/jpeg"


def save_inbound_image(workspace, data: bytes, stem: str) -> dict[str, str]:
    jpeg, media_type = process_image_bytes(data)
    safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in stem) or "photo"
    rel = f"attachments/{safe}.jpg"
    workspace.write_bytes(rel, jpeg)
    return {"path": rel, "media_type": media_type}


def user_image_blocks(images: list[dict[str, str]]) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for img in images:
        path = str(img.get("path") or "").strip()
        if not path:
            continue
        blocks.append(
            {
                "type": "image",
                "source": {
                    "type": "workspace_path",
                    "media_type": img.get("media_type") or "image/jpeg",
                    "path": path,
                },
            }
        )
    return blocks


def hydrate_workspace_images(messages: list[dict[str, Any]], workspace: Any) -> list[dict[str, Any]]:
    """Replace workspace_path image sources with base64 for the Anthropic API."""
    if workspace is None:
        return messages
    out: list[dict[str, Any]] = []
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            out.append(msg)
            continue
        blocks: list[Any] = []
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "image":
                blocks.append(block)
                continue
            src = block.get("source") or {}
            if src.get("type") != "workspace_path":
                blocks.append(block)
                continue
            path = str(src.get("path") or "")
            try:
                data = workspace.read_bytes(path)
            except (OSError, PermissionError, FileNotFoundError, IsADirectoryError, AttributeError) as e:
                log.warning("image hydrate failed for %s: %s", path, e)
                continue
            blocks.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": src.get("media_type") or sniff_media_type(data),
                        "data": base64.b64encode(data).decode("ascii"),
                    },
                }
            )
        new_msg = dict(msg)
        new_msg["content"] = blocks
        out.append(new_msg)
    return out


def generate_image_bytes(prompt: str) -> tuple[bytes, str]:
    key = settings.orbweaver_image_api_key.strip()
    if not key:
        raise RuntimeError("ORBWEAVER_IMAGE_API_KEY is not set")
    import httpx

    headers = {"Authorization": f"Bearer {key}"}
    payload = {
        "model": settings.orbweaver_image_model,
        "prompt": prompt,
        "n": 1,
        "size": settings.orbweaver_image_size,
        "response_format": "b64_json",
    }
    r = httpx.post(settings.orbweaver_image_api_url, headers=headers, json=payload, timeout=120.0)
    r.raise_for_status()
    item = ((r.json().get("data") or [None])[0]) or {}
    if item.get("b64_json"):
        raw = base64.b64decode(item["b64_json"])
        media = sniff_media_type(raw)
        if media == "application/octet-stream":
            media = "image/png"
        return raw, media
    if item.get("url"):
        img = httpx.get(item["url"], timeout=60.0, follow_redirects=True)
        img.raise_for_status()
        return img.content, sniff_media_type(img.content)
    raise RuntimeError("image API returned no image")
