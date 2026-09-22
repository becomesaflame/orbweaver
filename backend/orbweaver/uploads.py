"""Store an inbound file in a session workspace and describe it for the turn.

Shared by ``POST /v1/sessions/{id}/uploads`` (web UI) and the Telegram document
handler so both channels land files in the same place, sanitize names the same
way, and hand the agent the same kind of note.

Everything goes through the workspace abstraction (``write_bytes`` /
``stat_size``), never a raw host path: the workspace is what enforces the
working-set boundary.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any
from uuid import uuid4

from orbweaver.extract import (
    INLINE_TEXT_CHARS,
    KIND_IMAGE,
    ExtractResult,
    extract_text,
    human_size,
)
from orbweaver.image import save_inbound_image

log = logging.getLogger(__name__)

# 25 MB. Telegram's bot API caps downloads at 20 MB, so the web limit is the
# larger of the two and TELEGRAM_MAX_BYTES is checked separately.
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
TELEGRAM_MAX_BYTES = 20 * 1024 * 1024
ATTACHMENTS_DIR = "attachments"
MAX_STEM_CHARS = 80
MAX_SUFFIX_CHARS = 16
# After this many "name-N" attempts, stop scanning and use a random suffix.
COLLISION_ATTEMPTS = 50


class UploadError(Exception):
    """Client-fixable problem with an upload (bad name, too large)."""


class UploadTooLarge(UploadError):
    pass


class UploadNameError(UploadError):
    pass


@dataclass
class StoredUpload:
    """A file written into ``attachments/`` plus whatever text came out of it."""

    path: str
    original_name: str
    size: int
    result: ExtractResult
    image: dict[str, str] | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def kind(self) -> str:
        return self.result.kind

    def payload(self) -> dict[str, Any]:
        """JSON the web UI renders as a chip."""
        body: dict[str, Any] = {
            "path": self.path,
            "name": self.original_name,
            "size": self.size,
            "size_human": human_size(self.size),
            **self.result.payload(),
        }
        if self.image is not None:
            body["image"] = self.image
        if self.result.text:
            body["preview"] = self.result.preview()
        body.update(self.extra)
        return body


def sanitize_filename(raw: str) -> str:
    """Basename of ``raw``, with anything that is not [A-Za-z0-9._-] replaced.

    Directory components are dropped (a directory-drop in Chrome sends
    ``dir/file.txt``), but an explicit ``..`` segment is refused rather than
    silently flattened: nothing legitimate sends it, so it is worth reporting.
    """
    name = str(raw or "").replace("\\", "/").strip()
    segments = [seg for seg in name.split("/") if seg]
    if any(seg == ".." for seg in segments):
        raise UploadNameError("filename must not contain '..'")
    base = segments[-1] if segments else ""
    if base in {"", ".", ".."}:
        raise UploadNameError("filename is required")
    stem = PurePosixPath(base).stem
    suffix = PurePosixPath(base).suffix

    def clean(part: str) -> str:
        return "".join(c if (c.isascii() and (c.isalnum() or c in "._-")) else "_" for c in part)

    stem = clean(stem)[:MAX_STEM_CHARS].strip("._-") or "upload"
    suffix = clean(suffix)[:MAX_SUFFIX_CHARS]
    return stem + suffix


def _exists(workspace: Any, rel: str) -> bool:
    try:
        workspace.stat_size(rel)
    except (OSError, PermissionError, ValueError):
        return False
    except AttributeError:
        # Workspace without stat_size: fall back to a read.
        try:
            workspace.read_bytes(rel)
        except Exception:
            return False
        return True
    return True


def unique_attachment_path(workspace: Any, filename: str) -> str:
    """``attachments/<name>``, suffixed with -1, -2, … rather than overwriting."""
    safe = sanitize_filename(filename)
    stem = PurePosixPath(safe).stem
    suffix = PurePosixPath(safe).suffix
    rel = f"{ATTACHMENTS_DIR}/{safe}"
    if not _exists(workspace, rel):
        return rel
    for n in range(1, COLLISION_ATTEMPTS + 1):
        candidate = f"{ATTACHMENTS_DIR}/{stem}-{n}{suffix}"
        if not _exists(workspace, candidate):
            return candidate
    return f"{ATTACHMENTS_DIR}/{stem}-{uuid4().hex[:8]}{suffix}"


def check_size(size: int, limit: int = MAX_UPLOAD_BYTES) -> None:
    if size > limit:
        raise UploadTooLarge(
            f"file is {human_size(size)}; the limit is {human_size(limit)}"
        )


def store_upload(
    workspace: Any, data: bytes, filename: str, *, limit: int = MAX_UPLOAD_BYTES
) -> StoredUpload:
    """Write ``data`` under ``attachments/`` and extract its text.

    Images take the vision path (resized and re-encoded by
    ``save_inbound_image``), so the stored bytes are the normalized JPEG rather
    than the upload. Everything else is stored verbatim; the extracted text is
    a *view* of it, and the agent can always Read the file for the rest.

    Raises ``UploadError`` for a bad name or an oversized file. Extraction
    itself never raises: a corrupt document is stored with an explanatory note.
    """
    payload = data or b""
    check_size(len(payload), limit)
    result = extract_text(payload, filename)
    if result.kind == KIND_IMAGE:
        safe = sanitize_filename(filename)
        stem = PurePosixPath(safe).stem
        # save_inbound_image always writes <stem>.jpg, so de-duplicate on that.
        rel = unique_attachment_path(workspace, f"{stem}.jpg")
        img = save_inbound_image(workspace, payload, PurePosixPath(rel).stem)
        return StoredUpload(
            path=img["path"],
            original_name=filename,
            size=len(payload),
            result=result,
            image=img,
        )
    rel = unique_attachment_path(workspace, filename)
    workspace.write_bytes(rel, payload)
    return StoredUpload(
        path=rel, original_name=filename, size=len(payload), result=result
    )


def _headline(stored: StoredUpload) -> str:
    bits = [stored.kind, human_size(stored.size)]
    if stored.result.pages:
        bits.append(f"{stored.result.pages} pages")
    return f"[Attached file: {stored.path} ({', '.join(bits)})]"


def describe_for_turn(stored: StoredUpload, *, inline: int = INLINE_TEXT_CHARS) -> str:
    """One block of message text telling the agent what arrived and where.

    Text is inlined only up to ``inline`` characters; the file itself stays in
    the workspace, so the agent reads the rest with Read when it matters. That
    keeps a 200k-character extraction out of the first user message.
    """
    head = _headline(stored)
    if stored.image is not None:
        return head
    body = stored.result.text.strip()
    if not body:
        note = stored.result.note or "no text extracted"
        return f"{head}\n{note}"
    chunk = body if len(body) <= inline else body[:inline].rstrip()
    lines = [head]
    if stored.result.note:
        lines.append(stored.result.note)
    lines.append(chunk)
    if len(body) > inline or stored.result.truncated:
        lines.append(f"[... truncated here; Read {stored.path} for the full file ...]")
    return "\n".join(lines)


def describe_all(stored: list[StoredUpload], *, inline: int = INLINE_TEXT_CHARS) -> str:
    return "\n\n".join(describe_for_turn(s, inline=inline) for s in stored)
