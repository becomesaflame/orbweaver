"""POST /v1/sessions/{id}/uploads plus the storage helpers behind it."""

import io
import zipfile
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient
from PIL import Image

from orbweaver.app import app
from orbweaver.store import reset_store_for_tests
from orbweaver.uploads import (
    MAX_UPLOAD_BYTES,
    UploadNameError,
    UploadTooLarge,
    describe_for_turn,
    sanitize_filename,
    store_upload,
    unique_attachment_path,
)
from orbweaver.workspace import LocalWorkspace


@pytest.fixture(autouse=True)
def _store():
    reset_store_for_tests()


def _png_bytes(size: int = 8) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (size, size), (200, 140, 20)).save(buf, format="PNG")
    return buf.getvalue()


def _docx_bytes(text: str) -> bytes:
    xml = (
        '<?xml version="1.0"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        f"<w:body><w:p><w:r><w:t>{text}</w:t></w:r></w:p></w:body></w:document>"
    )
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("word/document.xml", xml)
    return buf.getvalue()


async def _session(client, headers, tmp_path, monkeypatch) -> str:
    from orbweaver.config import settings

    monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
    monkeypatch.setattr(settings, "workspace_root", str(tmp_path))
    r = await client.post(
        "/v1/sessions",
        json={"workspace_uri": "workspace:default", "workspace_kind": "local"},
        headers=headers,
    )
    assert r.status_code == 200, r.text
    return r.json()["id"]


# ------------------------------------------------------------- name sanitizing


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("notes.txt", "notes.txt"),
        ("/etc/passwd", "passwd"),
        ("sub/dir/report.pdf", "report.pdf"),
        ("C:\\Users\\me\\thing.md", "thing.md"),
        ("weird name (1).txt", "weird_name__1.txt"),
        ("no-extension", "no-extension"),
        (".hidden", "hidden"),
    ],
)
def test_sanitize_filename(raw, expected):
    assert sanitize_filename(raw) == expected


@pytest.mark.parametrize("raw", ["../../etc/passwd", "a/../../b.txt", "..", ""])
def test_sanitize_filename_rejects_traversal_and_empty(raw):
    with pytest.raises(UploadNameError):
        sanitize_filename(raw)


def test_sanitize_filename_caps_absurd_length():
    out = sanitize_filename("z" * 500 + ".txt")
    assert out.endswith(".txt")
    assert len(out) < 120


def test_unique_attachment_path_suffixes_collisions(tmp_path: Path):
    ws = LocalWorkspace("workspace:default", str(tmp_path))
    assert unique_attachment_path(ws, "a.txt") == "attachments/a.txt"
    ws.write_bytes("attachments/a.txt", b"one")
    assert unique_attachment_path(ws, "a.txt") == "attachments/a-1.txt"
    ws.write_bytes("attachments/a-1.txt", b"two")
    assert unique_attachment_path(ws, "a.txt") == "attachments/a-2.txt"


# ------------------------------------------------------------------ store_upload


def test_store_upload_writes_text_and_extracts(tmp_path: Path):
    ws = LocalWorkspace("workspace:default", str(tmp_path))
    stored = store_upload(ws, b"line one\nline two\n", "notes.txt")
    assert stored.path == "attachments/notes.txt"
    assert ws.read_bytes(stored.path) == b"line one\nline two\n"
    assert "line one" in stored.result.text
    assert stored.payload()["kind"] == "text"


def test_store_upload_never_overwrites(tmp_path: Path):
    ws = LocalWorkspace("workspace:default", str(tmp_path))
    first = store_upload(ws, b"first", "dup.txt")
    second = store_upload(ws, b"second", "dup.txt")
    assert first.path != second.path
    assert ws.read_bytes(first.path) == b"first"
    assert ws.read_bytes(second.path) == b"second"


def test_store_upload_image_takes_the_vision_path(tmp_path: Path):
    ws = LocalWorkspace("workspace:default", str(tmp_path))
    stored = store_upload(ws, _png_bytes(), "shot.png")
    assert stored.image is not None
    # save_inbound_image normalizes to JPEG regardless of the upload format.
    assert stored.path.endswith(".jpg")
    assert stored.result.text == ""
    assert ws.read_bytes(stored.path)


def test_store_upload_rejects_oversize(tmp_path: Path):
    ws = LocalWorkspace("workspace:default", str(tmp_path))
    with pytest.raises(UploadTooLarge):
        store_upload(ws, b"x" * 100, "big.txt", limit=50)


def test_store_upload_rejects_traversal_name(tmp_path: Path):
    ws = LocalWorkspace("workspace:default", str(tmp_path))
    with pytest.raises(UploadNameError):
        store_upload(ws, b"x", "../../escape.txt")


def test_store_upload_stores_corrupt_pdf_with_a_note(tmp_path: Path):
    """A malformed document must still land on disk; the note explains itself."""
    ws = LocalWorkspace("workspace:default", str(tmp_path))
    stored = store_upload(ws, b"%PDF-1.4\ngarbage", "broken.pdf")
    assert ws.read_bytes(stored.path) == b"%PDF-1.4\ngarbage"
    assert stored.payload()["ok"] is False
    assert stored.payload()["note"]


def test_store_upload_binary_is_stored_but_not_extracted(tmp_path: Path):
    ws = LocalWorkspace("workspace:default", str(tmp_path))
    stored = store_upload(ws, b"\x7fELF" + b"\x00" * 64, "a.out")
    assert stored.kind == "binary"
    assert stored.result.text == ""
    assert ws.read_bytes(stored.path)


# ---------------------------------------------------------- describe_for_turn


def test_describe_for_turn_inlines_short_text(tmp_path: Path):
    ws = LocalWorkspace("workspace:default", str(tmp_path))
    stored = store_upload(ws, b"the whole body", "small.txt")
    note = describe_for_turn(stored)
    assert "attachments/small.txt" in note
    assert "the whole body" in note
    assert "truncated here" not in note


def test_describe_for_turn_truncates_long_text_and_points_at_the_file(tmp_path: Path):
    ws = LocalWorkspace("workspace:default", str(tmp_path))
    stored = store_upload(ws, b"q" * 50_000, "long.txt")
    note = describe_for_turn(stored, inline=500)
    assert len(note) < 1200
    assert "attachments/long.txt" in note
    assert "truncated here" in note


def test_describe_for_turn_names_an_image_without_text(tmp_path: Path):
    ws = LocalWorkspace("workspace:default", str(tmp_path))
    stored = store_upload(ws, _png_bytes(), "pic.png")
    note = describe_for_turn(stored)
    assert stored.path in note
    assert "image" in note


# -------------------------------------------------------------- HTTP endpoint


@pytest.mark.asyncio
async def test_upload_text_file(tmp_path, monkeypatch, auth_header):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        sid = await _session(client, auth_header, tmp_path, monkeypatch)
        r = await client.post(
            f"/v1/sessions/{sid}/uploads",
            files={"file": ("notes.md", b"# Heading\n\nbody text\n", "text/markdown")},
            headers=auth_header,
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["path"] == "attachments/notes.md"
        assert body["kind"] == "text"
        assert body["ok"] is True
        assert "body text" in body["preview"]
        assert body["size"] == len(b"# Heading\n\nbody text\n")
        assert (tmp_path / "attachments" / "notes.md").is_file()


@pytest.mark.asyncio
async def test_upload_image_returns_marker(tmp_path, monkeypatch, auth_header):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        sid = await _session(client, auth_header, tmp_path, monkeypatch)
        r = await client.post(
            f"/v1/sessions/{sid}/uploads",
            files={"file": ("pic.png", _png_bytes(), "image/png")},
            headers=auth_header,
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["kind"] == "image"
        assert body["image"]["path"] == body["path"]
        assert body["path"].endswith(".jpg")


@pytest.mark.asyncio
async def test_upload_docx_extracts_text(tmp_path, monkeypatch, auth_header):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        sid = await _session(client, auth_header, tmp_path, monkeypatch)
        r = await client.post(
            f"/v1/sessions/{sid}/uploads",
            files={"file": ("report.docx", _docx_bytes("quarterly numbers"), "application/zip")},
            headers=auth_header,
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["kind"] == "docx"
        assert "quarterly numbers" in body["preview"]


@pytest.mark.asyncio
async def test_upload_binary_is_reported_not_dumped(tmp_path, monkeypatch, auth_header):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        sid = await _session(client, auth_header, tmp_path, monkeypatch)
        r = await client.post(
            f"/v1/sessions/{sid}/uploads",
            files={"file": ("a.out", b"\x7fELF" + b"\x00" * 64, "application/octet-stream")},
            headers=auth_header,
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["kind"] == "binary"
        assert body["ok"] is False
        assert "preview" not in body


@pytest.mark.asyncio
async def test_upload_oversize_is_413(tmp_path, monkeypatch, auth_header):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        sid = await _session(client, auth_header, tmp_path, monkeypatch)
        payload = b"z" * (MAX_UPLOAD_BYTES + 1024)
        r = await client.post(
            f"/v1/sessions/{sid}/uploads",
            files={"file": ("huge.txt", payload, "text/plain")},
            headers=auth_header,
        )
        assert r.status_code == 413, r.text
        assert "limit" in r.json()["detail"]
        assert not (tmp_path / "attachments" / "huge.txt").exists()


@pytest.mark.asyncio
async def test_upload_traversal_filename_is_rejected(tmp_path, monkeypatch, auth_header):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        sid = await _session(client, auth_header, tmp_path, monkeypatch)
        r = await client.post(
            f"/v1/sessions/{sid}/uploads",
            files={"file": ("../../escaped.txt", b"nope", "text/plain")},
            headers=auth_header,
        )
        assert r.status_code == 400, r.text
        assert not (tmp_path.parent / "escaped.txt").exists()


@pytest.mark.asyncio
async def test_upload_directory_prefix_is_flattened(tmp_path, monkeypatch, auth_header):
    """Chrome sends "dir/file.txt" for a folder drop; keep the basename."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        sid = await _session(client, auth_header, tmp_path, monkeypatch)
        r = await client.post(
            f"/v1/sessions/{sid}/uploads",
            files={"file": ("nested/dir/file.txt", b"ok", "text/plain")},
            headers=auth_header,
        )
        assert r.status_code == 200, r.text
        assert r.json()["path"] == "attachments/file.txt"


@pytest.mark.asyncio
async def test_upload_collisions_get_distinct_paths(tmp_path, monkeypatch, auth_header):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        sid = await _session(client, auth_header, tmp_path, monkeypatch)
        paths = []
        for body in (b"one", b"two", b"three"):
            r = await client.post(
                f"/v1/sessions/{sid}/uploads",
                files={"file": ("same.txt", body, "text/plain")},
                headers=auth_header,
            )
            assert r.status_code == 200, r.text
            paths.append(r.json()["path"])
        assert len(set(paths)) == 3
        assert paths[0] == "attachments/same.txt"
        assert (tmp_path / "attachments" / "same.txt").read_bytes() == b"one"


@pytest.mark.asyncio
async def test_upload_to_deleted_session_is_404(tmp_path, monkeypatch, auth_header):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        sid = await _session(client, auth_header, tmp_path, monkeypatch)
        gone = await client.delete(f"/v1/sessions/{sid}", headers=auth_header)
        assert gone.status_code == 200, gone.text
        r = await client.post(
            f"/v1/sessions/{sid}/uploads",
            files={"file": ("notes.txt", b"hi", "text/plain")},
            headers=auth_header,
        )
        assert r.status_code == 404, r.text
        assert "deleted" in r.json()["detail"]


@pytest.mark.asyncio
async def test_upload_to_unknown_session_is_404(tmp_path, monkeypatch, auth_header):
    del tmp_path, monkeypatch
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.post(
            "/v1/sessions/00000000-0000-4000-8000-000000000000/uploads",
            files={"file": ("notes.txt", b"hi", "text/plain")},
            headers=auth_header,
        )
        assert r.status_code == 404, r.text


@pytest.mark.asyncio
async def test_upload_without_auth_is_401(tmp_path, monkeypatch, auth_header):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        sid = await _session(client, auth_header, tmp_path, monkeypatch)
        r = await client.post(
            f"/v1/sessions/{sid}/uploads",
            files={"file": ("notes.txt", b"hi", "text/plain")},
        )
        assert r.status_code == 401, r.text


@pytest.mark.asyncio
async def test_upload_empty_file_does_not_500(tmp_path, monkeypatch, auth_header):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        sid = await _session(client, auth_header, tmp_path, monkeypatch)
        r = await client.post(
            f"/v1/sessions/{sid}/uploads",
            files={"file": ("nothing.txt", b"", "text/plain")},
            headers=auth_header,
        )
        assert r.status_code == 200, r.text
        assert r.json()["kind"] == "empty"
        assert r.json()["ok"] is False
