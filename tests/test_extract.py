"""Extraction must never raise and never blow the context budget."""

import builtins
import io
import zipfile

import pytest
from PIL import Image

from orbweaver.extract import (
    KIND_BINARY,
    KIND_DOCX,
    KIND_EMPTY,
    KIND_IMAGE,
    KIND_PDF,
    KIND_TEXT,
    MAX_TEXT_CHARS,
    detect_kind,
    extract_text,
    human_size,
)

try:  # pypdf is a declared dependency but extraction degrades without it.
    import pypdf as _pypdf
except ImportError:  # pragma: no cover - exercised on hosts without pypdf
    _pypdf = None

needs_pypdf = pytest.mark.skipif(_pypdf is None, reason="pypdf is not installed")


def _png_bytes(size: int = 8) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (size, size), (10, 120, 200)).save(buf, format="PNG")
    return buf.getvalue()


def _pdf_bytes(pages: list[str]) -> bytes:
    """Hand-rolled uncompressed PDF so the fixture needs no writer library."""
    objects: list[bytes] = []
    page_count = len(pages)
    kids = " ".join(f"{3 + 2 * i} 0 R" for i in range(page_count))
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objects.append(f"<< /Type /Pages /Kids [{kids}] /Count {page_count} >>".encode())
    for i, body in enumerate(pages):
        content = f"BT /F1 12 Tf 72 720 Td ({body}) Tj ET".encode()
        objects.append(
            (
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                f"/Resources << /Font << /F1 << /Type /Font /Subtype /Type1 "
                f"/BaseFont /Helvetica >> >> >> /Contents {4 + 2 * i} 0 R >>"
            ).encode()
        )
        objects.append(
            b"<< /Length " + str(len(content)).encode() + b" >>\nstream\n" + content + b"\nendstream"
        )
    out = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = []
    for n, obj in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{n} 0 obj\n".encode() + obj + b"\nendobj\n"
    xref_at = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_at}\n%%EOF\n"
    ).encode()
    return bytes(out)


def _docx_bytes(paragraphs: list[str], *, member: str = "word/document.xml") -> bytes:
    runs = "".join(
        f"<w:p><w:r><w:t>{p}</w:t></w:r></w:p>" for p in paragraphs
    )
    xml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        f"<w:body>{runs}</w:body></w:document>"
    )
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("[Content_Types].xml", "<Types/>")
        zf.writestr(member, xml)
    return buf.getvalue()


# ------------------------------------------------------------------ plain text


def test_extract_utf8_text():
    result = extract_text("hello — world\nsecond line\n".encode(), "notes.txt")
    assert result.kind == KIND_TEXT
    assert result.ok
    assert "hello — world" in result.text
    assert not result.truncated


def test_extract_latin1_fallback_keeps_bytes():
    # 0xe9 alone is invalid UTF-8; latin-1 maps it to "é" instead of U+FFFD.
    result = extract_text(b"caf\xe9 log line", "app.log")
    assert result.kind == KIND_TEXT
    assert result.text == "café log line"


@pytest.mark.parametrize(
    "name,body",
    [
        ("main.py", b"def f():\n    return 1\n"),
        ("data.json", b'{"a": [1, 2, 3]}'),
        ("README.md", b"# Title\n\nbody\n"),
        ("rows.csv", b"a,b\n1,2\n"),
        ("conf.yaml", b"key: value\nlist:\n  - one\n"),
        ("server.log", b"2026-01-01 ERROR boom\n"),
    ],
)
def test_extract_code_and_data_formats(name, body):
    result = extract_text(body, name)
    assert result.kind == KIND_TEXT
    assert result.ok
    assert result.text.strip() == body.decode().strip()


def test_extract_truncates_at_cap_and_says_so():
    result = extract_text(b"x" * (MAX_TEXT_CHARS + 5_000), "huge.txt")
    assert result.truncated
    assert "truncated" in result.note
    # The marker is appended after the cap, so the body itself stays capped.
    assert result.text.startswith("x" * 100)
    assert len(result.text) < MAX_TEXT_CHARS + 500
    assert "truncated" in result.text


def test_extract_empty_file_is_not_ok_but_does_not_raise():
    result = extract_text(b"", "empty.txt")
    assert result.kind == KIND_EMPTY
    assert not result.ok
    assert "empty" in result.note


# ------------------------------------------------------------------------ PDF


@needs_pypdf
def test_extract_pdf_text_per_page():
    result = extract_text(_pdf_bytes(["first page body", "second page body"]), "doc.pdf")
    assert result.kind == KIND_PDF
    assert result.ok
    assert result.pages == 2
    assert "first page body" in result.text
    assert "second page body" in result.text
    assert "--- page 2 ---" in result.text


@needs_pypdf
def test_extract_pdf_caps_a_huge_document(monkeypatch):
    monkeypatch.setattr("orbweaver.extract.MAX_TEXT_CHARS", 200)
    result = extract_text(_pdf_bytes([f"page number {i} content" for i in range(40)]), "big.pdf")
    assert result.truncated
    assert len(result.text) < 400
    # The cap must not hide how big the document really was.
    assert result.pages == 40


def test_extract_corrupt_pdf_returns_note(monkeypatch):
    result = extract_text(b"%PDF-1.4\nnot really a pdf at all", "broken.pdf")
    assert result.kind == KIND_PDF
    assert not result.ok
    assert result.note
    assert not result.text


def test_extract_pdf_without_pypdf_degrades(monkeypatch):
    """Guarded optional import: no pypdf must mean a note, not a traceback."""
    real_import = builtins.__import__

    def no_pypdf(name, *args, **kwargs):
        if name == "pypdf":
            raise ImportError("no module named pypdf")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_pypdf)
    result = extract_text(_pdf_bytes(["body"]), "doc.pdf")
    assert result.kind == KIND_PDF
    assert not result.ok
    assert "PDF support unavailable" in result.note


@needs_pypdf
def test_extract_pdf_with_no_text_layer_is_reported():
    """A scanned PDF is valid but yields nothing; say that instead of "ok, empty"."""
    result = extract_text(_pdf_bytes([" "]), "scan.pdf")
    assert result.kind == KIND_PDF
    assert not result.ok
    assert "no extractable text" in result.note


# ----------------------------------------------------------------------- DOCX


def test_extract_docx_paragraphs():
    result = extract_text(_docx_bytes(["First para", "Second para"]), "report.docx")
    assert result.kind == KIND_DOCX
    assert result.ok
    assert result.text.splitlines() == ["First para", "Second para"]


def test_extract_docx_unescapes_entities():
    result = extract_text(_docx_bytes(["a &lt; b &amp;&amp; c &gt; d"]), "r.docx")
    assert result.text == "a < b && c > d"


def test_extract_docx_without_document_xml_is_reported():
    data = _docx_bytes(["x"], member="word/other.xml")
    result = extract_text(data, "fake.docx")
    # Without word/document.xml it is just a zip, so it classifies as binary.
    assert result.kind == KIND_BINARY
    assert not result.ok
    assert "zip archive" in result.note


def test_extract_bogus_zip_named_docx_does_not_raise():
    result = extract_text(b"PK\x03\x04garbage-not-a-zip", "broken.docx")
    assert not result.ok
    assert result.note


def test_extract_docx_truncates(monkeypatch):
    monkeypatch.setattr("orbweaver.extract.MAX_TEXT_CHARS", 50)
    result = extract_text(_docx_bytes(["y" * 500]), "long.docx")
    assert result.truncated
    assert "truncated" in result.note


# --------------------------------------------------------------- images, binary


def test_extract_routes_images_to_the_vision_path():
    result = extract_text(_png_bytes(), "shot.png")
    assert result.kind == KIND_IMAGE
    assert result.is_image
    assert result.text == ""
    assert result.media_type == "image/png"


def test_extract_binary_blob_is_described_not_dumped():
    blob = b"\x7fELF" + bytes(range(256)) * 4
    result = extract_text(blob, "a.out")
    assert result.kind == KIND_BINARY
    assert not result.ok
    assert "ELF executable" in result.note
    assert result.text == ""


def test_extract_xlsx_is_named_specifically():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("xl/workbook.xml", "<workbook/>")
    result = extract_text(buf.getvalue(), "sheet.xlsx")
    assert result.kind == KIND_BINARY
    assert "Excel workbook" in result.note


def test_extract_nul_bytes_are_binary_even_without_magic():
    result = extract_text(b"plain looking\x00but not", "notes.txt")
    assert result.kind == KIND_BINARY
    assert not result.ok


# ------------------------------------------------------------------- sniffing


@needs_pypdf
def test_pdf_wearing_a_txt_extension_is_still_extracted():
    result = extract_text(_pdf_bytes(["hidden pdf body"]), "actually.txt")
    assert result.kind == KIND_PDF
    assert "hidden pdf body" in result.text


def test_text_wearing_a_pdf_extension_is_read_as_text():
    result = extract_text(b"just a log line\n", "mislabeled.pdf")
    assert result.kind == KIND_TEXT
    assert result.text.strip() == "just a log line"


def test_png_wearing_a_docx_extension_is_still_an_image():
    result = extract_text(_png_bytes(), "sneaky.docx")
    assert result.kind == KIND_IMAGE


def test_detect_kind_on_empty_and_unknown():
    assert detect_kind(b"", "x.txt") == KIND_EMPTY
    assert detect_kind(b"hello", "x") == KIND_TEXT
    assert detect_kind(b"%PDF-1.7\n...", "x") == KIND_PDF


# -------------------------------------------------------------------- helpers


def test_human_size():
    assert human_size(512) == "512 B"
    assert human_size(2048) == "2.0 KB"
    assert human_size(5 * 1024 * 1024) == "5.0 MB"


def test_preview_marks_where_it_stopped():
    result = extract_text(b"z" * 10_000, "big.txt")
    preview = result.preview(limit=100)
    assert len(preview) < 200
    assert "more in the stored file" in preview


def test_payload_is_json_safe():
    payload = extract_text(b"hello", "a.txt").payload()
    assert payload["kind"] == KIND_TEXT
    assert payload["chars"] == 5
    assert payload["truncated"] is False
