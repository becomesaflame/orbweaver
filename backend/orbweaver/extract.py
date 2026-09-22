"""Turn an uploaded file into text the agent can read.

Uploads arrive from the web composer and from Telegram documents. Images are
deliberately *not* handled here: they belong to the vision path in
``orbweaver.image``, which resizes them and emits a workspace-path marker.
Everything else is decoded to text, or reported as an unsupported binary.

Two rules drive the shape of this module:

* **Never raise.** A corrupt PDF, a ``.docx`` that is not a zip, a truncated
  upload — none of that may 500 the gateway or kill a turn. Every failure comes
  back as an ``ExtractResult`` with ``ok=False`` and a human-readable ``note``.
* **Always cap.** A 500-page PDF would otherwise swallow the context window on
  the first turn. Extraction stops at ``MAX_TEXT_CHARS`` and says so.

Detection is by content first, extension second: a ``.txt`` that is really a
PDF is extracted as a PDF, and a ``.pdf`` that is really a log file is read as
text.
"""

from __future__ import annotations

import logging
import re
import zipfile
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from orbweaver.image import sniff_media_type

log = logging.getLogger(__name__)

# ~200k chars is roughly 50k tokens: large enough for a real document, small
# enough that one upload cannot exhaust the turn's budget on its own.
MAX_TEXT_CHARS = 200_000
# How much of the extracted text is inlined into the turn message. The rest
# stays in the workspace for the agent to Read on demand.
INLINE_TEXT_CHARS = 4_000
# Same probe window workspace.read_text_for_tool uses to call a file binary.
BINARY_PROBE_BYTES = 8192

KIND_TEXT = "text"
KIND_PDF = "pdf"
KIND_DOCX = "docx"
KIND_IMAGE = "image"
KIND_BINARY = "binary"
KIND_EMPTY = "empty"

TRUNCATION_MARKER = "\n\n[... truncated: extraction stopped at {cap} characters ...]"

_PDF_MAGIC = b"%PDF-"
_ZIP_MAGIC = (b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08")
_DOCX_MEMBER = "word/document.xml"

# Magic numbers only used to *name* a binary we refuse to extract, so the note
# can say "zip archive" instead of "unsupported binary".
_BINARY_LABELS: tuple[tuple[bytes, str], ...] = (
    (b"\x1f\x8b", "gzip archive"),
    (b"BZh", "bzip2 archive"),
    (b"\xfd7zXZ", "xz archive"),
    (b"7z\xbc\xaf\x27\x1c", "7-zip archive"),
    (b"\x7fELF", "ELF executable"),
    (b"MZ", "Windows executable"),
    (b"\xca\xfe\xba\xbe", "Java class or Mach-O binary"),
    (b"OggS", "Ogg media"),
    (b"ID3", "MP3 audio"),
    (b"\x00\x00\x00\x18ftyp", "MP4 video"),
    (b"\x00\x00\x00\x20ftyp", "MP4 video"),
    (b"SQLite format 3", "SQLite database"),
    (b"%!PS", "PostScript document"),
    (b"\xd0\xcf\x11\xe0", "legacy Microsoft Office document (.doc/.xls)"),
)

# Office Open XML zips that are not Word documents. Recognised so the note is
# specific, not because we extract them.
_OOXML_LABELS: tuple[tuple[str, str], ...] = (
    ("xl/workbook.xml", "Excel workbook (.xlsx)"),
    ("ppt/presentation.xml", "PowerPoint deck (.pptx)"),
    ("content.xml", "OpenDocument file"),
)

_DOCX_PARA_END = re.compile(r"</w:p\s*>")
_DOCX_BREAK = re.compile(r"<w:(?:br|cr)\b[^>]*/?>")
_DOCX_TAB = re.compile(r"<w:tab\b[^>]*/?>")
_XML_TAG = re.compile(r"<[^>]*>")
_XML_ENTITIES = (
    ("&lt;", "<"),
    ("&gt;", ">"),
    ("&quot;", '"'),
    ("&apos;", "'"),
    ("&#10;", "\n"),
    ("&#9;", "\t"),
    # Ampersand last: undoing it first would let "&amp;lt;" become "<".
    ("&amp;", "&"),
)


@dataclass
class ExtractResult:
    """Outcome of one extraction attempt. Never an exception.

    ``ok`` is False when there is no usable text (binary, empty, corrupt,
    missing optional dependency); ``note`` then explains why in one line the
    UI and the agent can both read.
    """

    kind: str
    text: str = ""
    note: str = ""
    truncated: bool = False
    size: int = 0
    pages: int | None = None
    media_type: str = ""
    ok: bool = True

    @property
    def is_image(self) -> bool:
        return self.kind == KIND_IMAGE

    def preview(self, limit: int = INLINE_TEXT_CHARS) -> str:
        """Leading slice of the extracted text, for a chip or a turn message."""
        if len(self.text) <= limit:
            return self.text
        return self.text[:limit].rstrip() + "\n[... more in the stored file ...]"

    def payload(self) -> dict[str, object]:
        """JSON-safe summary for the upload endpoint."""
        return {
            "kind": self.kind,
            "ok": self.ok,
            "note": self.note,
            "truncated": self.truncated,
            "chars": len(self.text),
            "pages": self.pages,
            "media_type": self.media_type,
        }


def human_size(nbytes: int) -> str:
    if nbytes < 1024:
        return f"{nbytes} B"
    if nbytes < 1024 * 1024:
        return f"{nbytes / 1024:.1f} KB"
    return f"{nbytes / (1024 * 1024):.1f} MB"


def _cap(text: str) -> tuple[str, bool]:
    if len(text) <= MAX_TEXT_CHARS:
        return text, False
    return text[:MAX_TEXT_CHARS] + TRUNCATION_MARKER.format(cap=MAX_TEXT_CHARS), True


def _decode_text(data: bytes) -> str:
    """Strict UTF-8, else latin-1.

    latin-1 rather than ``errors="replace"``: a Windows-1252 log decodes to
    readable text under latin-1, where replacement would pepper it with U+FFFD
    and lose the bytes for good. latin-1 maps all 256 byte values, so this
    always returns something.
    """
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("latin-1")


def _looks_binary(data: bytes) -> bool:
    return b"\x00" in data[:BINARY_PROBE_BYTES]


def _zip_members(data: bytes) -> list[str] | None:
    """Member names, or None when the bytes are not a readable zip."""
    try:
        with zipfile.ZipFile(BytesIO(data)) as zf:
            return zf.namelist()
    except (zipfile.BadZipFile, OSError, ValueError, EOFError):
        return None
    except Exception:
        log.debug("zip inspection failed", exc_info=True)
        return None


def _binary_label(data: bytes, filename: str) -> str:
    for magic, label in _BINARY_LABELS:
        if data.startswith(magic):
            return label
    media = sniff_media_type(data)
    if media != "application/octet-stream":
        return media
    suffix = Path(filename).suffix.lower().lstrip(".")
    return f"{suffix} file" if suffix else "unknown binary"


def detect_kind(data: bytes, filename: str) -> str:
    """Classify by magic bytes, falling back to a NUL probe then text.

    Extension is only consulted for the zip family, where the container alone
    cannot tell a .docx from a .jar.
    """
    if not data:
        return KIND_EMPTY
    if data.startswith(_PDF_MAGIC):
        return KIND_PDF
    if sniff_media_type(data) != "application/octet-stream":
        return KIND_IMAGE
    if data.startswith(_ZIP_MAGIC):
        members = _zip_members(data)
        if members and _DOCX_MEMBER in members:
            return KIND_DOCX
        return KIND_BINARY
    if _looks_binary(data):
        return KIND_BINARY
    return KIND_TEXT


def _extract_plain(data: bytes) -> ExtractResult:
    text, truncated = _cap(_decode_text(data))
    return ExtractResult(
        kind=KIND_TEXT,
        text=text,
        truncated=truncated,
        size=len(data),
        media_type="text/plain",
        ok=True,
        note="truncated" if truncated else "",
    )


def _extract_pdf(data: bytes) -> ExtractResult:
    try:
        import pypdf
    except ImportError:
        return ExtractResult(
            kind=KIND_PDF,
            size=len(data),
            media_type="application/pdf",
            ok=False,
            note=(
                "PDF support unavailable: pypdf is not installed. "
                "The file is stored in the workspace but no text was extracted."
            ),
        )
    try:
        reader = pypdf.PdfReader(BytesIO(data))
        page_count = len(reader.pages)
    except Exception as e:
        return ExtractResult(
            kind=KIND_PDF,
            size=len(data),
            media_type="application/pdf",
            ok=False,
            note=f"could not read PDF: {type(e).__name__}: {e}"[:400],
        )
    chunks: list[str] = []
    total = 0
    truncated = False
    read_pages = 0
    failed_pages = 0
    for index in range(page_count):
        if total >= MAX_TEXT_CHARS:
            truncated = True
            break
        try:
            body = reader.pages[index].extract_text() or ""
        except Exception:
            # One unreadable page must not lose the other 499.
            failed_pages += 1
            log.debug("pdf page %d extraction failed", index + 1, exc_info=True)
            continue
        read_pages += 1
        if not body.strip():
            continue
        chunk = f"--- page {index + 1} ---\n{body.strip()}"
        chunks.append(chunk)
        total += len(chunk) + 2
    text, hard_truncated = _cap("\n\n".join(chunks))
    truncated = truncated or hard_truncated
    notes: list[str] = []
    if truncated:
        notes.append(f"truncated at {MAX_TEXT_CHARS} characters")
    if failed_pages:
        notes.append(f"{failed_pages} of {page_count} pages could not be parsed")
    if not text.strip():
        notes.append("no extractable text (likely a scanned or image-only PDF)")
    return ExtractResult(
        kind=KIND_PDF,
        text=text,
        truncated=truncated,
        size=len(data),
        pages=page_count,
        media_type="application/pdf",
        ok=bool(text.strip()),
        note="; ".join(notes),
    )


def _docx_xml_to_text(xml: str) -> str:
    """Flatten WordprocessingML runs into plain text.

    Paragraph and break tags become newlines *before* the blanket tag strip, so
    the result keeps its line structure. Entities are unescaped last, after the
    angle brackets that could be mistaken for tags are gone.
    """
    body = _DOCX_PARA_END.sub("\n", xml)
    body = _DOCX_BREAK.sub("\n", body)
    body = _DOCX_TAB.sub("\t", body)
    body = _XML_TAG.sub("", body)
    for entity, char in _XML_ENTITIES:
        body = body.replace(entity, char)
    lines = [line.rstrip() for line in body.split("\n")]
    return "\n".join(lines).strip()


def _extract_docx(data: bytes) -> ExtractResult:
    """Read word/document.xml out of the zip with stdlib only.

    A .docx is a zip of XML; ``python-docx`` would add a dependency for a tag
    strip. The cost is that fancier content (tables keep no column structure,
    footnotes and headers live in other parts and are skipped) comes out flatter
    than a real parser would give.
    """
    media = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    try:
        with zipfile.ZipFile(BytesIO(data)) as zf:
            raw = zf.read(_DOCX_MEMBER)
    except KeyError:
        return ExtractResult(
            kind=KIND_DOCX,
            size=len(data),
            media_type=media,
            ok=False,
            note="not a Word document: the zip has no word/document.xml",
        )
    except Exception as e:
        return ExtractResult(
            kind=KIND_DOCX,
            size=len(data),
            media_type=media,
            ok=False,
            note=f"could not read DOCX: {type(e).__name__}: {e}"[:400],
        )
    text, truncated = _cap(_docx_xml_to_text(_decode_text(raw)))
    notes: list[str] = []
    if truncated:
        notes.append(f"truncated at {MAX_TEXT_CHARS} characters")
    if not text.strip():
        notes.append("no extractable text in the document body")
    return ExtractResult(
        kind=KIND_DOCX,
        text=text,
        truncated=truncated,
        size=len(data),
        media_type=media,
        ok=bool(text.strip()),
        note="; ".join(notes),
    )


def _extract_binary(data: bytes, filename: str) -> ExtractResult:
    label = _binary_label(data, filename)
    members = _zip_members(data) if data.startswith(_ZIP_MAGIC) else None
    if members is not None:
        label = "zip archive"
        for member, ooxml in _OOXML_LABELS:
            if member in members:
                label = ooxml
                break
    return ExtractResult(
        kind=KIND_BINARY,
        size=len(data),
        media_type=sniff_media_type(data),
        ok=False,
        note=(
            f"unsupported binary: {label}, {human_size(len(data))}. "
            "Stored in the workspace but not extracted as text."
        ),
    )


def extract_text(data: bytes, filename: str) -> ExtractResult:
    """Extract text from an uploaded file. Never raises.

    ``filename`` is advisory: detection reads the bytes first, so a mislabelled
    upload still lands in the right extractor. Images return ``KIND_IMAGE``
    with no text — callers route those to ``orbweaver.image`` instead.
    """
    payload = data or b""
    kind = detect_kind(payload, filename)
    try:
        if kind == KIND_EMPTY:
            return ExtractResult(
                kind=KIND_EMPTY, size=0, ok=False, note="the uploaded file is empty"
            )
        if kind == KIND_IMAGE:
            return ExtractResult(
                kind=KIND_IMAGE,
                size=len(payload),
                media_type=sniff_media_type(payload),
                ok=True,
                note="image: handled as vision input, not text",
            )
        if kind == KIND_PDF:
            return _extract_pdf(payload)
        if kind == KIND_DOCX:
            return _extract_docx(payload)
        if kind == KIND_TEXT:
            return _extract_plain(payload)
        return _extract_binary(payload, filename)
    except Exception as e:
        # Belt and braces: the per-format helpers already catch their own
        # failures, and a malformed upload must never reach the caller as an
        # exception.
        log.exception("extraction failed for %s", filename)
        return ExtractResult(
            kind=kind,
            size=len(payload),
            ok=False,
            note=f"extraction failed: {type(e).__name__}: {e}"[:400],
        )
