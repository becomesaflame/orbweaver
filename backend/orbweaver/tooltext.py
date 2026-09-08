"""Format tool output so one round carries usable text instead of chrome or a clip."""

from __future__ import annotations

import re
from html import unescape
from html.parser import HTMLParser
from typing import Any

READ_CHAR_CAP = 20_000
READ_DEFAULT_LINES = 400
WEBFETCH_CHAR_CAP = 24_000
WEBFETCH_RAW_CAP = 500_000


def _as_int(value: Any, default: int | None) -> int | None:
    if value is None or value == "":
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def format_read(
    text: str,
    *,
    path: str = "",
    offset: Any = None,
    limit: Any = None,
    char_cap: int = READ_CHAR_CAP,
) -> str:
    """Numbered window over a file, with an explicit next-offset footer when truncated."""
    lines = text.splitlines()
    total = len(lines)
    if total == 0:
        label = path or "file"
        return f"(empty {label})"
    raw_off = _as_int(offset, 1) or 1
    if raw_off < 0:
        start = max(0, total + raw_off)
    else:
        start = min(max(0, raw_off - 1), total)
    max_lines = _as_int(limit, READ_DEFAULT_LINES)
    if max_lines is None or max_lines < 0:
        max_lines = READ_DEFAULT_LINES
    window = lines[start : start + max_lines]
    clipped = False
    while window:
        body = _number_lines(window, start + 1)
        if len(body) <= char_cap:
            break
        if len(window) == 1:
            body = body[:char_cap]
            clipped = True
            break
        window = window[:-1]
    shown = len(window)
    end_line = start + shown
    header = f"{path}: " if path else ""
    header += f"lines {start + 1}-{end_line} of {total}"
    out = header + "\n" + body
    if end_line < total:
        out += (
            f"\n\n[truncated; {total - end_line} lines remain. "
            f"Read offset={end_line + 1} to continue.]"
        )
    elif clipped:
        out += f"\n\n[truncated to {char_cap} chars on this line.]"
    return out


def _number_lines(lines: list[str], first: int) -> str:
    width = len(str(first + len(lines) - 1))
    return "\n".join(f"{i:>{width}}|{line}" for i, line in enumerate(lines, first))


_SKIP_TAGS = frozenset({"script", "style", "noscript", "svg", "template", "iframe"})
_BLOCK_TAGS = frozenset(
    {
        "p",
        "div",
        "br",
        "li",
        "tr",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "section",
        "article",
        "header",
        "footer",
        "pre",
        "blockquote",
        "ul",
        "ol",
        "table",
        "thead",
        "tbody",
    }
)


class _HTMLText(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._skip = 0
        self._title_parts: list[str] = []
        self._in_title = False
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _SKIP_TAGS:
            self._skip += 1
            return
        if tag == "title":
            self._in_title = True
        if self._skip:
            return
        if tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP_TAGS and self._skip:
            self._skip -= 1
            return
        if tag == "title":
            self._in_title = False
        if self._skip:
            return
        if tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self._title_parts.append(data)
        if self._skip:
            return
        self.parts.append(data)

    def text(self) -> str:
        title = unescape("".join(self._title_parts)).strip()
        body = unescape("".join(self.parts))
        body = re.sub(r"[ \t]+", " ", body)
        body = re.sub(r"\n[ \t]+", "\n", body)
        body = re.sub(r"\n{3,}", "\n\n", body).strip()
        if title and title.lower() not in body[:800].lower():
            return f"{title}\n\n{body}".strip()
        return body


def looks_html(content_type: str, body: str) -> bool:
    ctype = (content_type or "").lower()
    if "html" in ctype:
        return True
    head = body.lstrip()[:256].lower()
    return head.startswith("<!doctype html") or head.startswith("<html")


def html_to_text(body: str) -> str:
    parser = _HTMLText()
    try:
        parser.feed(body)
        parser.close()
    except Exception:
        return re.sub(r"<[^>]+>", " ", body)
    return parser.text()


def format_webfetch(
    url: str,
    status_code: int,
    content_type: str,
    body: str,
    *,
    char_cap: int = WEBFETCH_CHAR_CAP,
) -> str:
    raw = body[:WEBFETCH_RAW_CAP]
    extracted = False
    if looks_html(content_type, raw):
        text = html_to_text(raw)
        extracted = True
        if len(raw) >= 5000 and len(text) < 800:
            note = (
                "No readable article text; this looks like a JavaScript-rendered page. "
                "Do not keep fetching nearby docs URLs expecting the article. "
                "Prefer a raw/source URL, or answer from what you already know."
            )
            header = _fetch_header(url, status_code, content_type, extracted=True, truncated=False)
            return f"{header}\n{note}\n"
    else:
        text = raw
    truncated = len(text) > char_cap
    text = text[:char_cap]
    header = _fetch_header(url, status_code, content_type, extracted=extracted, truncated=truncated)
    return f"{header}\n{text}"


def _fetch_header(
    url: str,
    status_code: int,
    content_type: str,
    *,
    extracted: bool,
    truncated: bool,
) -> str:
    bits = [f"HTTP {status_code} {url}"]
    if content_type:
        bits.append(f"content-type: {content_type.split(';')[0].strip()}")
    if extracted:
        bits.append("extracted readable text from HTML")
    if truncated:
        bits.append(f"truncated to {WEBFETCH_CHAR_CAP} chars")
    return " | ".join(bits)


def grep_regex(pattern: str) -> re.Pattern[str] | None:
    """Compile a grep pattern. Models often send `a\\|b` when they mean `a|b`."""
    spec = (pattern or "").strip()
    if not spec:
        return None
    if r"\|" in spec:
        spec = spec.replace(r"\|", "|")
    try:
        return re.compile(spec)
    except re.error:
        return re.compile(re.escape(pattern))
