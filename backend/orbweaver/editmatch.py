"""Fallback matching for StrReplace and small unified diffs for edit results.

Tiers, tried in order until one yields at least one match (Cline's
SEARCH/REPLACE strategy):

    exact        -> ``old_string`` is a substring
    line-number  -> a ``  12| `` / ``12: `` prefix stripped from every line
    line-trimmed -> each line compared ``.strip()``ed, indentation of the
                    file preserved in the replacement region
    block-anchor -> first and last line match (>= 3 lines), same line count
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass

DIFF_LINE_CAP = 200
NEAREST_CANDIDATES = 3
_LINE_NUMBER_PREFIX = re.compile(r"^\s*\d+[|:]\s?")

TIER_EXACT = "exact"
TIER_LINE_NUMBERS = "line-number prefixes stripped"
TIER_TRIMMED = "line-trimmed match (indentation preserved)"
TIER_ANCHOR = "first/last line anchor match"


@dataclass
class Span:
    start: int
    end: int
    replacement: str
    line: int  # 1-based first line of the span


@dataclass
class MatchResult:
    spans: list[Span]
    tier: str
    old_used: str

    @property
    def count(self) -> int:
        return len(self.spans)

    def apply(self, content: str, *, replace_all: bool) -> str:
        spans = self.spans if replace_all else self.spans[:1]
        out = content
        for span in sorted(spans, key=lambda s: s.start, reverse=True):
            out = out[: span.start] + span.replacement + out[span.end :]
        return out


def strip_line_numbers(text: str) -> str:
    lines = text.split("\n")
    if not all(_LINE_NUMBER_PREFIX.match(ln) for ln in lines if ln.strip()):
        return text
    return "\n".join(_LINE_NUMBER_PREFIX.sub("", ln) for ln in lines)


def _leading_ws(line: str) -> str:
    return line[: len(line) - len(line.lstrip())]


def reindent(
    new: str,
    old_first_indent: str,
    file_first_indent: str,
    indent_map: dict[str, str] | None = None,
) -> str:
    """Re-indent ``new`` so it sits at the file's indentation, not old_string's.

    Lines whose indentation appeared in old_string take the indentation of the
    matching file line (``indent_map``); other lines shift by the delta of the
    first matched line.
    """
    mapping = dict(indent_map or {})
    mapping.setdefault(old_first_indent, file_first_indent)
    if all(k == v for k, v in mapping.items()):
        return new
    out: list[str] = []
    for line in new.split("\n"):
        indent = _leading_ws(line)
        if not line.strip():
            out.append(line)
        elif indent in mapping:
            out.append(mapping[indent] + line[len(indent) :])
        elif old_first_indent and line.startswith(old_first_indent):
            out.append(file_first_indent + line[len(old_first_indent) :])
        elif not old_first_indent:
            out.append(file_first_indent + line)
        else:
            out.append(line)
    return "\n".join(out)


def _line_offsets(lines: list[str]) -> list[int]:
    offsets: list[int] = []
    pos = 0
    for line in lines:
        offsets.append(pos)
        pos += len(line) + 1
    return offsets


def _exact_spans(content: str, old: str, new: str) -> list[Span]:
    spans: list[Span] = []
    start = content.find(old)
    while start != -1:
        spans.append(Span(start, start + len(old), new, content.count("\n", 0, start) + 1))
        start = content.find(old, start + len(old))
    return spans


def _line_spans(
    content: str,
    old: str,
    new: str,
    *,
    anchor_only: bool,
) -> list[Span]:
    clines = content.split("\n")
    trailing_nl = old.endswith("\n")
    olines = old[:-1].split("\n") if trailing_nl else old.split("\n")
    n = len(olines)
    if n == 0 or (anchor_only and n < 3):
        return []
    if not anchor_only and not any(ln.strip() for ln in olines):
        return []
    offsets = _line_offsets(clines)
    otrim = [ln.strip() for ln in olines]
    spans: list[Span] = []
    for i in range(len(clines) - n + 1):
        if anchor_only:
            if clines[i].strip() != otrim[0] or clines[i + n - 1].strip() != otrim[-1]:
                continue
            if not otrim[0] or not otrim[-1]:
                continue
        elif any(clines[i + j].strip() != otrim[j] for j in range(n)):
            continue
        start = offsets[i]
        end = offsets[i + n - 1] + len(clines[i + n - 1])
        if trailing_nl and end < len(content) and content[end] == "\n":
            end += 1
        indent_map: dict[str, str] = {}
        for j in range(n):
            if olines[j].strip() and clines[i + j].strip() == otrim[j]:
                indent_map.setdefault(_leading_ws(olines[j]), _leading_ws(clines[i + j]))
        replacement = reindent(new, _leading_ws(olines[0]), _leading_ws(clines[i]), indent_map)
        spans.append(Span(start, end, replacement, i + 1))
    return spans


def find_replacement(content: str, old: str, new: str) -> MatchResult | None:
    """Locate ``old`` in ``content`` using the fallback tiers; None when nothing matches."""
    spans = _exact_spans(content, old, new)
    if spans:
        return MatchResult(spans, TIER_EXACT, old)
    stripped = strip_line_numbers(old)
    tiers: list[str] = []
    if stripped != old and stripped.strip():
        tiers.append(TIER_LINE_NUMBERS)
        spans = _exact_spans(content, stripped, new)
        if spans:
            return MatchResult(spans, TIER_LINE_NUMBERS, stripped)
    candidate = stripped if stripped.strip() else old
    spans = _line_spans(content, candidate, new, anchor_only=False)
    if spans:
        return MatchResult(spans, " + ".join(tiers + [TIER_TRIMMED]), candidate)
    spans = _line_spans(content, candidate, new, anchor_only=True)
    if spans:
        return MatchResult(spans, " + ".join(tiers + [TIER_ANCHOR]), candidate)
    return None


def nearest_lines(content: str, old: str, *, limit: int = NEAREST_CANDIDATES) -> list[str]:
    """``lineno|text`` for the file lines closest to old_string's first non-blank line."""
    probe = next((ln.strip() for ln in strip_line_numbers(old).split("\n") if ln.strip()), "")
    if not probe:
        return []
    clines = content.split("\n")
    by_text: dict[str, int] = {}
    for i, ln in enumerate(clines, 1):
        key = ln.strip()
        if key and key not in by_text:
            by_text[key] = i
    close = difflib.get_close_matches(probe, list(by_text), n=limit, cutoff=0.5)
    if not close:
        return []
    return [f"{by_text[text]}|{clines[by_text[text] - 1]}" for text in close]


def unified_diff(before: str, after: str, path: str, *, cap: int = DIFF_LINE_CAP) -> str:
    lines = list(
        difflib.unified_diff(
            before.split("\n"),
            after.split("\n"),
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
            lineterm="",
        )
    )
    if len(lines) > cap:
        rest = len(lines) - cap
        lines = lines[:cap] + [f"[diff truncated; {rest} more lines]"]
    return "\n".join(lines)
