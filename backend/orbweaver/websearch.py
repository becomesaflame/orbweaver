"""WebSearch: DuckDuckGo HTML by default, Brave Search when a key is set."""

from __future__ import annotations

import re
from html import unescape
from html.parser import HTMLParser
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

from orbweaver import __version__
from orbweaver.config import settings

DDG_HTML = "https://html.duckduckgo.com/html/"
BRAVE_SEARCH = "https://api.search.brave.com/res/v1/web/search"
MAX_RESULTS = 10
DEFAULT_RESULTS = 8
USER_AGENT = f"Orbweaver/{__version__}"


class WebSearchError(Exception):
    """Search provider failed or returned nothing usable."""


def clamp_results(value: Any) -> int:
    try:
        n = int(value)
    except (TypeError, ValueError):
        n = DEFAULT_RESULTS
    return max(1, min(n, MAX_RESULTS))


def unwrap_ddg_url(href: str) -> str:
    raw = (href or "").strip()
    if not raw:
        return ""
    if raw.startswith("//"):
        raw = "https:" + raw
    parsed = urlparse(raw)
    host = (parsed.netloc or "").lower()
    if "duckduckgo.com" in host and parsed.path.startswith("/l"):
        uddg = parse_qs(parsed.query).get("uddg") or []
        if uddg:
            return unquote(uddg[0])
    return raw


class _DDGHTML(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.hits: list[dict[str, str]] = []
        self._in_title = False
        self._in_snip = False
        self._href = ""
        self._title: list[str] = []
        self._snip: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        d = {k: (v or "") for k, v in attrs}
        cls = d.get("class", "")
        if tag == "a" and "result__a" in cls.split():
            self._in_title = True
            self._href = d.get("href", "")
            self._title = []
        elif "result__snippet" in cls.split():
            self._in_snip = True
            self._snip = []

    def handle_endtag(self, tag: str) -> None:
        if self._in_title and tag == "a":
            self._in_title = False
            title = unescape("".join(self._title)).strip()
            url = unwrap_ddg_url(self._href)
            if title and url:
                self.hits.append({"title": title, "url": url, "snippet": ""})
        if self._in_snip and tag in {"a", "div", "td", "span"}:
            self._in_snip = False
            snippet = unescape("".join(self._snip)).strip()
            snippet = re.sub(r"\s+", " ", snippet)
            if snippet and self.hits and not self.hits[-1]["snippet"]:
                self.hits[-1]["snippet"] = snippet

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self._title.append(data)
        elif self._in_snip:
            self._snip.append(data)


def parse_ddg_html(body: str) -> list[dict[str, str]]:
    parser = _DDGHTML()
    try:
        parser.feed(body)
        parser.close()
    except Exception:
        return []
    return parser.hits


def parse_brave_json(payload: Any) -> list[dict[str, str]]:
    if not isinstance(payload, dict):
        return []
    web = payload.get("web") or {}
    rows = web.get("results") if isinstance(web, dict) else None
    if not isinstance(rows, list):
        return []
    out: list[dict[str, str]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        title = str(row.get("title") or "").strip()
        url = str(row.get("url") or "").strip()
        snippet = str(row.get("description") or row.get("snippet") or "").strip()
        if title and url:
            out.append({"title": title, "url": url, "snippet": snippet})
    return out


def format_hits(query: str, provider: str, hits: list[dict[str, str]]) -> str:
    if not hits:
        return (
            f"WebSearch {provider}: {query}\nNo results. Try a shorter query, or WebFetch a URL "
            "you already know."
        )
    lines = [
        f"WebSearch {provider}: {query} ({len(hits)} results)",
        "Use WebFetch on a specific result URL for the full page. Do not guess nearby docs URLs.",
        "",
    ]
    for i, hit in enumerate(hits, 1):
        lines.append(f"{i}. {hit['title']}")
        lines.append(f"   {hit['url']}")
        if hit.get("snippet"):
            lines.append(f"   {hit['snippet']}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _headers() -> dict[str, str]:
    return {"User-Agent": USER_AGENT, "Accept": "text/html,application/json"}


def search_duckduckgo(query: str, limit: int) -> list[dict[str, str]]:
    import httpx

    r = httpx.get(
        DDG_HTML,
        params={"q": query},
        headers=_headers(),
        timeout=20.0,
        follow_redirects=True,
    )
    if r.status_code >= 400:
        raise WebSearchError(
            f"DuckDuckGo HTTP {r.status_code}. "
            "Set ORBWEAVER_BRAVE_API_KEY to use Brave Search instead."
        )
    hits = parse_ddg_html(r.text)
    return hits[:limit]


def search_brave(query: str, limit: int) -> list[dict[str, str]]:
    import httpx

    key = settings.brave_api_key
    if not key:
        raise WebSearchError(
            "Brave Search is selected but ORBWEAVER_BRAVE_API_KEY is empty."
        )
    r = httpx.get(
        BRAVE_SEARCH,
        params={"q": query, "count": limit},
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "application/json",
            "X-Subscription-Token": key,
        },
        timeout=20.0,
        follow_redirects=True,
    )
    if r.status_code >= 400:
        raise WebSearchError(f"Brave Search HTTP {r.status_code}: {r.text[:200]}")
    try:
        payload = r.json()
    except ValueError as e:
        raise WebSearchError(f"Brave Search returned non-JSON: {e}") from e
    return parse_brave_json(payload)[:limit]


def search(query: str, *, max_results: int = DEFAULT_RESULTS) -> tuple[list[dict[str, str]], str]:
    limit = clamp_results(max_results)
    provider = settings.search_provider
    if provider == "brave":
        return search_brave(query, limit), "brave"
    return search_duckduckgo(query, limit), "duckduckgo"


def run_websearch(inp: dict[str, Any]) -> str:
    query = str(inp.get("query") or "").strip()
    if not query:
        return "WebSearch requires a query."
    limit = clamp_results(inp.get("max_results"))
    try:
        hits, provider = search(query, max_results=limit)
    except WebSearchError as e:
        return f"WebSearch failed: {e}"
    except Exception as e:
        return f"WebSearch failed: {e}"
    return format_hits(query, provider, hits)
