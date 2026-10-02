"""WebSearch: DuckDuckGo HTML, then Brave's public results page.

Brave's JSON API is used first when ``ORBWEAVER_BRAVE_API_KEY`` is set and the
provider is ``brave``. DuckDuckGo's HTML endpoint often answers a datacenter
IP with HTTP 202 and an ``anomaly.js`` bot wall. That page has no result
links; treating it as an empty SERP made every query look like "No results".
"""

from __future__ import annotations

import logging
import re
from html import unescape
from html.parser import HTMLParser
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

from orbweaver import __version__
from orbweaver.config import settings

log = logging.getLogger(__name__)

DDG_HTML = "https://html.duckduckgo.com/html/"
BRAVE_HTML = "https://search.brave.com/search"
BRAVE_SEARCH = "https://api.search.brave.com/res/v1/web/search"
MAX_RESULTS = 10
DEFAULT_RESULTS = 8
USER_AGENT = f"Orbweaver/{__version__}"


class WebSearchError(Exception):
    """Search provider failed or returned nothing usable.

    ``fallback`` means another provider should be tried. ``broken`` means the
    page was a bot wall or an unparseable document, not a transient HTTP blip
    and not a genuine empty results page. ``run_websearch`` logs those so
    self-heal intake (which only sees ``log.exception``) can see the outage.
    """

    def __init__(self, message: str, *, fallback: bool = False, broken: bool = False) -> None:
        super().__init__(message)
        self.fallback = fallback
        self.broken = broken


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


def _classes(attrs: dict[str, str]) -> list[str]:
    return attrs.get("class", "").split()


class _BraveHTML(HTMLParser):
    """Brave's public SERP. Result cards are ``div[data-type=web]``."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.hits: list[dict[str, str]] = []
        self._in_result = False
        self._depth = 0
        self._href = ""
        self._in_title = False
        self._title: list[str] = []
        self._in_snip = False
        self._snip_depth = 0
        self._snip: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        d = {k: (v or "") for k, v in attrs}
        classes = _classes(d)
        if tag == "div" and d.get("data-type") == "web" and not self._in_result:
            self._in_result = True
            self._depth = 1
            self._href = ""
            self._in_title = False
            self._title = []
            self._in_snip = False
            self._snip_depth = 0
            self._snip = []
            return
        if not self._in_result:
            return
        if tag == "div":
            self._depth += 1
            if self._in_snip:
                self._snip_depth += 1
        if tag == "a" and not self._href:
            href = d.get("href", "").strip()
            if href.startswith("http") and "search.brave.com" not in href:
                self._href = href
        if tag == "div" and "title" in classes and "search-snippet-title" in classes:
            titled = d.get("title", "").strip()
            if titled:
                self._title = [titled]
            else:
                self._in_title = True
                self._title = []
        if tag == "div" and "generic-snippet" in classes and not self._snip:
            self._in_snip = True
            self._snip_depth = 1

    def handle_endtag(self, tag: str) -> None:
        if not self._in_result:
            return
        if self._in_title and tag == "div":
            self._in_title = False
        if self._in_snip and tag == "div":
            self._snip_depth -= 1
            if self._snip_depth <= 0:
                self._in_snip = False
        if tag == "div":
            self._depth -= 1
            if self._depth <= 0:
                self._finish()

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self._title.append(data)
        elif self._in_snip:
            self._snip.append(data)

    def _finish(self) -> None:
        title = re.sub(r"\s+", " ", unescape("".join(self._title))).strip()
        snippet = re.sub(r"\s+", " ", unescape("".join(self._snip))).strip()
        if title and self._href:
            self.hits.append({"title": title, "url": self._href, "snippet": snippet})
        self._in_result = False


def parse_brave_html(body: str) -> list[dict[str, str]]:
    parser = _BraveHTML()
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


def _status_is_blip(status: int) -> bool:
    """Rate limits and 5xx are outages, not a parser that needs a code change."""
    return status >= 500 or status == 429


def _ddg_challenged(body: str) -> bool:
    low = body.lower()
    return "challenge-form" in low or "anomaly.js" in low


def classify_ddg(status: int, body: str) -> str:
    """``ok`` | ``fallback-blip`` | ``fallback-broken``.

    A real results document contains ``result__a`` or the ``#links`` container
    (including a genuine empty SERP). The bot wall is HTTP 202 plus
    ``anomaly.js`` and has neither, which used to parse as zero hits.
    """
    low = body.lower()
    if "result__a" in body or 'id="links"' in low:
        return "ok"
    if _status_is_blip(status):
        return "fallback-blip"
    if status != 200 or _ddg_challenged(body):
        return "fallback-broken"
    if "no results" in low:
        return "ok"
    return "fallback-broken"


def _transport_error(exc: Exception) -> WebSearchError | None:
    mod = type(exc).__module__
    if mod.startswith(("httpx", "httpcore")):
        return WebSearchError(f"{type(exc).__name__}: {exc}", fallback=True, broken=False)
    return None


def search_duckduckgo(query: str, limit: int) -> list[dict[str, str]]:
    import httpx

    try:
        r = httpx.get(
            DDG_HTML,
            params={"q": query},
            headers=_headers(),
            timeout=20.0,
            follow_redirects=True,
        )
    except Exception as e:
        wrapped = _transport_error(e)
        if wrapped is not None:
            raise wrapped from e
        raise
    kind = classify_ddg(r.status_code, r.text)
    if kind != "ok":
        raise WebSearchError(
            f"DuckDuckGo returned an unusable page (HTTP {r.status_code}).",
            fallback=True,
            broken=kind == "fallback-broken",
        )
    hits = parse_ddg_html(r.text)
    return hits[:limit]


def search_brave(query: str, limit: int) -> list[dict[str, str]]:
    import httpx

    key = settings.brave_api_key
    if not key:
        raise WebSearchError(
            "Brave Search is selected but ORBWEAVER_BRAVE_API_KEY is empty.",
            fallback=True,
            broken=False,
        )
    try:
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
    except Exception as e:
        wrapped = _transport_error(e)
        if wrapped is not None:
            raise wrapped from e
        raise
    if r.status_code != 200:
        raise WebSearchError(
            f"Brave Search HTTP {r.status_code}: {r.text[:200]}",
            fallback=True,
            broken=not _status_is_blip(r.status_code),
        )
    try:
        payload = r.json()
    except ValueError as e:
        raise WebSearchError(
            f"Brave Search returned non-JSON: {e}", fallback=True, broken=True
        ) from e
    return parse_brave_json(payload)[:limit]


def search_brave_html(query: str, limit: int) -> list[dict[str, str]]:
    """Public Brave SERP. No API key. Used when DuckDuckGo is bot-walled."""
    import httpx

    try:
        r = httpx.get(
            BRAVE_HTML,
            params={"q": query},
            headers=_headers(),
            timeout=20.0,
            follow_redirects=True,
        )
    except Exception as e:
        wrapped = _transport_error(e)
        if wrapped is not None:
            raise wrapped from e
        raise
    if r.status_code != 200:
        raise WebSearchError(
            f"Brave results page HTTP {r.status_code}.",
            fallback=True,
            broken=not _status_is_blip(r.status_code),
        )
    hits = parse_brave_html(r.text)
    if not hits:
        raise WebSearchError(
            "Brave results page had no web hits.",
            fallback=True,
            broken=True,
        )
    return hits[:limit]


def _first_provider(
    attempts: list[tuple[str, Any]],
) -> tuple[list[dict[str, str]], str]:
    errors: list[WebSearchError] = []
    for label, fn in attempts:
        try:
            hits = fn()
        except WebSearchError as e:
            if not e.fallback:
                raise
            errors.append(e)
            continue
        return hits, label
    broken = any(e.broken for e in errors)
    detail = "; ".join(str(e) for e in errors) or "no search provider available"
    raise WebSearchError(detail, broken=broken)


def search(query: str, *, max_results: int = DEFAULT_RESULTS) -> tuple[list[dict[str, str]], str]:
    limit = clamp_results(max_results)
    provider = settings.search_provider
    if provider == "brave" and settings.brave_api_key:
        attempts: list[tuple[str, Any]] = [
            ("brave", lambda: search_brave(query, limit)),
            ("brave", lambda: search_brave_html(query, limit)),
        ]
    else:
        attempts = [
            ("duckduckgo", lambda: search_duckduckgo(query, limit)),
            ("brave", lambda: search_brave_html(query, limit)),
        ]
    return _first_provider(attempts)


def run_websearch(inp: dict[str, Any]) -> str:
    query = str(inp.get("query") or "").strip()
    if not query:
        return "WebSearch requires a query."
    limit = clamp_results(inp.get("max_results"))
    try:
        hits, provider = search(query, max_results=limit)
    except WebSearchError as e:
        # Self-heal fingerprints orbweaver.* ERROR records with exc_info.
        # "No results" is a normal tool string and never reached intake, which
        # is why a bot wall on every query did not open a repair turn.
        if e.broken:
            log.exception("WebSearch provider page was unusable")
        return f"WebSearch failed: {e}"
    except Exception as e:
        log.exception("WebSearch failed")
        return f"WebSearch failed: {e}"
    return format_hits(query, provider, hits)
