import asyncio
import logging
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from orbweaver.agent import run_tools
from orbweaver.compact.persist import persist_tool_result
from orbweaver.config import settings
from orbweaver.websearch import (
    classify_ddg,
    format_hits,
    parse_brave_html,
    parse_brave_json,
    parse_ddg_html,
    run_websearch,
    unwrap_ddg_url,
)
from orbweaver.workspace import LocalWorkspace

DDG_HTML = """
<html><body>
<div class="result">
  <h2 class="result__title">
    <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fdocs.cursor.com%2Fagent%2Foverview">
      Cursor agent tool limits
    </a>
  </h2>
  <a class="result__snippet" href="#">Cursor agent loops stop after the model ends the turn.</a>
</div>
<div class="result">
  <h2 class="result__title">
    <a class="result__a" href="https://docs.aws.amazon.com/bedrock/latest/userguide/agents-max-iterations.html">
      Amazon Bedrock agents max iterations
    </a>
  </h2>
  <div class="result__snippet">Default maximum is 5 model invocations per trace.</div>
</div>
</body></html>
"""

# DuckDuckGo answers datacenter IPs with HTTP 202 and this bot wall. It has no
# result__a links, so the old parser reported "No results" for every query.
DDG_CHALLENGE = """
<html><body>
<form id="challenge-form" action="//duckduckgo.com/anomaly.js?q=bubblewrap"></form>
<p>Unfortunately, bots use DuckDuckGo too.</p>
</body></html>
"""

BRAVE_HTML = """
<div class="snippet" data-pos="0" data-type="web">
  <a href="https://github.com/containers/bubblewrap">
    <div class="title search-snippet-title" title="Bubblewrap">Bubblewrap</div>
  </a>
  <div class="generic-snippet"><div class="content">Unprivileged sandbox for Linux.</div></div>
</div>
<div class="snippet" data-pos="1" data-type="web">
  <a href="https://wiki.archlinux.org/title/Bubblewrap">
    <div class="title search-snippet-title" title="ArchWiki Bubblewrap">ArchWiki Bubblewrap</div>
  </a>
  <div class="generic-snippet"><div>Arch wiki page.</div></div>
</div>
"""


def test_unwrap_ddg_redirect():
    href = "//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fpage&rut=abc"
    assert unwrap_ddg_url(href) == "https://example.com/page"
    assert unwrap_ddg_url("https://example.com/direct") == "https://example.com/direct"


def test_parse_ddg_html_results():
    hits = parse_ddg_html(DDG_HTML)
    assert len(hits) == 2
    assert hits[0]["title"] == "Cursor agent tool limits"
    assert hits[0]["url"] == "https://docs.cursor.com/agent/overview"
    assert "ends the turn" in hits[0]["snippet"]
    assert hits[1]["url"].endswith("agents-max-iterations.html")
    assert "5 model invocations" in hits[1]["snippet"]


def test_parse_brave_html_results():
    hits = parse_brave_html(BRAVE_HTML)
    assert len(hits) == 2
    assert hits[0]["title"] == "Bubblewrap"
    assert hits[0]["url"] == "https://github.com/containers/bubblewrap"
    assert "Unprivileged sandbox" in hits[0]["snippet"]
    assert hits[1]["url"].endswith("/Bubblewrap")


def test_classify_ddg_botwall_is_broken():
    assert classify_ddg(202, DDG_CHALLENGE) == "fallback-broken"
    assert classify_ddg(403, "blocked") == "fallback-broken"
    assert classify_ddg(200, DDG_HTML) == "ok"
    assert classify_ddg(200, '<div id="links"></div>No results.') == "ok"
    assert classify_ddg(200, "<html>not a serp</html>") == "fallback-broken"
    assert classify_ddg(503, "upstream") == "fallback-blip"


def test_parse_brave_json_results():
    hits = parse_brave_json(
        {
            "web": {
                "results": [
                    {
                        "title": "OpenAI Agents SDK",
                        "url": "https://openai.github.io/openai-agents-python/",
                        "description": "max_turns defaults to 10.",
                    }
                ]
            }
        }
    )
    assert hits == [
        {
            "title": "OpenAI Agents SDK",
            "url": "https://openai.github.io/openai-agents-python/",
            "snippet": "max_turns defaults to 10.",
        }
    ]


def test_format_hits_points_at_webfetch():
    text = format_hits(
        "agent max turns",
        "duckduckgo",
        [{"title": "Docs", "url": "https://example.com/a", "snippet": "use 50 steps"}],
    )
    assert "WebSearch duckduckgo: agent max turns (1 results)" in text
    assert "https://example.com/a" in text
    assert "WebFetch" in text
    assert "Do not guess" in text


class _Resp:
    def __init__(self, status, text="", payload=None):
        self.status_code = status
        self.text = text
        self._payload = payload

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


def test_run_websearch_duckduckgo(monkeypatch):
    monkeypatch.setattr(settings, "orbweaver_search_provider", "duckduckgo")
    monkeypatch.setattr(settings, "orbweaver_brave_api_key", "")
    monkeypatch.setattr(settings, "brave_search_api_key", "")

    def fake_get(url, **kwargs):
        assert "duckduckgo.com" in url
        assert kwargs["params"]["q"] == "cline maxRequests"
        return _Resp(200, DDG_HTML)

    monkeypatch.setattr("httpx.get", fake_get)
    out = run_websearch({"query": "cline maxRequests", "max_results": 1})
    assert "Cursor agent tool limits" in out
    assert "https://docs.cursor.com/agent/overview" in out
    assert "Amazon Bedrock" not in out


def test_run_websearch_brave(monkeypatch):
    monkeypatch.setattr(settings, "orbweaver_search_provider", "brave")
    monkeypatch.setattr(settings, "orbweaver_brave_api_key", "brave-test-key")

    def fake_get(url, **kwargs):
        assert url.endswith("/web/search")
        assert kwargs["headers"]["X-Subscription-Token"] == "brave-test-key"
        return _Resp(
            200,
            payload={
                "web": {
                    "results": [
                        {
                            "title": "LangGraph recursion_limit",
                            "url": "https://langchain-ai.github.io/langgraph/",
                            "description": "Default recursion_limit is 25.",
                        }
                    ]
                }
            },
        )

    monkeypatch.setattr("httpx.get", fake_get)
    out = run_websearch({"query": "langgraph recursion_limit"})
    assert "LangGraph recursion_limit" in out
    assert "recursion_limit is 25" in out


def test_ddg_botwall_falls_back_to_brave_html(monkeypatch, caplog):
    monkeypatch.setattr(settings, "orbweaver_search_provider", "duckduckgo")
    monkeypatch.setattr(settings, "orbweaver_brave_api_key", "")
    monkeypatch.setattr(settings, "brave_search_api_key", "")

    def fake_get(url, **kwargs):
        if "duckduckgo.com" in url:
            return _Resp(202, DDG_CHALLENGE)
        if "search.brave.com" in url:
            assert kwargs["params"]["q"] == "bubblewrap sandbox linux"
            return _Resp(200, BRAVE_HTML)
        raise AssertionError(url)

    monkeypatch.setattr("httpx.get", fake_get)
    with caplog.at_level(logging.ERROR, logger="orbweaver.websearch"):
        out = run_websearch({"query": "bubblewrap sandbox linux", "max_results": 1})
    assert "No results" not in out
    assert "WebSearch failed" not in out
    assert "https://github.com/containers/bubblewrap" in out
    assert "wiki.archlinux.org" not in out
    assert "Unprivileged sandbox" in out
    assert "WebSearch brave:" in out
    assert caplog.records == []


def test_genuine_empty_serp_does_not_fallback(monkeypatch):
    monkeypatch.setattr(settings, "orbweaver_search_provider", "duckduckgo")
    monkeypatch.setattr(settings, "orbweaver_brave_api_key", "")
    monkeypatch.setattr(settings, "brave_search_api_key", "")
    calls: list[str] = []

    def fake_get(url, **kwargs):
        calls.append(url)
        return _Resp(200, '<div id="links" class="results">No results.</div>')

    monkeypatch.setattr("httpx.get", fake_get)
    out = run_websearch({"query": "zzzqqqxxxnonexistent"})
    assert "No results" in out
    assert "WebSearch failed" not in out
    assert calls == ["https://html.duckduckgo.com/html/"]


def test_run_websearch_both_providers_unusable_is_failure_not_empty(monkeypatch, caplog):
    monkeypatch.setattr(settings, "orbweaver_search_provider", "duckduckgo")
    monkeypatch.setattr(settings, "orbweaver_brave_api_key", "")
    monkeypatch.setattr(settings, "brave_search_api_key", "")
    monkeypatch.setattr("httpx.get", lambda *_a, **_k: _Resp(202, DDG_CHALLENGE))
    with caplog.at_level(logging.ERROR, logger="orbweaver.websearch"):
        out = run_websearch({"query": "anything"})
    assert "No results" not in out
    assert "WebSearch failed" in out
    assert "unusable page" in out
    assert any(r.exc_info and r.exc_info[0] is not None for r in caplog.records)


@pytest.mark.asyncio
async def test_unusable_websearch_enqueues_selfheal(monkeypatch, tmp_path):
    """The bot wall used to be a successful tool result, so self-heal never ran."""
    from orbweaver.selfheal import attach_log_handler, bind_loop, reset_for_tests
    from orbweaver.store import reset_store_for_tests

    monkeypatch.setattr(settings, "orbweaver_selfheal", True)
    monkeypatch.setattr(settings, "workspace_root", str(tmp_path))
    monkeypatch.setattr(settings, "orbweaver_selfheal_cooldown_s", 86_400.0)
    monkeypatch.setattr(settings, "orbweaver_selfheal_daily_cap", 3)
    monkeypatch.setattr(settings, "orbweaver_selfheal_telegram_chat_id", 0)
    monkeypatch.setattr(settings, "orbweaver_search_provider", "duckduckgo")
    monkeypatch.setattr(settings, "orbweaver_brave_api_key", "")
    monkeypatch.setattr(settings, "brave_search_api_key", "")
    monkeypatch.setattr("httpx.get", lambda *_a, **_k: _Resp(202, DDG_CHALLENGE))
    reset_for_tests()
    store = reset_store_for_tests()
    bind_loop(asyncio.get_running_loop())
    attach_log_handler()
    try:
        out = run_websearch({"query": "bubblewrap sandbox linux"})
        assert "No results" not in out
        deadline = asyncio.get_running_loop().time() + 2
        jobs: list = []
        while asyncio.get_running_loop().time() < deadline:
            jobs = await store.due_jobs(datetime.now(UTC) + timedelta(days=1))
            if jobs:
                break
            await asyncio.sleep(0)
        assert jobs
        assert jobs[0].payload.get("selfheal") is True
        assert "WebSearchError:websearch.py:" in jobs[0].payload.get("fingerprint", "")
    finally:
        reset_for_tests()


def test_empty_query():
    assert "requires a query" in run_websearch({"query": "  "})


@pytest.mark.asyncio
async def test_run_tools_websearch(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "orbweaver_search_provider", "duckduckgo")
    monkeypatch.setattr(settings, "orbweaver_brave_api_key", "")
    monkeypatch.setattr(settings, "brave_search_api_key", "")
    monkeypatch.setattr("httpx.get", lambda *_a, **_k: _Resp(200, DDG_HTML))
    ws = LocalWorkspace("workspace:default", str(tmp_path))
    out = await run_tools(
        "WebSearch",
        {"query": "cursor agent tool limits"},
        {"workspace": ws, "store": None, "session_id": uuid4(), "workspace_kind": "local"},
    )
    assert "docs.cursor.com" in out
    stored, rel = persist_tool_result(ws, "toolu_s", "WebSearch", out)
    assert rel is None
    assert stored == out
