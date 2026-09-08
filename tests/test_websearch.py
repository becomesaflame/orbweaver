from uuid import uuid4

import pytest

from orbweaver.agent import run_tools
from orbweaver.compact.persist import persist_tool_result
from orbweaver.config import settings
from orbweaver.websearch import (
    format_hits,
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


def test_run_websearch_ddg_403_mentions_brave_key(monkeypatch):
    monkeypatch.setattr(settings, "orbweaver_search_provider", "duckduckgo")
    monkeypatch.setattr(settings, "orbweaver_brave_api_key", "")
    monkeypatch.setattr(settings, "brave_search_api_key", "")
    monkeypatch.setattr("httpx.get", lambda *_a, **_k: _Resp(403, "blocked"))
    out = run_websearch({"query": "anything"})
    assert "WebSearch failed" in out
    assert "ORBWEAVER_BRAVE_API_KEY" in out


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
