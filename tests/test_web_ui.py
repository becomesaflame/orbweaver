"""Web UI: vendored static assets are served, and tool_call events carry a summary."""

from types import SimpleNamespace
from uuid import uuid4

import anthropic
import pytest
from httpx import ASGITransport, AsyncClient

from orbweaver.agent import _tool_call_summary, agent_turn
from orbweaver.app import WEB_DIR, app
from orbweaver.config import settings
from orbweaver.store import reset_store_for_tests
from orbweaver.workspace import LocalWorkspace

VENDORED = ("marked.min.js", "purify.min.js")


@pytest.fixture(autouse=True)
def _store():
    reset_store_for_tests()


@pytest.mark.asyncio
async def test_vendored_static_files_are_served():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        for name in VENDORED:
            r = await client.get(f"/ui/vendor/{name}")
            assert r.status_code == 200, (name, r.status_code)
            assert "javascript" in r.headers["content-type"], r.headers["content-type"]
            assert len(r.content) > 1000
        readme = await client.get("/ui/vendor/README.md")
        assert readme.status_code == 200
        for name in VENDORED:
            assert name in readme.text


@pytest.mark.asyncio
async def test_index_references_vendored_scripts():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        for path in ("/", "/ui/"):
            r = await client.get(path)
            assert r.status_code == 200, path
            for name in VENDORED:
                assert f"/ui/vendor/{name}" in r.text, (path, name)
            assert "function renderEvent(ev)" in r.text
    assert (WEB_DIR / "vendor" / "README.md").is_file()


class _ToolUse:
    def __init__(self, name, inp, uid="tu-read"):
        self.type = "tool_use"
        self.id = uid
        self.name = name
        self.input = inp


class _FakeAnthropic:
    def __init__(self, responses):
        self._responses = list(responses)
        self.messages = self

    async def create(self, **kwargs):
        if not self._responses:
            return SimpleNamespace(content=[SimpleNamespace(type="text", text="done")])
        return self._responses.pop(0)


@pytest.mark.asyncio
async def test_tool_call_event_carries_summary(tmp_path, monkeypatch):
    (tmp_path / "hello.txt").write_text("hi\n", encoding="utf-8")
    monkeypatch.setattr(settings, "anthropic_api_key", "sk-test")
    client = _FakeAnthropic([SimpleNamespace(content=[_ToolUse("Read", {"path": "hello.txt"})])])
    monkeypatch.setattr(anthropic, "AsyncAnthropic", lambda *a, **k: client)
    store = reset_store_for_tests()
    sid = uuid4()
    events = await agent_turn(store, sid, "read it", LocalWorkspace("workspace:default", str(tmp_path)))
    call = next(e for e in events if e.kind == "tool_call")
    assert call.payload["name"] == "Read"
    assert call.payload["summary"] == "hello.txt"
    result = next(e for e in events if e.kind == "tool_result")
    assert result.payload["tool_use_id"] == call.payload["id"]


def test_tool_call_summary_never_raises():
    assert _tool_call_summary("Bash", {"command": "ls -la"}) == "ls -la"
    assert _tool_call_summary("Read", None) == "Read"
    assert _tool_call_summary("Weird", {"x": object()}) == "Weird"
