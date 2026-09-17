"""Web UI: vendored static assets are served, and tool_call events carry a summary."""

from types import SimpleNamespace
from uuid import UUID, uuid4

import anthropic
import pytest
from httpx import ASGITransport, AsyncClient

from orbweaver.agent import _tool_call_summary, agent_turn
from orbweaver.app import WEB_DIR, app
from orbweaver.config import settings
from orbweaver.store import reset_store_for_tests
from orbweaver.turns import acquire as acquire_turn
from orbweaver.turns import release as release_turn
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
            # Winamp chrome: LCD title well, no duplicate model header, no player/EQ chrome.
            assert 'id="ctx-meter"' in r.text
            assert 'id="chat-where"' in r.text
            assert "function paintContext(ctx)" in r.text
            assert "function paintConnection(ok)" in r.text
            assert 'id="app-model"' not in r.text
            assert "Agent EQ" not in r.text
            assert ">PREV<" not in r.text
            # Switching chats must rebuild the log even if a turn is in flight
            # on the target session (live permission wait / streaming).
            assert "if (flightFor(sid) && logSid === sid)" in r.text
            assert 'addEventListener("hashchange"' in r.text
            assert "if (inFlight && inFlight.sid === sid) { syncComposer(); return; }" not in r.text
            # A turn running in one chat must not block send() in another.
            assert "if (currentFlight()) return;" in r.text
            assert "const flights = new Map()" in r.text
            assert "if (inFlight) return;" not in r.text
            # Rail activity: a live turn spins, a finished one stays highlighted
            # until clicked. The gateway's `running` flag is what makes turns
            # started elsewhere (Telegram, cron, another tab) animate here.
            assert "function paintChatState(b, sid)" in r.text
            assert "function paintActivity()" in r.text
            assert "function setGatewayRunning(ids)" in r.text
            assert 'classList.toggle("running", running)' in r.text
            assert 'classList.toggle("done", done)' in r.text
            assert "b.onclick = () => { markSeen(s.id); setActive(s); };" in r.text
            assert "function markSeen(sid)" in r.text
            assert "function trackUnseen()" in r.text
            assert "function startActivityPoll()" in r.text
            assert ".chat.running .chat-state .spin" in r.text
            assert "@keyframes chat-spin" in r.text
            assert ".chat.done {" in r.text
            # An answered approval drops its Allow / Deny buttons rather than
            # leaving them on screen (disabled) for a second click.
            assert "function clearPermissionActions(card)" in r.text
            assert 'card.querySelectorAll(".perm-actions").forEach((n) => n.remove())' in r.text
            assert '.perm-actions button").forEach((b) => { b.disabled = true; })' not in r.text
    assert (WEB_DIR / "vendor" / "README.md").is_file()


@pytest.mark.asyncio
async def test_list_sessions_reports_running_turns(tmp_path, monkeypatch, auth_header):
    """The rail's spinner needs /v1/sessions to say which sessions have a live turn.

    Without this the sidebar can only animate turns this browser tab started;
    a turn from Telegram, cron, or a second tab would look idle.
    """
    monkeypatch.setattr(settings, "workspace_root", str(tmp_path))
    monkeypatch.setattr(settings, "anthropic_api_key", "")
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        made = await client.post(
            "/v1/sessions",
            json={"workspace_uri": "workspace:default", "workspace_kind": "local"},
            headers=auth_header,
        )
        assert made.status_code == 200, made.text
        sid = made.json()["id"]

        idle = await client.get("/v1/sessions", headers=auth_header)
        row = next(s for s in idle.json()["sessions"] if s["id"] == sid)
        assert row["running"] is False

        state = acquire_turn(UUID(sid), channel="telegram")
        assert state is not None
        try:
            busy = await client.get("/v1/sessions", headers=auth_header)
            row = next(s for s in busy.json()["sessions"] if s["id"] == sid)
            assert row["running"] is True
        finally:
            release_turn(UUID(sid), state)

        after = await client.get("/v1/sessions", headers=auth_header)
        row = next(s for s in after.json()["sessions"] if s["id"] == sid)
        assert row["running"] is False


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
