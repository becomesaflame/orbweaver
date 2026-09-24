"""CI failure routing and the pull-request association gh pr create records."""

from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest
from httpx import ASGITransport, AsyncClient

from orbweaver.app import app
from orbweaver.ci_operator import failure_prompt, handle_ci_failure
from orbweaver.config import settings
from orbweaver.pulls import attach_pull, is_pr_create, parse_pr_url, record_pull, session_for_pull
from orbweaver.store import SESSION_TYPE, Entity, new_uuid, reset_store_for_tests, session_at_id


def _session(title: str = "desk") -> Entity:
    uid = new_uuid()
    return Entity(
        id=uid,
        at_id=session_at_id(uid),
        at_type=SESSION_TYPE,
        jsonld={
            "@type": SESSION_TYPE,
            "title": title,
            "status": "active",
            "channel": "web",
            "workspace_uri": "workspace:orbweaver",
            "created_at": datetime.now(UTC).isoformat(),
        },
    )


def test_parse_pr_url_and_create_argv():
    got = parse_pr_url("https://github.com/becomesaflame/orbweaver/pull/199\n")
    assert got == {
        "repo": "becomesaflame/orbweaver",
        "number": 199,
        "url": "https://github.com/becomesaflame/orbweaver/pull/199",
    }
    assert parse_pr_url("nothing") is None
    assert is_pr_create(["pr", "create", "--fill"])
    assert not is_pr_create(["pr", "view", "1"])
    assert not is_pr_create(["--help"])


@pytest.mark.asyncio
async def test_record_pull_is_found_and_not_duplicated():
    store = reset_store_for_tests()
    ent = _session()
    await store.put_entity(ent)
    pull = {
        "repo": "becomesaflame/orbweaver",
        "number": 12,
        "url": "https://github.com/becomesaflame/orbweaver/pull/12",
    }
    assert await record_pull(str(ent.id), pull) is True
    assert await record_pull(str(ent.id), pull) is False
    found = await session_for_pull("becomesaflame/orbweaver", 12)
    assert found is not None and found.id == ent.id
    assert await session_for_pull("becomesaflame/orbweaver", 99) is None
    assert await record_pull(str(uuid4()), pull) is False


@pytest.mark.asyncio
async def test_ci_failure_prompts_the_session_that_opened_the_pr():
    store = reset_store_for_tests()
    ent = _session("owner")
    await store.put_entity(ent)
    await attach_pull(
        ent,
        {
            "repo": "becomesaflame/orbweaver",
            "number": 41,
            "url": "https://github.com/becomesaflame/orbweaver/pull/41",
        },
    )
    body = {
        "repo": "becomesaflame/orbweaver",
        "number": 41,
        "url": "https://github.com/becomesaflame/orbweaver/pull/41",
        "head_ref": "feat/ci-failure-operator",
        "head_sha": "abc123def456",
        "run_url": "https://github.com/becomesaflame/orbweaver/actions/runs/1",
        "failed_jobs": ["test"],
    }
    first = await handle_ci_failure(body)
    assert first["status"] == "prompted"
    assert first["session_id"] == str(ent.id)
    again = await handle_ci_failure(body)
    assert again["status"] == "already_queued"
    jobs = await store.due_jobs(datetime.now(UTC))
    assert len(jobs) == 1
    assert jobs[0].session_id == ent.id
    assert "Do not open a second pull request" in jobs[0].payload["message"]
    assert "feat/ci-failure-operator" in failure_prompt(body, created=False)


@pytest.mark.asyncio
async def test_ci_failure_creates_a_session_when_none_owns_the_pr(monkeypatch):
    store = reset_store_for_tests()
    monkeypatch.setattr(settings, "orbweaver_ci_workspace", "workspace:orbweaver")
    result = await handle_ci_failure(
        {"repo": "becomesaflame/orbweaver", "number": 7, "head_ref": "fix/x"}
    )
    assert result["status"] == "created"
    ent = await store.get_entity(UUID(result["session_id"]))
    assert ent is not None
    assert ent.jsonld["workspace_uri"] == "workspace:orbweaver"
    assert ent.jsonld["pull_requests"][0]["number"] == 7
    found = await session_for_pull("becomesaflame/orbweaver", 7)
    assert found is not None and found.id == ent.id


@pytest.mark.asyncio
async def test_ci_webhook_requires_the_shared_secret(monkeypatch):
    reset_store_for_tests()
    monkeypatch.setattr(settings, "orbweaver_ci_webhook_secret", "")
    payload = {"repo": "becomesaflame/orbweaver", "number": 1}
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        closed = await client.post("/v1/webhooks/ci-failure", json=payload)
        assert closed.status_code == 503
        monkeypatch.setattr(settings, "orbweaver_ci_webhook_secret", "s3cret")
        denied = await client.post(
            "/v1/webhooks/ci-failure",
            json=payload,
            headers={"authorization": "Bearer no"},
        )
        assert denied.status_code == 401
        ok = await client.post(
            "/v1/webhooks/ci-failure",
            json=payload,
            headers={"authorization": "Bearer s3cret"},
        )
        assert ok.status_code == 200, ok.text
        assert ok.json()["status"] == "created"
