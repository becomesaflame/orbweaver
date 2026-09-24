"""Route a failed PR check to the agent that opened the pull request.

GitHub Actions posts here after the ``ci`` workflow fails. The session is
the one whose ``pull_requests`` list contains that PR (written when ``gh pr
create`` succeeded). That session gets a one-shot cron turn. When nothing
is associated, a new chat is created on the CI workspace and the same turn
runs there, with the PR recorded on it so the next failure finds it.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import Any

from orbweaver.channels.router import create_candidate_session
from orbweaver.config import settings
from orbweaver.pulls import attach_pull, session_for_pull
from orbweaver.store import Job, get_store, new_uuid

log = logging.getLogger(__name__)


def failure_prompt(body: dict[str, Any], *, created: bool) -> str:
    repo = str(body.get("repo") or "")
    number = int(body.get("number") or 0)
    url = str(body.get("url") or f"https://github.com/{repo}/pull/{number}")
    head = str(body.get("head_ref") or "").strip()
    sha = str(body.get("head_sha") or "").strip()
    run_url = str(body.get("run_url") or "").strip()
    jobs = [str(j) for j in (body.get("failed_jobs") or []) if str(j).strip()]
    where = f"branch `{head}`" if head else "its head branch"
    if sha:
        where += f" at `{sha[:12]}`"
    lines = [
        f"CI failed on {url} ({where}).",
        f"Failed jobs: {', '.join(jobs)}." if jobs else "The ci workflow failed.",
    ]
    if run_url:
        lines.append(f"Run: {run_url}")
    if created:
        lines.append(
            "No existing agent was associated with this pull request. "
            f"Check out `{head or 'the PR head'}` in this workspace, fix the failure, "
            "commit, and push to that branch. Do not open a second pull request."
        )
    else:
        lines.append(
            "You opened this pull request. Fix the failure on its branch, commit, "
            "and push. Do not open a second pull request."
        )
    return "\n".join(lines)


async def handle_ci_failure(body: dict[str, Any]) -> dict[str, Any]:
    """Enqueue a fix turn. Idempotent while a job for this PR is already queued."""
    repo = str(body.get("repo") or "").strip()
    try:
        number = int(body["number"])
    except (KeyError, TypeError, ValueError):
        number = 0
    if "/" not in repo or number < 1:
        raise ValueError("repo (owner/name) and number are required")
    store = get_store()
    now = datetime.now(UTC)
    for job in await store.due_jobs(now + timedelta(days=3650)):
        payload = job.payload or {}
        if payload.get("ci_failure") and payload.get("repo") == repo and int(payload.get("number") or 0) == number:
            return {
                "status": "already_queued",
                "session_id": str(job.session_id) if job.session_id else "",
                "job_id": str(job.id),
            }
    owner = await session_for_pull(repo, number)
    created = owner is None
    if owner is None:
        title = f"CI #{number}"
        owner = await create_candidate_session(
            store,
            title=title,
            workspace_uri=settings.orbweaver_ci_workspace,
            channel="web",
        )
        url = str(body.get("url") or f"https://github.com/{repo}/pull/{number}")
        await attach_pull(owner, {"repo": repo, "number": number, "url": url})
    message = failure_prompt(body, created=created)
    job = Job(
        id=new_uuid(),
        due_at=now,
        payload={
            "message": message,
            "ci_failure": True,
            "repo": repo,
            "number": number,
        },
        session_id=owner.id,
    )
    await store.put_job(job)
    log.info("ci failure %s#%s -> session %s created=%s", repo, number, owner.id, created)
    return {
        "status": "created" if created else "prompted",
        "session_id": str(owner.id),
        "job_id": str(job.id),
    }
