"""Pull requests an agent opened, recorded on its session.

Agents open PRs with ``gh pr create`` through the host ``gh`` proxy. A
successful create stores ``{repo, number, url}`` on the session so a later
CI failure can find that agent without guessing from titles or branch names.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any
from uuid import UUID

from orbweaver.store import SESSION_STATUS_DELETED, SESSION_TYPE, Entity, get_store

log = logging.getLogger(__name__)

_PR_URL = re.compile(r"https://github\.com/([^/\s]+)/([^/\s]+)/pull/(\d+)")
_MAX_PER_SESSION = 20

_loop: asyncio.AbstractEventLoop | None = None


def bind_loop(loop: asyncio.AbstractEventLoop | None) -> None:
    """Loop the gh-proxy thread uses to write a session. None detaches it."""
    global _loop
    _loop = loop


def parse_pr_url(text: str) -> dict[str, Any] | None:
    """First ``https://github.com/<owner>/<repo>/pull/<n>`` in ``text``."""
    match = _PR_URL.search(text or "")
    if match is None:
        return None
    owner, repo, number = match.group(1), match.group(2), match.group(3)
    return {
        "repo": f"{owner}/{repo}",
        "number": int(number),
        "url": match.group(0),
    }


def is_pr_create(argv: list[str]) -> bool:
    """True for ``gh pr create`` (flags anywhere after the subcommand)."""
    words = [a for a in argv if not str(a).startswith("-")]
    return len(words) >= 2 and words[0] == "pr" and words[1] == "create"


def note_created_pull(session_id: str, stdout: str) -> None:
    """Schedule a store write from the gh-proxy thread. No-op without a loop or URL."""
    parsed = parse_pr_url(stdout)
    if parsed is None or not str(session_id or "").strip():
        return
    loop = _loop
    if loop is None or not loop.is_running():
        log.info("pull create %s not recorded (no gateway loop)", parsed["url"])
        return
    asyncio.run_coroutine_threadsafe(record_pull(session_id, parsed), loop)


async def record_pull(session_id: str, pull: dict[str, Any]) -> bool:
    """Append ``pull`` to the session. False when the session is missing."""
    try:
        uid = UUID(str(session_id))
    except ValueError:
        return False
    store = get_store()
    ent = await store.get_entity(uid)
    if ent is None or ent.at_type != SESSION_TYPE:
        return False
    if str((ent.jsonld or {}).get("status") or "") == SESSION_STATUS_DELETED:
        return False
    return await attach_pull(ent, pull)


async def attach_pull(ent: Entity, pull: dict[str, Any]) -> bool:
    """Append ``pull`` unless this session already lists that repo and number."""
    repo = str(pull.get("repo") or "")
    try:
        number = int(pull["number"])
    except (KeyError, TypeError, ValueError):
        return False
    url = str(pull.get("url") or f"https://github.com/{repo}/pull/{number}")
    if not repo or number < 1:
        return False
    current = list((ent.jsonld or {}).get("pull_requests") or [])
    if any(_same(item, repo, number) for item in current):
        return False
    current.append({"repo": repo, "number": number, "url": url})
    ent.jsonld["pull_requests"] = current[-_MAX_PER_SESSION:]
    await get_store().put_entity(ent)
    log.info("recorded %s on session %s", url, ent.id)
    return True


async def session_for_pull(repo: str, number: int) -> Entity | None:
    """Newest non-deleted session that lists this pull, or None."""
    want_repo = str(repo or "").strip()
    found: Entity | None = None
    found_at = ""
    for ent in await get_store().list_entities(SESSION_TYPE):
        jsonld = ent.jsonld or {}
        if str(jsonld.get("status") or "") == SESSION_STATUS_DELETED:
            continue
        if not any(_same(item, want_repo, number) for item in jsonld.get("pull_requests") or []):
            continue
        created = str(jsonld.get("created_at") or "")
        if found is None or created >= found_at:
            found, found_at = ent, created
    return found


def _same(item: Any, repo: str, number: int) -> bool:
    if not isinstance(item, dict):
        return False
    try:
        got = int(item["number"])
    except (KeyError, TypeError, ValueError):
        return False
    return str(item.get("repo") or "") == repo and got == number
