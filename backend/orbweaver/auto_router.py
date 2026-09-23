"""Lightweight LLM that picks a concrete model for the Auto router."""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any

from orbweaver.config import settings
from orbweaver.open_models import OPEN_MODELS

log = logging.getLogger(__name__)

PROMPT_CHARS = 2000
ROUTER_MAX_TOKENS = 32
ROUTER_TIMEOUT_S = 4.0

ROUTER_SYSTEM = """You route one coding-agent user prompt to a single model.
Reply with exactly one model id from the catalog. No punctuation, no explanation, no markdown.

Prefer a cheaper/faster model for short questions, lookups, greetings, and trivial edits.
Prefer a strong coding model for multi-file changes, debugging, refactors, and tool-heavy work.
Prefer a reasoning model for hard algorithms, proofs, or ambiguous design.
Prefer a large-context model when the prompt is a huge paste.
If unsure, pick the first catalog entry (the default coding model)."""

CLAUDE_BLURBS: dict[str, str] = {
    "claude-haiku-4-5": "fastest Claude; short Q&A and cheap classification",
    "claude-sonnet-4-6": "default coding agent; tool use, edits, most tasks",
    "claude-sonnet-5": "stronger coding and longer reasoning than 4.6",
    "claude-opus-5": "hardest problems; deep reasoning; expensive",
    "claude-fable-5-1": "Claude Fable",
}

_ID_TOKEN = re.compile(r"[A-Za-z0-9._:-]+")


def _blurb(mid: str) -> str:
    if mid in CLAUDE_BLURBS:
        return CLAUDE_BLURBS[mid]
    spec = OPEN_MODELS.get(mid)
    if spec is not None and spec.description:
        return spec.description
    from orbweaver.model_routing import model_label

    return model_label(mid)


def catalog_text(candidates: list[str]) -> str:
    lines = ["Catalog (first entry is the default if you are unsure):"]
    for mid in candidates:
        lines.append(f"- {mid} — {_blurb(mid)}")
    return "\n".join(lines)


def parse_router_choice(text: str, candidates: list[str]) -> str | None:
    """Map a router reply to a unique catalog id, or None if it is not usable."""
    from orbweaver.model_routing import resolve_model_pick

    raw = (text or "").strip()
    if not raw:
        return None
    line = raw.splitlines()[0].strip().strip("`\"'")
    lowered = {mid.lower(): mid for mid in candidates}
    if line.lower() in lowered:
        return lowered[line.lower()]
    contained = [mid for mid in candidates if mid.lower() in line.lower()]
    if contained:
        return max(contained, key=len)
    chosen, _matches = resolve_model_pick(line, candidates)
    if chosen:
        return chosen
    token = _ID_TOKEN.search(line)
    if token is None:
        return None
    needle = token.group(0)
    if needle.lower() in lowered:
        return lowered[needle.lower()]
    chosen, _matches = resolve_model_pick(needle, candidates)
    return chosen or None


def _router_model_id() -> str:
    """Cheap hosted model that classifies the prompt; empty means skip the LLM."""
    from orbweaver.llm import hosted_provider
    from orbweaver.model_routing import is_auto_model

    configured = settings.orbweaver_router_model.strip()
    if not configured or is_auto_model(configured):
        return ""
    if hosted_provider(configured) != "none":
        return configured
    for cheap in ("claude-haiku-4-5", "glm-5.3-flash", "qwen3.6-35b"):
        if hosted_provider(cheap) != "none":
            return cheap
    return ""


def _user_payload(prompt: str, candidates: list[str], *, extra: str = "") -> str:
    body = (prompt or "").strip()
    if len(body) > PROMPT_CHARS:
        body = body[:PROMPT_CHARS] + "\n…[truncated]"
    if not body:
        body = "(empty prompt)"
    bits = [catalog_text(candidates)]
    if extra.strip():
        bits.append(extra.strip())
    bits.append("User prompt:\n" + body)
    return "\n\n".join(bits)


def _resp_text(resp: Any) -> str:
    parts = [
        str(getattr(b, "text", "") or "")
        for b in (getattr(resp, "content", None) or [])
        if getattr(b, "type", None) == "text"
    ]
    return "".join(parts)


async def route_auto_model(
    prompt: str,
    *,
    skip: set[str] | None = None,
    extra: str = "",
    client: Any | None = None,
) -> str:
    """Pick a live model for Auto: LLM choice, else the ranked default.

    Fail open: any router error, timeout, or unparseable reply uses the ranked
    fallback so a blip on Haiku cannot stall the turn.
    """
    from orbweaver.llm import make_hosted_client
    from orbweaver.model_routing import pick_auto_model, routed_models
    from orbweaver.spend import charged_create

    candidates = routed_models(skip=skip)
    ranked = pick_auto_model(skip=skip)
    if not candidates:
        return ""
    if len(candidates) == 1:
        return candidates[0]

    router_id = _router_model_id()
    if not router_id:
        return ranked

    hosted = client
    if hosted is None:
        hosted = make_hosted_client(router_id)
    if hosted is None:
        return ranked

    payload = _user_payload(prompt, candidates, extra=extra)
    try:
        resp = await asyncio.wait_for(
            charged_create(
                hosted,
                model=router_id,
                max_tokens=ROUTER_MAX_TOKENS,
                system=ROUTER_SYSTEM,
                messages=[{"role": "user", "content": payload}],
            ),
            timeout=ROUTER_TIMEOUT_S,
        )
    except Exception as e:
        log.warning("auto router failed; using %s: %s", ranked, e)
        return ranked

    chosen = parse_router_choice(_resp_text(resp), candidates)
    if not chosen:
        log.info("auto router reply was not a catalog id; using %s", ranked)
        return ranked
    if chosen != ranked:
        log.info("auto router chose %s (ranked default %s)", chosen, ranked)
    return chosen
