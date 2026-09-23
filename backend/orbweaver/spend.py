"""Daily UTC spend caps per LLM provider.

0 / unset = unlimited. Anthropic dollars are tokens × ``pricing`` list rates;
Earth Runtime (provider string ``openrouter``) uses ``usage.cost`` when the
API returns it. The ledger lives in ``instance_meta`` so it survives a restart
and is shared across turns in one gateway process.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any

from orbweaver.config import settings
from orbweaver.pricing import estimate_prompt_usd, prompt_tokens_of, usage_cost_usd
from orbweaver.store import get_store

log = logging.getLogger(__name__)

MICRO = 1_000_000
_lock = asyncio.Lock()

CAPPED_PROVIDERS = ("anthropic", "openrouter")


class SpendCapped(Exception):
    """This provider's daily USD cap would be exceeded by the next call."""

    def __init__(self, provider: str, spent_usd: float, cap_usd: float) -> None:
        self.provider = provider
        self.spent_usd = spent_usd
        self.cap_usd = cap_usd
        super().__init__(
            f"{provider} daily spend ${spent_usd:.2f} exceeds cap ${cap_usd:.2f} UTC"
        )


def cap_usd(provider: str) -> float | None:
    """Configured cap in USD, or None when the provider is unlimited."""
    if provider == "anthropic":
        raw = float(settings.orbweaver_spend_cap_anthropic_usd_day or 0.0)
    elif provider == "openrouter":
        raw = float(settings.orbweaver_spend_cap_openrouter_usd_day or 0.0)
    else:
        return None
    if raw <= 0:
        return None
    return raw


def spend_key(provider: str, day: str | None = None) -> str:
    day = day or datetime.now(UTC).date().isoformat()
    return f"spend:{provider}:{day}"


def _to_micro(usd: float) -> int:
    return round(float(usd) * MICRO)


def _from_micro(raw: int) -> float:
    return raw / MICRO


async def spent_usd(provider: str) -> float:
    store = get_store()
    raw = await store.meta_get(spend_key(provider))
    try:
        return _from_micro(int(raw or 0))
    except (TypeError, ValueError):
        return 0.0


async def _add_usd(provider: str, delta_usd: float) -> float:
    if provider not in CAPPED_PROVIDERS:
        return 0.0
    delta_micro = _to_micro(delta_usd)
    if delta_micro == 0:
        return await spent_usd(provider)
    key = spend_key(provider)
    async with _lock:
        store = get_store()
        raw = await store.meta_get(key)
        try:
            current = int(raw or 0)
        except (TypeError, ValueError):
            current = 0
        nxt = max(0, current + delta_micro)
        await store.meta_set(key, str(nxt))
        return _from_micro(nxt)


async def snapshot() -> dict[str, Any]:
    day = datetime.now(UTC).date().isoformat()
    out: dict[str, Any] = {"utc_date": day, "providers": {}}
    for provider in CAPPED_PROVIDERS:
        cap = cap_usd(provider)
        spent = await spent_usd(provider)
        out["providers"][provider] = {
            "cap_usd": cap or 0.0,
            "spent_usd": round(spent, 6),
            "unlimited": cap is None,
        }
    return out


async def reserve(provider: str, estimate_usd: float) -> None:
    """Add ``estimate_usd`` if it fits; raise ``SpendCapped`` otherwise.

    No-op when the provider has no cap. Callers must ``refund`` or ``settle``.
    """
    cap = cap_usd(provider)
    if cap is None:
        return
    est = max(0.0, float(estimate_usd or 0.0))
    async with _lock:
        store = get_store()
        key = spend_key(provider)
        raw = await store.meta_get(key)
        try:
            current = int(raw or 0)
        except (TypeError, ValueError):
            current = 0
        nxt = current + _to_micro(est)
        spent = _from_micro(current)
        if spent >= cap - 1e-12 or _from_micro(nxt) > cap + 1e-9:
            raise SpendCapped(provider, spent, cap)
        await store.meta_set(key, str(nxt))


async def refund(provider: str, estimate_usd: float) -> None:
    if cap_usd(provider) is None:
        return
    await _add_usd(provider, -max(0.0, float(estimate_usd or 0.0)))


async def settle(provider: str, estimate_usd: float, actual_usd: float) -> None:
    """Replace a reservation with the real cost; record actual when uncapped."""
    cap = cap_usd(provider)
    actual = max(0.0, float(actual_usd or 0.0))
    if cap is None:
        await _add_usd(provider, actual)
        return
    await _add_usd(provider, actual - max(0.0, float(estimate_usd or 0.0)))


async def run_charged(
    model: str,
    kwargs: dict[str, Any],
    call: Callable[[], Awaitable[Any]],
) -> Any:
    """Reserve estimated USD, run ``call``, settle from usage; refund on error."""
    from orbweaver.llm import select_provider

    model = (model or str(kwargs.get("model") or "") or settings.orbweaver_model).strip()
    provider = select_provider(model)
    estimate = 0.0
    if provider == "anthropic":
        estimate = estimate_prompt_usd(model, prompt_tokens_of(kwargs))
    try:
        await reserve(provider, estimate)
    except SpendCapped:
        log.warning(
            "spend cap hit provider=%s model=%s estimate_usd=%.6f",
            provider,
            model,
            estimate,
        )
        raise
    try:
        resp = await call()
    except BaseException:
        await refund(provider, estimate)
        raise
    usage = getattr(resp, "usage", None)
    actual = usage_cost_usd(provider, model, usage)
    await settle(provider, estimate, actual)
    return resp


async def charged_create(client: Any, **kwargs: Any) -> Any:
    async def _go() -> Any:
        return await client.messages.create(**kwargs)

    return await run_charged(str(kwargs.get("model") or ""), kwargs, _go)
