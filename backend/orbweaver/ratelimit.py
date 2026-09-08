"""Shared rate limiter: Postgres when ORBWEAVER_STORE=postgres, else a file.

Preserves the historical 120 hits / 60-second sliding window. Keys are hashed
so JWTs are not written to disk or to the hits table.
"""

from __future__ import annotations

import asyncio
import fcntl
import hashlib
import json
import os
from pathlib import Path
from time import time
from typing import Any, Protocol

from orbweaver.config import settings

WINDOW_SECONDS = 60.0
MAX_HITS = 120


class RateLimiter(Protocol):
    async def hit(self, key: str, now: float | None = None) -> bool:
        """Record one hit. Return True if the request is allowed."""


def _bucket(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def rate_limit_data_dir() -> Path:
    raw = (settings.orbweaver_data_dir or "").strip()
    if raw:
        return Path(raw).expanduser()
    return Path.home() / ".orbweaver"


def _hit_file(path: Path, now: float, window: float, max_hits: int) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    with os.fdopen(fd, "r+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        raw = handle.read()
        hits: list[float] = []
        if raw.strip():
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, list):
                    hits = [float(x) for x in parsed]
            except (TypeError, ValueError, json.JSONDecodeError):
                hits = []
        hits = [stamp for stamp in hits if now - stamp < window]
        allowed = len(hits) < max_hits
        if allowed:
            hits.append(now)
        handle.seek(0)
        handle.truncate()
        json.dump(hits, handle)
        handle.flush()
        os.fsync(handle.fileno())
        return allowed


class FileRateLimiter:
    """Sliding-window counter persisted under ``<data_dir>/rate-limit/``."""

    def __init__(
        self,
        data_dir: Path | str,
        *,
        window_seconds: float | None = None,
        max_hits: int | None = None,
    ) -> None:
        self._dir = Path(data_dir) / "rate-limit"
        self._dir.mkdir(parents=True, exist_ok=True)
        self.window_seconds = WINDOW_SECONDS if window_seconds is None else window_seconds
        self.max_hits = MAX_HITS if max_hits is None else max_hits

    def _path_for(self, key: str) -> Path:
        return self._dir / f"{_bucket(key)}.json"

    async def hit(self, key: str, now: float | None = None) -> bool:
        ts = time() if now is None else now
        return await asyncio.to_thread(
            _hit_file,
            self._path_for(key),
            ts,
            self.window_seconds,
            self.max_hits,
        )


class PostgresRateLimiter:
    def __init__(self, store: Any) -> None:
        self._store = store

    async def hit(self, key: str, now: float | None = None) -> bool:
        ts = time() if now is None else now
        return await self._store.rate_limit_hit(_bucket(key), ts, WINDOW_SECONDS, MAX_HITS)


_LIMITER: RateLimiter | None = None


def get_rate_limiter() -> RateLimiter:
    global _LIMITER
    if _LIMITER is None:
        _LIMITER = _build_limiter()
    return _LIMITER


def _build_limiter() -> RateLimiter:
    if settings.orbweaver_store == "postgres":
        from orbweaver.store import get_store

        store = get_store()
        if callable(getattr(store, "rate_limit_hit", None)):
            return PostgresRateLimiter(store)
    return FileRateLimiter(rate_limit_data_dir())


def reset_rate_limiter_for_tests(data_dir: Path | str | None = None) -> FileRateLimiter:
    global _LIMITER
    limiter = FileRateLimiter(data_dir or rate_limit_data_dir())
    _LIMITER = limiter
    return limiter
