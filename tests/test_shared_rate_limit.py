import asyncio
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from orbweaver.app import app
from orbweaver.config import settings
from orbweaver.ratelimit import (
    FileRateLimiter,
    PostgresRateLimiter,
    get_rate_limiter,
    reset_rate_limiter_for_tests,
)


def _worker_hits(data_dir: str, key: str, n: int, now: float) -> int:
    async def _run() -> int:
        limiter = FileRateLimiter(Path(data_dir), max_hits=120)
        allowed = 0
        for _ in range(n):
            if await limiter.hit(key, now=now):
                allowed += 1
        return allowed

    return asyncio.run(_run())


@pytest.mark.asyncio
async def test_file_limiter_survives_restart(tmp_path: Path):
    now = 1_700_000_000.0
    first = FileRateLimiter(tmp_path, max_hits=120)
    for i in range(50):
        assert await first.hit("alice", now=now + i * 0.001)

    restarted = FileRateLimiter(tmp_path, max_hits=120)
    for i in range(70):
        assert await restarted.hit("alice", now=now + 1 + i * 0.001)
    assert await restarted.hit("alice", now=now + 2) is False

    later = FileRateLimiter(tmp_path, max_hits=120)
    assert await later.hit("alice", now=now + 2.5) is False
    assert await later.hit("alice", now=now + 61) is True


@pytest.mark.asyncio
async def test_file_limiter_increment_atomicity(tmp_path: Path):
    limiter = FileRateLimiter(tmp_path, max_hits=120)
    now = 1_700_000_100.0
    results = await asyncio.gather(*[limiter.hit("bob", now=now) for _ in range(200)])
    assert sum(1 for ok in results if ok) == 120
    assert sum(1 for ok in results if not ok) == 80

    again = FileRateLimiter(tmp_path, max_hits=120)
    assert await again.hit("bob", now=now + 0.5) is False


def test_file_limiter_cross_process_atomicity(tmp_path: Path):
    now = 1_700_000_200.0
    per_proc = 80
    with ProcessPoolExecutor(max_workers=4) as pool:
        futs = [
            pool.submit(_worker_hits, str(tmp_path), "carol", per_proc, now)
            for _ in range(4)
        ]
        allowed = sum(f.result() for f in futs)
    assert allowed == 120


@pytest.mark.asyncio
async def test_postgres_backend_when_store_is_postgres(tmp_path: Path, monkeypatch):
    rows: list[float] = []

    class FakePg:
        async def rate_limit_hit(self, key: str, now: float, window: float, max_hits: int) -> bool:
            rows[:] = [t for t in rows if now - t < window]
            if len(rows) >= max_hits:
                return False
            rows.append(now)
            return True

    monkeypatch.setattr(settings, "orbweaver_store", "postgres")
    monkeypatch.setattr(settings, "orbweaver_data_dir", str(tmp_path))
    from orbweaver import ratelimit as rl

    monkeypatch.setattr(rl, "_LIMITER", None)
    monkeypatch.setattr(rl, "MAX_HITS", 3)
    monkeypatch.setattr("orbweaver.store.get_store", lambda: FakePg())

    limiter = get_rate_limiter()
    assert isinstance(limiter, PostgresRateLimiter)
    now = 1_700_000_300.0
    assert await limiter.hit("dave", now=now)
    assert await limiter.hit("dave", now=now)
    assert await limiter.hit("dave", now=now)
    assert await limiter.hit("dave", now=now) is False

    monkeypatch.setattr(rl, "_LIMITER", None)
    restarted = get_rate_limiter()
    assert isinstance(restarted, PostgresRateLimiter)
    assert await restarted.hit("dave", now=now) is False


@pytest.mark.asyncio
async def test_http_rate_limit_uses_shared_store(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(settings, "orbweaver_data_dir", str(tmp_path))
    reset_rate_limiter_for_tests(tmp_path)
    limiter = FileRateLimiter(tmp_path, max_hits=3)
    from orbweaver import ratelimit as rl

    monkeypatch.setattr(rl, "_LIMITER", limiter)
    limiter.max_hits = 3

    transport = ASGITransport(app=app, client=("10.0.0.1", 123))
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        for _ in range(3):
            r = await client.get("/v1/sessions")
            assert r.status_code == 401
        blocked = await client.get("/v1/sessions")
        assert blocked.status_code == 429
        health = await client.get("/health")
        assert health.status_code == 200

    # New limiter instance, same data dir: still limited (simulated restart).
    after = FileRateLimiter(tmp_path, max_hits=3)
    assert await after.hit("10.0.0.1") is False
