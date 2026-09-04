"""JWT issuance and Bearer extraction."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from fastapi import HTTPException, Request
from jose import JWTError, jwt

from orbweaver.config import settings

ALGO = "HS256"


def mint_token(sub: str, extra: dict | None = None, hours: int = 24 * 14) -> str:
    payload = {
        "sub": sub,
        "exp": datetime.now(UTC) + timedelta(hours=hours),
        **(extra or {}),
    }
    return jwt.encode(payload, settings.orbweaver_jwt_secret, algorithm=ALGO)


def decode_token(token: str) -> dict:
    try:
        return jwt.decode(token, settings.orbweaver_jwt_secret, algorithms=[ALGO])
    except JWTError as e:
        raise HTTPException(status_code=401, detail="invalid token") from e


def require_user(request: Request) -> dict:
    header = request.headers.get("authorization") or ""
    if header.lower().startswith("bearer "):
        return decode_token(header.split(" ", 1)[1].strip())
    q = request.query_params.get("token")
    if q:
        return decode_token(q)
    raise HTTPException(status_code=401, detail="missing bearer token")
