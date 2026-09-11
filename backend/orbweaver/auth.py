"""JWT issuance and Bearer extraction."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from fastapi import HTTPException, Request, WebSocket
from jose import JWTError, jwt

from orbweaver.config import Settings, settings

ALGO = "HS256"

INSECURE_DEFAULT_SECRET = "dev-secret-change-me"
MIN_SECRET_BYTES = 32

# WebSocket clients send `Sec-WebSocket-Protocol: bearer, <jwt>`; the gateway
# echoes `bearer` back. The token never touches the URL or access logs.
WS_BEARER_SUBPROTOCOL = "bearer"


class InsecureJwtSecretError(RuntimeError):
    """ORBWEAVER_JWT_SECRET is missing, the public default, or too short."""


def jwt_secret_problem(secret: str) -> str | None:
    """Why `secret` must not be used to sign tokens, or None when it is fine."""
    if not secret:
        return "ORBWEAVER_JWT_SECRET is empty"
    if secret == INSECURE_DEFAULT_SECRET:
        return f"ORBWEAVER_JWT_SECRET is the public default {INSECURE_DEFAULT_SECRET!r}"
    if len(secret.encode("utf-8")) < MIN_SECRET_BYTES:
        return (
            f"ORBWEAVER_JWT_SECRET is {len(secret.encode('utf-8'))} bytes; "
            f"at least {MIN_SECRET_BYTES} are required"
        )
    return None


def check_jwt_secret(cfg: Settings | None = None) -> None:
    """Raise InsecureJwtSecretError unless the secret is safe or dev mode is on.

    Anyone who can read the source can forge a Bearer token for every endpoint
    when the default secret is in use, so the gateway refuses to start.
    """
    cfg = cfg if cfg is not None else settings
    problem = jwt_secret_problem(cfg.orbweaver_jwt_secret)
    if problem is None or cfg.orbweaver_dev_insecure:
        return
    raise InsecureJwtSecretError(
        f"{problem}. Set ORBWEAVER_JWT_SECRET to a random value of at least "
        f"{MIN_SECRET_BYTES} bytes, e.g. "
        "`python3 -c \"import secrets; print(secrets.token_urlsafe(48))\"`, "
        "or set ORBWEAVER_DEV_INSECURE=1 for a local, non-networked dev run."
    )


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


def websocket_subprotocol_token(websocket: WebSocket) -> tuple[bool, str | None]:
    """Parse `Sec-WebSocket-Protocol: bearer, <jwt>`.

    Returns (bearer_offered, token). `bearer_offered` is True when the client
    listed the `bearer` subprotocol at all, so the server can echo it back even
    when the token turns out to be missing or invalid.
    """
    offered: list[str] = []
    for line in websocket.headers.getlist("sec-websocket-protocol"):
        offered.extend(p.strip() for p in line.split(",") if p.strip())
    bearer_offered = False
    token: str | None = None
    for proto in offered:
        if proto.lower() == WS_BEARER_SUBPROTOCOL:
            bearer_offered = True
        elif bearer_offered and token is None:
            token = proto
    return bearer_offered, token
