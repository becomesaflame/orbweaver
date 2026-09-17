from __future__ import annotations

import argparse
import logging
import sys
from copy import deepcopy
from typing import Any

LOG_FORMAT = "%(levelname)s %(name)s %(message)s"
HANDLER_NAME = "orbweaver"
_LEVELS = {
    "critical": logging.CRITICAL,
    "error": logging.ERROR,
    "warning": logging.WARNING,
    "warn": logging.WARNING,
    "info": logging.INFO,
    "debug": logging.DEBUG,
}


def parse_log_level(raw: str) -> int:
    key = (raw or "info").strip().lower()
    if key not in _LEVELS:
        raise ValueError(
            f"invalid ORBWEAVER_LOG_LEVEL={raw!r}; "
            "use debug, info, warning, error, or critical"
        )
    return _LEVELS[key]


def gateway_log_config(level: str | None = None) -> dict[str, Any]:
    """Uvicorn's dictConfig plus a root handler so orbweaver.* logs reach stderr.

    Uvicorn's default config sets only ``uvicorn.*`` loggers. INFO on
    ``orbweaver.*`` is dropped, and ERROR goes through ``logging.lastResort``
    as a bare message with no logger name.
    """
    import uvicorn.config as uvicorn_config

    from orbweaver.config import settings

    numeric = parse_log_level(level if level is not None else settings.orbweaver_log_level)
    cfg = deepcopy(uvicorn_config.LOGGING_CONFIG)
    cfg["formatters"][HANDLER_NAME] = {"format": LOG_FORMAT}
    cfg["handlers"][HANDLER_NAME] = {
        "class": "logging.StreamHandler",
        "formatter": HANDLER_NAME,
        "stream": "ext://sys.stderr",
    }
    cfg["root"] = {"level": logging.getLevelName(numeric), "handlers": [HANDLER_NAME]}
    return cfg


def configure_logging(level: str | None = None) -> None:
    """Attach a named stderr handler before uvicorn.run (JWT refusal, warnings)."""
    from orbweaver.config import settings

    numeric = parse_log_level(level if level is not None else settings.orbweaver_log_level)
    root = logging.getLogger()
    root.setLevel(numeric)
    existing = next((h for h in root.handlers if getattr(h, "name", None) == HANDLER_NAME), None)
    if existing is None:
        handler = logging.StreamHandler(sys.stderr)
        handler.name = HANDLER_NAME
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
        root.addHandler(handler)
        existing = handler
    existing.setLevel(numeric)
    logging.getLogger("orbweaver").setLevel(numeric)


def main() -> None:
    parser = argparse.ArgumentParser(prog="orbweaver")
    sub = parser.add_subparsers(dest="cmd")
    sub.add_parser("serve", help="run the FastAPI gateway")
    mint = sub.add_parser(
        "mint",
        help="print a JWT (run on the gateway host; paste into the web client or VS Code)",
    )
    mint.add_argument("--sub", default="web", help="token subject (default: web)")
    mint.add_argument("--hours", type=int, default=24 * 14, help="lifetime in hours (default: 336)")
    snap = sub.add_parser("snapshot", help="phase 6: export/import (not implemented yet)")
    snap.add_argument("action", choices=["export", "import"])
    drain = sub.add_parser(
        "drain",
        help="refuse new turns and wait until running ones finish (used by deploy)",
    )
    drain.add_argument("--url", default="", help="gateway base URL (default http://<host>:<port>)")
    drain.add_argument("--timeout", type=float, default=None, help="seconds to wait (default 900)")
    drain.add_argument("--interval", type=float, default=1.0, help="poll interval in seconds")
    args = parser.parse_args()
    if args.cmd == "snapshot":
        raise SystemExit("snapshot export/import ships in phase 6")
    if args.cmd == "mint":
        from orbweaver.auth import mint_token

        print(mint_token(args.sub, hours=args.hours))
        return
    if args.cmd == "drain":
        import httpx

        from orbweaver.config import settings
        from orbweaver.drain import wait_for_idle

        url = (args.url or "").strip() or f"http://{settings.orbweaver_host}:{settings.orbweaver_port}"
        timeout = settings.orbweaver_drain_timeout_s if args.timeout is None else args.timeout
        with httpx.Client(base_url=url.rstrip("/"), timeout=15.0) as client:
            wait_for_idle(client, timeout_s=timeout, interval_s=args.interval)
        return
    import uvicorn

    from orbweaver.auth import InsecureJwtSecretError, check_jwt_secret
    from orbweaver.config import settings

    try:
        configure_logging()
    except ValueError as e:
        raise SystemExit(f"refusing to start: {e}") from e
    try:
        check_jwt_secret(settings)
    except InsecureJwtSecretError as e:
        logging.getLogger("orbweaver").error("refusing to start: %s", e)
        raise SystemExit(1) from e
    from orbweaver.model_routing import unsupported_configured_models

    # A typo here only breaks the channels that use it, so warn instead of
    # refusing to start; turns on that model answer with no_llm_echo.
    for env_name, model_id in unsupported_configured_models():
        logging.getLogger("orbweaver").warning(
            "%s=%s is not a model Orbweaver can route; turns using it will not reach a provider",
            env_name,
            model_id,
        )
    uvicorn.run(
        "orbweaver.app:app",
        host=settings.orbweaver_host,
        port=settings.orbweaver_port,
        reload=False,
        log_config=gateway_log_config(),
        log_level=logging.getLevelName(parse_log_level(settings.orbweaver_log_level)).lower(),
    )


if __name__ == "__main__":
    main()
