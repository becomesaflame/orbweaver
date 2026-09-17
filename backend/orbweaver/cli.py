from __future__ import annotations

import argparse
import logging


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
        check_jwt_secret(settings)
    except InsecureJwtSecretError as e:
        # No handlers are configured yet, so logging's last-resort handler
        # writes this to stderr; uvicorn would otherwise own logging setup.
        logging.getLogger("orbweaver").error("refusing to start: %s", e)
        raise SystemExit(1) from e
    uvicorn.run(
        "orbweaver.app:app",
        host=settings.orbweaver_host,
        port=settings.orbweaver_port,
        reload=False,
    )


if __name__ == "__main__":
    main()
