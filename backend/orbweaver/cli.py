from __future__ import annotations

import argparse


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
    args = parser.parse_args()
    if args.cmd == "snapshot":
        raise SystemExit("snapshot export/import ships in phase 6")
    if args.cmd == "mint":
        from orbweaver.auth import mint_token

        print(mint_token(args.sub, hours=args.hours))
        return
    import uvicorn

    from orbweaver.config import settings

    uvicorn.run(
        "orbweaver.app:app",
        host=settings.orbweaver_host,
        port=settings.orbweaver_port,
        reload=False,
    )


if __name__ == "__main__":
    main()
