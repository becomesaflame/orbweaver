from __future__ import annotations

import argparse
import os


def main() -> None:
    parser = argparse.ArgumentParser(prog="orbweaver")
    sub = parser.add_subparsers(dest="cmd")
    sub.add_parser("serve", help="run the FastAPI gateway")
    snap = sub.add_parser("snapshot", help="phase 6: export/import (not implemented yet)")
    snap.add_argument("action", choices=["export", "import"])
    args = parser.parse_args()
    if args.cmd == "snapshot":
        raise SystemExit("snapshot export/import ships in phase 6")
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
