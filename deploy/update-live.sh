#!/usr/bin/env bash
# Fast-forward the production checkout to origin/main and restart the gateway.
# Intended for a self-hosted runner on the live host, after CI has promoted RC.
set -euo pipefail

LIVE="${ORBWEAVER_LIVE:-/home/orbweaver/orbweaver}"
cd "$LIVE"

git fetch origin main
git checkout main
git merge --ff-only origin/main

if [[ -x backend/.venv/bin/pip ]]; then
  (cd backend && .venv/bin/pip install -e ".[dev]")
fi

if systemctl --user cat orbweaver.service >/dev/null 2>&1; then
  systemctl --user restart orbweaver
else
  echo "orbweaver.service is not installed; skip restart" >&2
fi

curl -fsS --retry 5 --retry-delay 1 http://127.0.0.1:8080/health
