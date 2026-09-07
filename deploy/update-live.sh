#!/usr/bin/env bash
# Fast-forward the production checkout to origin/main and restart the gateway.
# Intended for a self-hosted runner on the live host, after a green merge to main.
set -euo pipefail

LIVE="${ORBWEAVER_LIVE:-/home/orbweaver/orbweaver}"
cd "$LIVE"

git fetch origin main
git checkout main
git merge --ff-only origin/main

if [[ -x backend/.venv/bin/pip ]]; then
  (cd backend && .venv/bin/pip install -e ".[dev]")
fi

uid="$(id -u)"
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/${uid}}"
export DBUS_SESSION_BUS_ADDRESS="${DBUS_SESSION_BUS_ADDRESS:-unix:path=${XDG_RUNTIME_DIR}/bus}"

if systemctl --user cat orbweaver.service >/dev/null 2>&1; then
  systemctl --user restart orbweaver
else
  echo "orbweaver.service is not installed; skip restart" >&2
fi

# curl --retry does not retry connection-refused unless asked; uvicorn is not listening yet right after restart.
curl -fsS --retry 10 --retry-delay 1 --retry-connrefused --retry-max-time 30 http://127.0.0.1:8080/health
