#!/usr/bin/env bash
# Restart the live orbweaver user unit from spark (or any sudoer).
# --machine=orbweaver@.host fails (spark cannot use machined). A login
# shell for orbweaver also lacks the user-session bus. Point systemctl
# at /run/user/<orbweaver-uid>/bus the same way deploy/update-live.sh does.
set -euo pipefail

uid="$(id -u orbweaver)"
runtime="/run/user/${uid}"
bus="unix:path=${runtime}/bus"

sudo -u orbweaver env \
  XDG_RUNTIME_DIR="${runtime}" \
  DBUS_SESSION_BUS_ADDRESS="${bus}" \
  systemctl --user restart orbweaver

# curl --retry does not retry connection-refused unless asked; uvicorn is
# not listening yet right after restart.
curl -fsS --retry 10 --retry-delay 1 --retry-connrefused --retry-max-time 30 \
  http://127.0.0.1:8080/health
echo
