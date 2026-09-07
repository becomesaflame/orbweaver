# Orbweaver working copy

This tree is the agent-editable checkout (`workspace:orbweaver`). The running gateway is a different clone. Do not restart the gateway, edit systemd units, or pull in `/home/orbweaver/orbweaver`.

## Git

- Work on `release-candidate`. Create it from `origin/release-candidate` if needed.
- Never push `main`. Never force-push. Never rewrite history.
- `main` is production. Only GitHub Actions may fast-forward it after tests pass on `release-candidate`.
- After local tests, `git push origin HEAD:release-candidate` (or push the current RC branch). CI promotes and deploys; you do not.

## Trust

Trusted git remote: `git@github.com:becomesaflame/orbweaver.git`. Pushing that remote's `release-candidate` branch is allowed when the user asked to land or ship the work.
