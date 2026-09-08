# Orbweaver

This tree is the agent-editable checkout (`workspace:orbweaver`). The running
gateway is a separate clone at `/home/orbweaver/orbweaver` on `main`. Do not
restart the gateway, edit systemd units, or change files in the live clone.

Trusted git remote: `git@github.com:becomesaflame/orbweaver.git`.

## Layout

| Tree | Branch | Role |
| --- | --- | --- |
| `/home/orbweaver/workspaces/orbweaver` | feature branches → `main` | Agent work |
| `/home/orbweaver/orbweaver` | `main` only | Production gateway |

`main` is production. Do not commit on it. Do not push it. GitHub Actions
deploys a SHA after it lands on `main` through a pull request.

## Development process

Work on a **feature branch**. Do not commit on `main`.

1. `git fetch origin && git checkout -b <topic> origin/main`
2. Implement, run `python -m pytest tests -q` locally when practical.
3. Bump `__version__` in `backend/orbweaver/__init__.py` (see Versioning).
4. Push the feature branch. Open a **draft PR into `main`** as soon as there
   is something to review. Keep pushing the feature branch; CI tests run on
   the PR while it is draft.
5. When the feature is ready: mark the PR ready and **enable auto-merge
   (squash)** (`gh pr merge --squash --auto`). GitHub merges when `test`
   is green. Do not click Merge yourself. Never force-push. Never rewrite
   history. Never push `main`.

Do not `git push origin HEAD:main` from a feature branch. The auto-merged
squash is what updates production.

## CI pipeline

Workflows: `.github/workflows/ci.yml` (PRs only) and
`.github/workflows/deploy.yml` (`main` pushes only).

```
feature branch  --PR into main-->  tests (GitHub-hosted)
ready + auto-merge              -->  GitHub squash-merges when test is green
push to main                    -->  if LIVE_HOST_DEPLOY: self-hosted runner
                                     on lampropeltis runs deploy/update-live.sh
```

Pytest on the PR is the merge gate. Auto-merge is what lands the SHA; there
is no Actions job that merges PRs. Deploy is a separate workflow so a PR
does not show skipped promote/deploy checks. GitHub's merge is not an
Actions `GITHUB_TOKEN` push, so deploy does start.

Failed PR tests mean auto-merge waits (or the PR stays open). `main` stays
on the last green SHA.

The live host pulls `origin/main`, checks out `main`, and restarts
`orbweaver.service`. Agents do not deploy.

## Production regressions

When a production failure is diagnosed, add a test that reproduces it
(same command path, same error class) with the fix. Inspecting argv is
not enough when the bug is a runtime mount or OpenSSH check: execute
bubblewrap. CI installs `bubblewrap` so those tests run on the PR.

## Versioning

Semantic version **MAJOR.MINOR.PATCH** in `backend/orbweaver/__init__.py`
(`__version__`). `pyproject.toml` reads that attribute. The web UI, `GET
/health`, Telegram (`/version` and `/start`), and the agent system prompt all
report the **running gateway** version, which lags this checkout until deploy.

Every squash-merge that ships to `main` must bump the version on the feature
branch before merge:

- **PATCH** — bug fix, docs, CI, or other non-breaking change.
- **MINOR** — new capability, backward compatible. While the major version is
  `0`, breaking changes also bump MINOR (and reset PATCH), not MAJOR.
- **MAJOR** — reserved for a 1.0-style compatibility break after 0.x.

Do not ship two merges with the same version. Do not bump on the live clone.
The agent prompt version is not this working copy until production has
deployed.

## Trust and permissions

Pushing the feature branch, opening a PR into `main`, and enabling squash
auto-merge is allowed when the user asked to land or ship the work. Pushing
`main`, force-push, history rewrite, editing systemd, and restarting the
gateway are not.

Sandboxed Bash may read host files (including `/var/log`). The system journal
socket and this uid's user journal socket are granted by default, so
`journalctl -u` works without `permissions: ["all"]`. Extra sockets (local
Postgres) still go in `allowUnixSockets`. Do not use `permissions: ["all"]`
for log inspection. Do not restart the gateway or edit systemd units.
