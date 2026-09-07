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
5. When the feature is ready: squash, mark the PR ready, merge into `main`.
   Never force-push. Never rewrite history. Never push `main`.

Do not `git push origin HEAD:main` from a feature branch. The PR squash-merge
is what updates production.

## CI pipeline

Workflow: `.github/workflows/ci.yml`.

```
feature branch  --PR into main-->  tests (GitHub-hosted)
squash-merge into main          -->  push to main
                                 -->  if LIVE_HOST_DEPLOY: self-hosted runner
                                      on lampropeltis runs deploy/update-live.sh
```

PR checks are the merge gate. Deploy runs on the `main` push after merge
(GitHub's merge is not an Actions `GITHUB_TOKEN` push, so that workflow does
start). There is no `release-candidate` branch and no promote job.

Failed PR tests mean do not merge. `main` stays on the last green SHA.

The live host pulls `origin/main`, checks out `main`, and restarts
`orbweaver.service`. Agents do not deploy.

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

Pushing the feature branch and opening/merging a PR into `main` is allowed
when the user asked to land or ship the work. Pushing `main`, force-push,
history rewrite, editing systemd, and restarting the gateway are not.
