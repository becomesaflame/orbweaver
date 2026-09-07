# Orbweaver

This tree is the agent-editable checkout (`workspace:orbweaver`). The running
gateway is a separate clone at `/home/orbweaver/orbweaver` on `main`. Do not
restart the gateway, edit systemd units, or change files in the live clone.

Trusted git remote: `git@github.com:becomesaflame/orbweaver.git`.

## Layout

| Tree | Branch | Role |
| --- | --- | --- |
| `/home/orbweaver/workspaces/orbweaver` | feature branches → `release-candidate` | Agent work |
| `/home/orbweaver/orbweaver` | `main` only | Production gateway |

`main` is production. Only GitHub Actions may update it, and only with a
fast-forward from a green `release-candidate`.

## Development process

Work on a **feature branch**. Do not commit on `main` or directly on
`release-candidate`.

1. `git fetch origin && git checkout -b <topic> origin/release-candidate`
2. Implement, run `python -m pytest tests -q` locally when practical.
3. Bump `__version__` in `backend/orbweaver/__init__.py` (see Versioning).
4. Push the feature branch. Open a **draft PR into `release-candidate`** as
   soon as there is something to review. Keep pushing the feature branch; CI
   tests run on the PR while it is draft.
5. When the feature is ready: squash, mark the PR ready, merge into
   `release-candidate`. Never force-push. Never rewrite history. Never push
   `main`.

Do not `git push origin HEAD:release-candidate` from a feature branch. The PR
squash-merge is what updates RC.

## CI pipeline

Workflow: `.github/workflows/ci.yml`.

```
feature branch  --push-->  tests (GitHub-hosted)
draft/open PR to RC     -->  tests (GitHub-hosted)
squash-merge into RC    -->  push to release-candidate
                         -->  tests again
                         -->  if green: fast-forward main to that SHA
                         -->  if LIVE_HOST_DEPLOY: self-hosted runner on
                              lampropeltis runs deploy/update-live.sh
```

Corrections vs a "merge to main triggers deploy" picture:

- GitHub does **not** deploy on a `main` push event. The promote job pushes
  `main` with `GITHUB_TOKEN`, which does not start a second workflow.
  Deploy is the next job in the **same** `release-candidate` run, after
  promote succeeds.
- `main` is updated with a **fast-forward only** (`git push` of the RC tip).
  There is no merge commit on `main`.
- A push to a feature branch runs tests only. Promote and deploy run only
  when `release-candidate` itself is pushed (the squash-merge).
- Failed tests leave `main` on the last green SHA. RC may be ahead until
  someone fixes it; do not skip CI or force-push RC.

The live host pulls `origin/main`, checks out `main`, and restarts
`orbweaver.service`. Agents do not deploy.

## Versioning

Semantic version **MAJOR.MINOR.PATCH** in `backend/orbweaver/__init__.py`
(`__version__`). `pyproject.toml` reads that attribute. The web UI, `GET
/health`, Telegram (`/version` and `/start`), and the agent system prompt all
report the **running gateway** version, which lags this checkout until deploy.

Every squash-merge that ships to RC must bump the version on the feature
branch before merge:

- **PATCH** — bug fix, docs, CI, or other non-breaking change (this includes
  process/docs updates).
- **MINOR** — new capability, backward compatible. While the major version is
  `0`, breaking changes also bump MINOR (and reset PATCH), not MAJOR.
- **MAJOR** — reserved for a 1.0-style compatibility break after 0.x.

Do not ship two merges with the same version. Do not bump on the live clone.
The agent prompt version is not this working copy until production has
deployed.

## Trust and permissions

Pushing the feature branch and opening/merging a PR into `release-candidate`
is allowed when the user asked to land or ship the work. Pushing `main`,
force-push, history rewrite, editing systemd, and restarting the gateway are
not.
