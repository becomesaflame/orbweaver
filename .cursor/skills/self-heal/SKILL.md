---
name: self-heal
description: Diagnose a production Orbweaver error, add a regression test, and open a fix PR.
---

# Self-heal

You were started because the gateway fingerprinted a production failure. Work only on **that** fingerprint (in the user message). One PR. This workspace is `workspace:orbweaver-selfheal`, not the live clone.

## Skip

If `git ls-remote origin 'refs/heads/selfheal/*'` or `gh pr list --search '<fingerprint>' --state all` already has this fingerprint, reply `SELFHEAL_NOOP` and stop.

## Fix

1. Clone `git@github.com:becomesaflame/orbweaver.git` into this directory if `.git` is missing.
2. `git fetch origin && git checkout -B selfheal/<slug> origin/main` (slug = fingerprint with non-alphanumerics as `-`).
3. Write a test that raises the same error class on the same command path, then fix it.
4. `python -m pytest tests -q`. Bump PATCH in `backend/orbweaver/__init__.py`.
5. Push the feature branch. `gh pr create --draft` with fingerprint, traceback excerpt, diagnosis, test output.
6. You may `gh pr ready` and `gh pr merge --squash --auto` when the fix is ready. Never push `main`. Never edit `/home/orbweaver/orbweaver`, systemd, or deploy.

Reply with the PR URL, or `SELFHEAL_NOOP`.
