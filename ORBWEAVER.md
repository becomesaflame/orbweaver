# Orbweaver

This tree is the agent-attached checkout (`workspace:orbweaver`). Concurrent
agents share it — isolate in a git worktree before editing (see below). The
running gateway is a separate clone at `/home/orbweaver/orbweaver` on `main`.
Do not restart the gateway, edit systemd units, or change files in the live
clone.

Trusted git remote: `git@github.com:becomesaflame/orbweaver.git`.

## Layout

| Tree | Branch | Role |
| --- | --- | --- |
| `/home/orbweaver/workspaces/orbweaver` | whatever is checked out | Shared attach point; do not edit |
| `.worktrees/<slug>` inside that workspace | one feature branch | Per-agent work |
| `/home/orbweaver/orbweaver` | `main` only | Production gateway |

`main` is production. Do not commit on it. Do not push it. GitHub Actions
deploys a SHA after it lands on `main` through a pull request.

## One worktree per agent

Do not edit, commit, or `git checkout` in the checkout you were opened in.
Before the first edit:

```bash
git fetch origin
git worktree add -b <topic> .worktrees/<slug> origin/main
cd .worktrees/<slug>
```

The path is **inside** the session workspace. Sandboxed Bash can only write
that tree; `/home/orbweaver/workspaces/` itself is read-only, so a sibling
`../orbweaver-worktrees/` fails. `.worktrees/` is gitignored. File tools are
rooted at the session workspace: edit `.worktrees/<slug>/...`. Bash: `cd` or
`git -C`. Branch off `origin/main`. Read-only work needs no worktree.

Tear it down once the PR is `MERGED` (`gh pr view <n> --json state`), from
the original checkout. Report unmerged work instead of deleting it.

```bash
cd <original checkout>
git worktree remove .worktrees/<slug>
git branch -D <topic>                  # -d refuses after a squash-merge
git push origin --delete <topic>       # this repo does not delete branches on merge
```

Do not clone the repo to get isolation. Do not touch a worktree or topic
branch you did not create.

## Development process

Work on a **feature branch** in your worktree. Do not commit on `main`. Do not
`git checkout` a different branch in the shared tree.

1. `git fetch origin && git worktree add -b <topic> .worktrees/<slug> origin/main`
2. Implement, run `python -m pytest tests -q` locally when practical.
3. Bump `__version__` in `backend/orbweaver/__init__.py` (see Versioning).
4. Push the feature branch. Open a **draft PR into `main`** as soon as there
   is something to review. Keep pushing the feature branch; CI tests run on
   the PR while it is draft.
5. When the feature is ready: mark the PR ready and **enable auto-merge
   (squash)** (`gh pr merge --squash --auto`). GitHub merges when `test`
   is green. Do not click Merge yourself. Never push, force-push, or rewrite
   `main`.
6. **Stay with the PR until it merges.** Opening it and enabling auto-merge
   is not the end of the task. Watch CI, fix conflicts, and fix failing
   checks (see Stay with the PR). Parallel PRs land while yours is in CI;
   walking away is how the version bump conflicts and auto-merge stalls.

Do not `git push origin HEAD:main` from a feature branch. The auto-merged
squash is what updates production.

Your own feature branch is yours: rebase it and `git push --force-with-lease`
freely. Prefer that over merging `main` back in, especially for the version
bump — see Versioning. Leave branches you do not own alone.

## Stay with the PR

After every push that updates the PR, and especially after enabling
auto-merge, wait and check progress. Do not start unrelated work and
abandon this PR. CI still running is a reason to wait, not to leave.

1. Watch required checks to completion: `gh pr checks --watch`.
2. Refresh merge state: `gh pr view --json mergeable,mergeStateStatus,url`.
3. If the PR is `CONFLICTING`, `DIRTY`, or `BEHIND`: `git fetch origin &&
   git rebase origin/main`, resolve, confirm the version bump (`git diff
   origin/main -- backend/orbweaver/__init__.py` must be non-empty), then
   `git push --force-with-lease` and watch CI again.
4. If a required check failed: read that job's log, fix the failure, push,
   and watch again. Do not change CI config to make a failure pass.
5. Repeat until the PR is `MERGED`, or you are blocked on a human decision
   (ambiguous intent, security). Report the blocker; do not leave the PR
   conflicted or red.

Auto-merge waits forever on a DIRTY or failing PR. Your job is to get it
green and mergeable, then stay until GitHub squash-merges.

## CI pipeline

Workflows: `.github/workflows/ci.yml` (PRs only) and
`.github/workflows/deploy.yml` (`main` pushes only).

```
feature branch  --PR into main-->  tests (GitHub-hosted)
ready + auto-merge              -->  GitHub squash-merges when test is green
push to main                    -->  if LIVE_HOST_DEPLOY: self-hosted runner
                                     on lampropeltis runs deploy/update-live.sh
```

Pytest, ruff, mypy, and vscode `tsc` on the PR are the merge gate.
Playwright web smoke is a separate job (skip with repository variable
`SKIP_PLAYWRIGHT`). Auto-merge is what lands the SHA; there
is no Actions job that merges PRs. Deploy is a separate workflow so a PR
does not show skipped promote/deploy checks. GitHub's merge is not an
Actions `GITHUB_TOKEN` push, so deploy does start.

Failed PR tests mean auto-merge waits (or the PR stays open). Stay and fix
them. `main` stays on the last green SHA.

The live host pulls `origin/main`, checks out `main`, waits until running
turns finish (`orbweaver drain`), then restarts `orbweaver.service`. Agents
do not deploy.

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

Another PR landing first takes the number you bumped to and puts yours behind
`main`. Rebase and re-bump rather than merging `main` into the branch, so the
branch keeps one version-bump commit instead of accumulating merge commits
that resolve the same line twice:

```bash
git fetch origin && git rebase origin/main   # resolve __init__.py to a number above main
git diff origin/main -- backend/orbweaver/__init__.py   # must still show your bump
git push --force-with-lease
```

A rebase that reports success can leave you with **no** bump at all. If the PR
that landed first used the number you bumped to, your hunk is already applied,
git drops it silently, and there is no conflict to notice. An empty diff on
that second line means re-bump to the next free number before pushing.

Do **not** put the version in the PR title (or the squash-merge commit title
template). Parallel PRs each bump against the same `origin/main` base, so the
number in the title goes stale as soon as another PR merges first. The
authoritative version is `__version__` on the branch at merge time; retitle
only if needed after rebasing onto current `main`.

## Trust and permissions

Pushing the feature branch, opening a PR into `main`, and enabling squash
auto-merge is allowed when the user asked to land or ship the work. Rebasing
and force-pushing **your own** feature branch is allowed too. Pushing,
force-pushing, or rewriting `main`, force-pushing a branch you do not own,
editing systemd, and restarting the gateway are not.

Sandboxed Bash may read host files (including `/var/log`). The system journal
socket and this uid's user journal socket are granted by default, so
`journalctl -u` works without `permissions: ["all"]`. Extra sockets (local
Postgres) still go in `allowUnixSockets`. Do not use `permissions: ["all"]`
for log inspection. Do not restart the gateway or edit systemd units.

## Self-heal

When `ORBWEAVER_SELFHEAL=1`, the gateway fingerprints `orbweaver.*` ERROR
tracebacks (and some `turn_aborted` reasons) and enqueues a headless turn on
`workspace:orbweaver-selfheal`. Clone that sibling of `workspace:orbweaver`
before enabling. `ORBWEAVER_SELFHEAL_TELEGRAM_CHAT_ID` gets a message when a
turn is enqueued and again when a PR URL is in the result (not on no-op /
cooldown skips). Caps:
`ORBWEAVER_SELFHEAL_DAILY_CAP` (default 3) and `ORBWEAVER_SELFHEAL_COOLDOWN_S`
(default 7 days). Duplicate fingerprints do not start a second turn. This does
not refuse `gh pr merge`; silent auto-merge is a skill choice, not a sandbox
rule. Pushing `main` is still forbidden.
