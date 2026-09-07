# Provenance

Borrowed trees, origins, and rewrite priority. Update this file when vendoring or wrapping code.

| Path | Upstream | License | Notes | Rewrite? |
|------|----------|---------|-------|----------|
| Agent loop | Original Orbweaver Python (`agent.py`) | n/a | Anthropic Messages API tool loop, prompt-projection compaction. | no |
| Compaction cascade | Claude Code leak-port patterns (boundary, microcompact, session notes, cache-prefixed summary); public writeups of the CLI compact pipeline | mixed / pattern-only | Patterns only. Original prompts and Python. No leaked TypeScript copied. | n/a (patterns) |
| Permissions / auto mode / sandbox patterns | Claude Code leak ports (Clawd-Code, pycc, Claw Code); public Anthropic auto-mode and sandboxing docs; public Cursor sandbox.json / run-modes docs; anthropics/sandbox-runtime (Apache-2.0) as Linux bwrap reference | mixed / pattern-only | Pipeline order, two-stage reasoning-blind classifier, denial limits, bwrap write confinement, named full_network/all escalation, domain allowlist, Unix-socket grants. No leaked TypeScript or classifier prompt files copied. | n/a (patterns) |
| VS Code inline diffs | Continue `VerticalDiffManager` / `VerticalDiffHandler` | Apache-2.0 | Reimplemented against VS Code decoration APIs; not a vendor copy of Continue source. | no |
| Telegram adapter | python-telegram-bot | LGPL | Dependency, not vendored. | no |
| FastAPI / pgvector / Anthropic SDK | respective projects | MIT / PostgreSQL / proprietary SDK | Dependencies. | no |

First-pass implementation is original Orbweaver code wrapping public APIs.
