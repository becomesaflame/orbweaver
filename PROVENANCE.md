# Provenance

Borrowed trees, origins, and rewrite priority. Update this file when vendoring or wrapping code.

| Path | Upstream | License | Notes | Rewrite? |
|------|----------|---------|-------|----------|
| Agent loop patterns | OpenHands SDK, Claude Code leak ports (Clawd-Code, pycc, Claw Code) | mixed / pattern-only | Patterns only: tool loop, compaction, permissions, skills, MCP, subagents. No leaked TypeScript copied. | n/a (patterns) |
| VS Code inline diffs | Continue `VerticalDiffManager` / `VerticalDiffHandler` | Apache-2.0 | Reimplemented against VS Code decoration APIs; not a vendor copy of Continue source. | no |
| Telegram adapter | python-telegram-bot | LGPL | Dependency, not vendored. | no |
| FastAPI / pgvector / Anthropic SDK | respective projects | MIT / PostgreSQL / proprietary SDK | Dependencies. | no |

First-pass implementation is original Orbweaver code wrapping public APIs.
