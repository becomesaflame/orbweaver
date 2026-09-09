# orbweaver

Personal multi-channel coding agent: Python backend, VS Code extension, Telegram, shared graph+vector memory.

**Architecture (living doc):** open [docs/architecture.html](docs/architecture.html) in a browser. GitHub shows HTML source; diagrams render locally (Mermaid from a CDN).

## First pass (phases 1–4)

Memory API, Anthropic agent loop, web chat, VS Code extension, Telegram + cron. Snapshot export/import is phase 6. Full CI matrix is phase 5.

### Run the gateway

```bash
cd backend
pip install -e ".[dev]"
export ORBWEAVER_STORE=memory   # default; use postgres with compose
export ANTHROPIC_API_KEY=...    # optional; without a provider the loop echoes
# Open models via Earth Runtime (https://earthruntime.com):
# export OPENROUTER_API_KEY=pk-prov-...
# export OPENROUTER_BASE_URL=https://api.earthruntime.com/v1
# export ORBWEAVER_MODEL=gpt-oss-120b   # or qwen3.6-35b, qwen3.8-27b, deepseek-v4-flash-0731
python3 -m orbweaver.cli serve
```

Mint a JWT on the gateway host (not over HTTP):

```bash
python3 -m orbweaver.cli mint
```

Open the web UI, paste the token, create a session (`workspace:default` or `file:./rel`), send a turn. HTTP minting is off unless `ORBWEAVER_ALLOW_HTTP_MINT=true`.

### Open models (Earth Runtime)

The agent loop speaks Anthropic Messages internally. For open-weight models it maps that to OpenAI-compatible `chat/completions` at Earth Runtime (the same env names OpenCode uses):

| `ORBWEAVER_MODEL` | Notes |
| --- | --- |
| `qwen3.6-35b` | Fastest; 262K context |
| `qwen3.8-27b` | Dense 27B; 262K context |
| `gpt-oss-120b` | Strongest of the three; 128K context |
| `deepseek-v4-flash-0731` | Reasoning enforced; 262K context |

Get a key from [earthruntime.com](https://earthruntime.com). The same catalog names work on `ORBWEAVER_CLASSIFIER_MODEL`, `ORBWEAVER_INJECTION_PROBE_MODEL`, and `ORBWEAVER_COMPACT_MODEL`: Claude ids stay on Anthropic, catalog names use Earth Runtime.

For a VPS, bind `ORBWEAVER_HOST=127.0.0.1` and publish the UI on your Tailscale interface (`tailscale serve http://127.0.0.1:8080`) instead of `0.0.0.0`.

### Postgres

```bash
docker compose up -d postgres
export ORBWEAVER_STORE=postgres
export DATABASE_URL=postgresql://orbweaver:orbweaver@localhost:5432/orbweaver
```

`docker compose up` also starts the gateway with a workspace bind mount.

### Hindsight (optional)

Long-term facts can go to a sibling [Hindsight](https://hindsight.vectorize.io) service. Pins, session events, and JSON-LD stay in Orbweaver. Set:

```bash
export HINDSIGHT_API_URL=http://127.0.0.1:8888
export HINDSIGHT_API_KEY=...          # same tenant key as the Hindsight server
export HINDSIGHT_BANK_ID=personal     # every harness must use this bank
```

```bash
docker compose --profile hindsight up -d
# gateway on the same compose network:
export HINDSIGHT_API_URL=http://hindsight:8888
```

Point Cursor / Claude Code at `http://127.0.0.1:8888/mcp/personal/` (or the Tailscale URL) with `Authorization: Bearer <HINDSIGHT_API_KEY>`. The gateway uses the HTTP SDK, not MCP (Orbweaver MCP is stdio-only).

### VS Code extension

```bash
cd vscode && npm install && npx tsc -p .
```

Then **Install from VSIX** or **Run Extension** from the `vscode/` folder. Set `orbweaver.gatewayUrl` and `orbweaver.token` (from `python -m orbweaver.cli mint`). Default workspace URI is `workspace:default` (override with `orbweaver.workspaceUri`). The extension opens sessions with `channel=vscode` so ProposePatch stays available. Chat streams over `/v1/sessions/{id}/ws` (Stop / inject / continue match web chat). ProposePatch opens a real `vscode.diff`; Accept / Reject / edit the file in place. Plan mode creates or iterates `.orbweaver/plan.md` (or HTML) with preview beside the editor.

### Telegram

Set `TELEGRAM_BOT_TOKEN` and `TELEGRAM_ALLOWLIST` (comma-separated user ids). Sessions use `LocalWorkspace` (bubblewrap). Leftover `workspace_kind=docker` rows are rewritten to `local` on the next turn. Voice notes go through `/v1/stt` when `faster-whisper` is installed (`pip install -e ".[stt]"`).

### Cron

The host polls due jobs every 30s. A finished turn notifies the originating Telegram chat (success and abort) and appends a `cron_result` session event for web chat. Recurrence is `minute`, `hour`, or `day` (also `every hour`), or a 5-field cron expression (`minute hour day-of-month month day-of-week`, e.g. `0 9 * * mon`). Day-of-week `0` is Monday; names like `mon` work. Omit recurrence for a one-shot — those are deleted after they run.

### Permissions and sandbox

Default mode is **auto**: in-project writes apply immediately; a transcript classifier (Sonnet, `ORBWEAVER_CLASSIFIER_MODEL`) reviews `Bash` with `permissions: ["full_network"]` or `["all"]`, WebFetch, Browser, and `SpawnSubagent`. It returns **allow**, **ask**, or **hard-deny**. `WebSearch` talks only to DuckDuckGo or Brave and is allowlisted like Grep; deny/ask rules still apply. Sandbox limits and soft-deny actions check in with the user; hard-deny (secrets, persistence, self-granting power) stays rare. A separate Haiku injection probe (`ORBWEAVER_INJECTION_PROBE_MODEL`) warns on untrusted tool output. Telegram/cron abort on classifier asks or repeated hard denials and **always tell the user why**. `AskUser` waits for a reply on web and Telegram; cron and other headless sessions abort instead of pretending to wait.

Linux Bash for `LocalWorkspace` runs in **bubblewrap**. The sandbox is write confinement, not a hidden host: files are readable, writes stay in the working set (workspace plus extra roots), a private `/dev` is remounted after the host bind so `/dev/null` works, `/run` is hidden except `allowUnixSockets` (the systemd journal sockets are granted by default), and outbound HTTP uses a CONNECT proxy with a package-manager domain allowlist. RFC1918, loopback, and cloud metadata are blocked on that network path. `unsandboxed: true` aliases `permissions: ["all"]` for one minor version.

Configure extra roots, sockets, and domains in `~/.orbweaver/sandbox.json`, `$WORKSPACE_ROOT/.orbweaver/sandbox.json`, or `ORBWEAVER_SANDBOX_CONFIG` (see `deploy/sandbox.json.example`). `denyRead` overlays skip paths that do not exist so a missing `~/.gnupg` cannot prevent the sandbox from starting.

### MCP servers

Sessions can use external MCP tools. Configure stdio servers in `~/.orbweaver/mcp.json`, `$WORKSPACE_ROOT/.orbweaver/mcp.json`, or `ORBWEAVER_MCP_CONFIG` (see `deploy/mcp.json.example`). Later files override command/args; `env` maps merge so host tokens survive a workspace command override. Tools appear as `mcp_<server>_<tool>` and go through the same deny/ask/allow/classifier pipeline as other tools — they are not blanket-allowlisted. Put API tokens in the host file, not the workspace. Phase 6 snapshot packs must omit `mcp.json` (and its `env` blocks); those secrets stay on the destination host. The agent cannot Read or sandbox-write `.orbweaver/mcp.json`.

On this host (Ubuntu with AppArmor userns restrictions):

```bash
sudo apt-get install -y bubblewrap
sudo tee /etc/apparmor.d/bwrap >/dev/null <<'EOF'
abi <abi/4.0>,
include <tunables/global>

profile bwrap /usr/bin/bwrap flags=(unconfined) {
  userns,
  include if exists <local/bwrap>
}
EOF
sudo systemctl reload apparmor
```

If bubblewrap cannot start, local Bash fails closed (`ORBWEAVER_SANDBOX_FAIL_IF_UNAVAILABLE=true`). Nested bwrap is skipped inside Compose containers.

### Browser tool (optional)

UI verification uses Playwright Chromium (`navigate`, `click`, `type`, `snapshot`, `screenshot`). It is classified like WebFetch, not auto-allowed. CI does not install the extra; wiring and permission tests always run.

```bash
cd backend
pip install -e ".[browser]"
playwright install chromium
```

### Tests

```bash
python3 -m pytest tests -q
cd backend && python -m ruff check . ../tests && python -m mypy orbweaver
cd vscode && npm ci && npx tsc -p . --noEmit
cd web && npm install && npx playwright test   # optional; skip in CI with SKIP_PLAYWRIGHT
```

Playwright integration tests skip unless the `[browser]` extra and Chromium are installed.
