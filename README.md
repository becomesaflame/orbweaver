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
export ANTHROPIC_API_KEY=...    # optional; without it the loop echoes
export ORBWEAVER_JWT_SECRET=$(python3 -c "import secrets; print(secrets.token_urlsafe(48))")
# Open models via Earth Runtime (https://earthruntime.com):
# export OPENROUTER_API_KEY=pk-prov-...
# export OPENROUTER_BASE_URL=https://api.earthruntime.com/v1
# export ORBWEAVER_MODEL=gpt-oss-120b   # or qwen3.6-35b, qwen3.8-27b, deepseek-v4-flash-0731
python3 -m orbweaver.cli serve
```

`serve` refuses to start when `ORBWEAVER_JWT_SECRET` is unset, the default, or shorter than 32 bytes (anyone with the source could forge tokens). For a throwaway local run set `ORBWEAVER_DEV_INSECURE=1` instead. Cross-origin browser clients need `ORBWEAVER_CORS_ORIGINS=https://a.example,https://b.example`; the bundled web UI is same-origin and needs nothing.

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

Shared open-model pools shed load as HTTP 429 (`temporarily rate-limited
upstream`). Those and transient 500/504 are retried in-process with exponential
backoff and full jitter, honouring `Retry-After` when the provider sends one:

| Env | Default | Meaning |
| --- | --- | --- |
| `ORBWEAVER_LLM_RETRIES` | `3` | Retries after the first attempt; `0` disables |
| `ORBWEAVER_LLM_RETRY_BASE_S` | `0.5` | First backoff window, doubling per attempt |
| `ORBWEAVER_LLM_RETRY_MAX_S` | `8.0` | Ceiling for any single wait |

HTTP 502 is deliberately *not* retried: Earth Runtime wraps context-overflow
failures as 502, and compacting the transcript is the right response, so that
call stays with the overflow path.

HTTP 503 (and Anthropic 529 overloaded) means the *model* is unavailable.
Orbweaver does not sleep-retry the same id; it falls back to a different model,
preferring a different provider (Anthropic ↔ Earth Runtime).

### Per-channel models and the picker

Each client channel has its own default. Web chat and Telegram default to
**Auto**. Auto runs the user prompt through a lightweight model
(`ORBWEAVER_ROUTER_MODEL`, Haiku by default) that replies with one id from the
live catalog; if that call fails or the reply is unusable, Auto uses the ranked
default (`ORBWEAVER_MODEL` first, then a different provider). Empty
`ORBWEAVER_ROUTER_MODEL` skips the LLM pass. VS Code, cron jobs, and subagents
use `ORBWEAVER_MODEL` unless overridden:

| Env | Channel | Unset default |
| --- | --- | --- |
| `ORBWEAVER_WEB_MODEL` | web chat | `auto` |
| `ORBWEAVER_TELEGRAM_MODEL` | Telegram | `auto` |
| `ORBWEAVER_VSCODE_MODEL` | VS Code extension | `ORBWEAVER_MODEL` |
| `ORBWEAVER_MODEL` | cron, subagents, Auto's ranked fallback | `claude-sonnet-4-6` |
| `ORBWEAVER_ROUTER_MODEL` | Auto's classifier | `claude-haiku-4-5` |

The web composer and the VS Code chat view have a model `<select>` fed by `GET /v1/models` (supported ids including `auto`, a short label, whether a key is configured for each, and the channel defaults). Picking one stores it on the session (`model` on the session JSON-LD; also accepted on `POST /v1/sessions`, `PATCH /v1/sessions/{id}`, `POST …/turns` and the WebSocket `text` frame), so every later turn of that chat, from any client, uses it. "Default" clears the override. Telegram does the same with `/model` (list) and `/model <id>` (switch; prefixes and labels match when unique; `/model auto` selects the router; `/model default` clears). A turn resolves its model as: turn `model` → session `model` → channel env → `ORBWEAVER_MODEL`; Auto is then replaced with a concrete id via the prompt router. The context window and compaction budget follow the model actually called. `orbweaver.model` in VS Code settings seeds new chats when no picker choice was made. `/health` reports the web default.

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

Each Telegram user gets one **operator** session (`workspace:default`). That session stays the dispatcher: inbound messages always run on it. The operator can list other sessions, summarize them, **prompt** one (`PromptSession`), or **create** a new web chat (`CreateSession`) when none of the existing ones is the right place. Prompting injects into a running turn or starts a new turn on that session — same event stream, workspace, model, todos, and that session's own tools. User events typed this way carry `via: "telegram"`. Replying to a tagged bot report (`[title · a1b2c3d4]`) injects into that session without an operator hop. Approvals and `AskUser` pings from any chat are forwarded here; turn completions are reported only for sessions Telegram recently prompted (plus errors and aborts). Turns run as background tasks so commands keep working while the agent is busy.

| Command | Effect |
| --- | --- |
| `/sessions` | List sessions (id prefix, title, age, running turn, waiting for an answer) |
| `/model` | List models for the operator chat |
| `/model <id>` | Switch the operator's model (prefix or label if unique); `/model default` clears |
| `/status` | Operator session, watching list, running turn, pending question |
| `/stop` | Cancel the operator turn, or the last prompted session; reply-to `/stop` cancels that tagged session |

The operator's own turns get extra allowlisted tools — `ListSessions`, `SessionDigest`, `PromptSession`, `CreateSession`, `StopSession` — so "what was I doing on the airbed controller?" is answered by reading the other session's log and prompting it, or starting a new web chat when none of the existing ones fit. A leftover `attached_session` from the old attach model is cleared on the next operator load.

### File uploads

Drag a file onto the chat, use the paperclip, or paste one; on Telegram, send
any document. Files land in the session workspace under `attachments/` and the
agent gets their text.

| Kind | Handling |
| --- | --- |
| Text, code, Markdown, JSON, CSV, YAML, logs | Decoded UTF-8, falling back to latin-1 |
| PDF | Text per page via `pypdf` |
| DOCX | Unzipped with the stdlib; tables flatten, headers/footnotes are dropped |
| Images | Routed to the existing vision path, never text-extracted |
| Anything else | Reported as unsupported binary, still stored |

Detection reads content before trusting the extension, so a mislabelled file
still parses. Malformed input never raises — a corrupt PDF or bogus zip comes
back as a note.

Two caps apply: extraction holds at most 200k characters, and at most 4k of
those are inlined into the turn, with the stored path named so the agent can
`Read` the rest. Uploads are capped at 25 MB (`413` past that); Telegram is
capped at its own 20 MB bot download limit.

### Cron

The host polls due jobs every 30s. A finished turn notifies the originating Telegram chat (success and abort) and appends a `cron_result` session event for web chat. Recurrence is `minute`, `hour`, or `day` (also `every hour`), or a 5-field cron expression (`minute hour day-of-month month day-of-week`, e.g. `0 9 * * mon`). Day-of-week `0` is Monday; names like `mon` work. Omit recurrence for a one-shot — those are deleted after they run.

### Permissions and sandbox

Default mode is **auto**: in-project writes apply immediately; a transcript classifier (Sonnet, `ORBWEAVER_CLASSIFIER_MODEL`) reviews `Bash` with `permissions: ["full_network"]` or `["all"]`, WebFetch, Browser, and `SpawnSubagent`. It returns **allow**, **ask**, or **hard-deny**. `WebSearch` talks only to DuckDuckGo or Brave and is allowlisted like Grep; deny/ask rules still apply. Sandbox limits and soft-deny actions check in with the user; hard-deny (secrets, persistence, self-granting power) stays rare. A separate Haiku injection probe (`ORBWEAVER_INJECTION_PROBE_MODEL`) warns on untrusted tool output. By default (`ORBWEAVER_INJECTION_PROBE_MODE=scoped`) it always probes WebFetch/WebSearch/Browser/MCP/memory results, probes Bash only when the command touches the network or the output has instruction-like phrasing, and skips in-project Read/Grep/Glob (except vendored dirs such as `node_modules`); `all` probes every result, `off` disables it. Probes for a tool round run concurrently after the round; the next LLM call waits at most `ORBWEAVER_INJECTION_PROBE_BUDGET_S` (3 s) for them, and a slower verdict lands as an `injection_warning` on the following prompt. Telegram/cron abort on classifier asks or repeated hard denials and **always tell the user why**. `AskUser` waits for a reply on web and Telegram; cron and other headless sessions abort instead of pretending to wait.

Linux Bash for `LocalWorkspace` runs in **bubblewrap**. The sandbox is write confinement, not a hidden host: files are readable, writes stay in the working set (workspace plus extra roots), a private `/dev` is remounted after the host bind so `/dev/null` works, `/run` is hidden except `allowUnixSockets` (the systemd journal sockets are granted by default), and outbound HTTP uses a CONNECT proxy with a package-manager domain allowlist. RFC1918, loopback, and cloud metadata are blocked on that network path. `unsandboxed: true` aliases `permissions: ["all"]` for one minor version.

Configure extra roots, sockets, and domains in `~/.orbweaver/sandbox.json`, `$WORKSPACE_ROOT/.orbweaver/sandbox.json`, or `ORBWEAVER_SANDBOX_CONFIG` (see `deploy/sandbox.json.example`). `denyRead` overlays skip paths that do not exist so a missing `~/.gnupg` cannot prevent the sandbox from starting.

Credential files are masked by default (`DEFAULT_DENY_READ` in `sandbox/policy.py`): `~/.ssh`, `~/.gnupg`, `~/.aws`, `~/.netrc`, `~/.git-credentials`, `~/.config/gh`, `~/.config/hub`, `~/.config/gcloud`, `~/.docker/config.json`, `~/.kube`, `~/.azure`, `~/.npmrc`, `~/.pypirc`, `~/.cargo/credentials*`, plus the running gateway checkout's `backend/.env` and other `.env*` files. Directories become an empty tmpfs and files read as `/dev/null`; the `Read` tool denies the same paths. Put an entry back with `allowRead` (e.g. `["~/.npmrc"]`) or `ORBWEAVER_SANDBOX_ALLOW_READ`. Private SSH keys never enter the sandbox by default: `~/.ssh/config`, `known_hosts`, and `*.pub` are re-exposed, and an ssh-agent (`SSH_AUTH_SOCK`) signs from the host. To bind keys anyway set `"ssh": {"bindIdentities": true}` (host `IdentityFile` entries and the standard `id_*` names) or list them in `"ssh": {"identities": [...]}`. `gh` is wrapped the same way: the sandbox talks to a host-side proxy so `gh pr` / `gh issue` can use the gateway user's `GH_TOKEN` / `GITHUB_TOKEN` / `~/.config/gh` without those files or variables entering the sandbox. `gh auth token`, `login`, and `logout` are refused. `allowRead` and `ssh` are honoured only in `~/.orbweaver/sandbox.json`, `ORBWEAVER_SANDBOX_CONFIG`, and the environment — the workspace file is agent-writable and cannot widen the sandbox.
Sandboxed Bash does not inherit the gateway environment. bwrap starts with `--clearenv` and receives only an allowlist (`PATH`, `HOME`, `USER`, `LOGNAME`, `SHELL`, `TERM`, `LANG`, `LC_*`, `TZ`, `TMPDIR`, host proxy variables, and `SSH_AUTH_SOCK` when that socket is granted). Add names or globs with `env.allow` in `sandbox.json` or `ORBWEAVER_SANDBOX_ENV_ALLOW`; `*KEY*`, `*SECRET*`, `*TOKEN*`, `*PASSWORD*`, `ORBWEAVER_*`, `DATABASE_URL`, `ANTHROPIC_*`, `OPENAI_*`, `OPENROUTER_*`, `TELEGRAM_*`, and `HINDSIGHT_*` are never passed through.
`WebFetch` and `Browser` run on the host, outside bubblewrap, but follow the same `networkPolicy`: the URL and every redirect hop are resolved first, loopback / RFC1918 / link-local (including `169.254.169.254`) addresses are refused, `WebFetch` connects to the vetted address (`Host` and SNI keep the hostname), and Playwright aborts sub-resources and JS navigations that fail the check. `deny` patterns always apply and `allow` patterns always grant; when nothing matches, `networkPolicy.webDefault` (`ORBWEAVER_SANDBOX_WEB_NETWORK_DEFAULT`, default `allow`) decides so the public web stays reachable while Bash keeps `default: deny`. Set `webDefault: "deny"` to confine both tools to the allowlist. WebFetch downloads at most 2 MB and only text-like content types.

### MCP servers

Sessions can use external MCP tools. Configure servers in `~/.orbweaver/mcp.json`, `$WORKSPACE_ROOT/.orbweaver/mcp.json`, or `ORBWEAVER_MCP_CONFIG` (see `deploy/mcp.json.example`). A server with `command` runs over stdio; one with `url` uses Streamable HTTP (MCP 2025-06-18, falling back to what the server negotiates) with optional `headers` for a bearer token. Later files override command/args/url; `env` and `headers` maps merge so host tokens survive a workspace command override. Tools appear as `mcp_<server>_<tool>` and go through the same deny/ask/allow/classifier pipeline as other tools — they are not blanket-allowlisted, except that tools the server annotates `readOnlyHint: true` are auto-allowed (`ORBWEAVER_MCP_AUTO_ALLOW_READONLY=false` to disable) and `destructiveHint: true` tools are framed as soft-deny for the classifier. Servers that advertise resources also get `mcp_<server>_read_resource` / `mcp_<server>_list_resources`. Per-server `timeout_s` (default 30) and `startup_timeout_s` (default 8) bound calls; a timeout is an error result, not a hang.

Stdio servers start with a minimal environment (`PATH`, `HOME`, `USER`, `LANG`, `LC_*`, `TERM`, `TMPDIR`, `TZ`) plus the server's explicit `env`. They never inherit `ANTHROPIC_*`, `OPENAI_*`, `ORBWEAVER_*`, `TELEGRAM_*`, `DATABASE_URL`, or anything matching `*KEY*`/`*SECRET*`/`*TOKEN*`/`*PASSWORD*`. To hand a server one specific host variable, list it in `envPassthrough` (per server, at the top of `mcp.json`, or `ORBWEAVER_MCP_ENV_PASSTHROUGH`); listed names are forwarded and can be referenced as `${VAR}` in `env` and `headers`. `${VAR}` for anything not listed expands to an empty string with a warning. Gateway namespaces (`ORBWEAVER_*`, `ANTHROPIC_*`, ...) cannot be passed through even when listed — copy the value into a differently named host variable if a server really needs it. Put API tokens in the host file, not the workspace. Phase 6 snapshot packs must omit `mcp.json` (and its `env` blocks); those secrets stay on the destination host. The agent cannot Read or sandbox-write `.orbweaver/mcp.json`.

### Tool hooks

Optional `PreToolUse` / `PostToolUse` / `PostToolUseFailure` commands live in `~/.orbweaver/hooks.json`, `$WORKSPACE_ROOT/.orbweaver/hooks.json`, or `ORBWEAVER_HOOKS_CONFIG` (see `deploy/hooks.json.example`). Later files append. PreToolUse runs before the permission gate and can deny, cancel the turn, or rewrite tool input. PostToolUse appends hook stdout/stderr onto a successful tool result; PostToolUseFailure runs when the result looks like an error. A hook that would prompt (type `prompt`, or JSON `ask`) fails closed instead of hanging — cron and other headless channels never get a TTY prompt. The agent cannot Read or sandbox-write `.orbweaver/hooks.json`.

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
cd web && npm install && npx playwright test   # optional; CI uses runner Chrome, skip with SKIP_PLAYWRIGHT
```

Playwright integration tests skip unless the `[browser]` extra and Chromium are installed.
