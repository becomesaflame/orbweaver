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
python3 -m orbweaver.cli serve
```

Open http://127.0.0.1:8080/ — mint a JWT, create a session (`workspace:default` or `file:./rel`), send a turn.

### Postgres

```bash
docker compose up -d postgres
export ORBWEAVER_STORE=postgres
export DATABASE_URL=postgresql://orbweaver:orbweaver@localhost:5432/orbweaver
```

`docker compose up` also starts the gateway with a workspace bind mount.

### VS Code extension

```bash
cd vscode && npm install && npx tsc -p .
```

Then **Install from VSIX** or **Run Extension** from the `vscode/` folder. Set `orbweaver.gatewayUrl` and optionally `orbweaver.token`. Default workspace URI is `workspace:default` (override with `orbweaver.workspaceUri`). Commands: Open Chat, Accept/Reject Diff, Open Plan (Markdown or HTML).

### Telegram

Set `TELEGRAM_BOT_TOKEN` and `TELEGRAM_ALLOWLIST` (comma-separated user ids). Sessions use `DockerWorkspace`. Voice notes go through `/v1/stt` when `faster-whisper` is installed (`pip install -e ".[stt]"`).

### Tests

```bash
python3 -m pytest tests -q
```
