from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    anthropic_api_key: str = ""
    anthropic_workspace_id: str = ""
    ollama_base_url: str = ""
    ollama_model: str = ""
    # Required in production: >= 32 bytes, not the default. `serve` refuses to
    # start otherwise unless ORBWEAVER_DEV_INSECURE=1.
    orbweaver_jwt_secret: str = "dev-secret-change-me"
    orbweaver_dev_insecure: bool = False
    # Comma-separated browser origins allowed for credentialed cross-origin
    # calls. Empty (default) means same-origin only; the bundled web UI is
    # served from the gateway itself and needs nothing here.
    orbweaver_cors_origins: str = ""
    orbweaver_model: str = "claude-sonnet-4-6"
    orbweaver_classifier_model: str = "claude-sonnet-4-6"
    orbweaver_injection_probe_model: str = "claude-haiku-4-5"
    orbweaver_permission_mode: str = "auto"
    orbweaver_permission_deny: str = ""
    orbweaver_permission_ask: str = ""
    orbweaver_permission_allow: str = ""
    orbweaver_automode_environment: str = "$defaults"
    orbweaver_automode_soft_deny: str = "$defaults"
    orbweaver_automode_hard_deny: str = "$defaults"
    orbweaver_automode_allow: str = "$defaults"
    orbweaver_sandbox: bool = True
    orbweaver_sandbox_fail_if_unavailable: bool = True
    orbweaver_auto_allow_bash_if_sandboxed: bool = True
    orbweaver_sandbox_config: str = ""
    orbweaver_mcp_config: str = ""
    # Host variable names (comma-separated) every MCP server may reference via
    # ${VAR} in mcp.json env/headers; per-server `envPassthrough` adds to it.
    orbweaver_mcp_env_passthrough: str = ""
    # Auto-allow mcp_* tools whose server annotates readOnlyHint: true.
    orbweaver_mcp_auto_allow_readonly: bool = True
    # Concurrency-safe tool calls from one assistant round that may run at once.
    orbweaver_max_parallel_tools: int = 8
    orbweaver_sandbox_additional_readonly: str = ""
    orbweaver_sandbox_additional_readwrite: str = ""
    orbweaver_sandbox_deny_read: str = ""
    orbweaver_sandbox_allow_read: str = ""
    orbweaver_sandbox_ssh_identities: str = ""
    orbweaver_sandbox_ssh_bind_identities: str = ""
    orbweaver_sandbox_unix_sockets: str = ""
    orbweaver_sandbox_allowed_domains: str = ""
    orbweaver_sandbox_denied_domains: str = ""
    orbweaver_sandbox_network_default: str = ""
    orbweaver_sandbox_web_network_default: str = ""
    orbweaver_sandbox_include_default_domains: str = ""
    # Subagents: process-wide cap on concurrently running children, per-child
    # defaults for wall time and tool rounds, and how long a finishing parent
    # turn waits for background children it never collected.
    orbweaver_max_concurrent_subagents: int = 4
    orbweaver_subagent_timeout_s: float = 600.0
    orbweaver_subagent_max_rounds: int = 24
    orbweaver_subagent_grace_s: float = 30.0
    # bwrap hardening (issue #114). Limits apply to the sandboxed bash and its
    # children via ulimit; 0 disables a limit. seccomp: auto | on | off.
    orbweaver_sandbox_max_procs: int = 512
    orbweaver_sandbox_max_mem_mb: int = 2048
    orbweaver_sandbox_max_open_files: int = 4096
    orbweaver_sandbox_seccomp: str = "auto"
    orbweaver_sandbox_hide_sys: bool = True
    orbweaver_sandbox_env_allow: str = ""  # extra env names/globs passed into sandboxed Bash
    orbweaver_checkpoints: bool = True  # per-turn git tree of the workspace for rewind
    orbweaver_checkpoint_keep_days: int = 14  # prune refs/orbweaver/checkpoints older than this
    orbweaver_pinned_token_cap: int = 4000
    orbweaver_skills_token_cap: int = 4000
    # Per-item cap for one instruction doc / rule / skill body inlined in a prompt.
    orbweaver_instruction_item_token_cap: int = 2000
    # User-level instructions; empty → <orbweaver_data_dir or ~/.orbweaver>/AGENTS.md
    orbweaver_user_instructions: str = ""
    context_window: int = 200_000
    output_reserve: int = 16_000
    static_token_estimate: int = 12_000
    # Compact when estimate_prompt_tokens exceeds this fraction of event_budget
    # (~85% of the 200k window). Claw Code uses 100k cumulative input tokens.
    compact_ratio: float = 0.85
    orbweaver_compact_model: str = "claude-haiku-4-5"
    # Microcompact (stub old tool results in the prompt) runs only under pressure:
    # when the payload estimate of the live window exceeds compact_micro_pressure
    # of event_budget it clears the oldest large results, in chunks of
    # (pressure - release) * event_budget, until the window is back under the
    # pressure line. Chunking keeps the message prefix byte-identical between
    # consecutive rounds so Anthropic prompt caching keeps hitting.
    compact_micro_pressure: float = 0.6
    compact_micro_release: float = 0.4
    # Floor: the newest N compactable results are never stubbed.
    compact_micro_keep: int = 5
    # Independent of pressure: a result older than this many tool rounds *and*
    # larger than this many chars is stubbed (0 rounds disables the age rule).
    compact_micro_stale_rounds: int = 24
    compact_micro_stale_chars: int = 20000
    compact_tool_result_chars: int = 8000
    compact_max_failures: int = 3
    compact_overflow_retries: int = 4
    compact_rehydrate_files: int = 5
    compact_rehydrate_chars_per_file: int = 20000
    compact_rehydrate_token_budget: int = 50000
    compact_notes_max_chars: int = 12000
    # Loop detection (orbweaver.stuck). Streaks count tool calls since the last user
    # message; stuck_window is how many recent tool calls are scanned.
    stuck_detection: bool = True
    stuck_repeat_threshold: int = 3
    stuck_alternating_threshold: int = 6
    stuck_window: int = 20
    database_url: str = "postgresql://orbweaver:orbweaver@localhost:5432/orbweaver"
    orbweaver_store: str = "memory"  # memory | postgres
    workspace_root: str = "."
    orbweaver_data_dir: str = ""  # empty → ~/.orbweaver (file-backed rate limits)
    telegram_bot_token: str = ""
    telegram_allowlist: str = ""
    orbweaver_image_api_key: str = ""
    orbweaver_image_api_url: str = "https://api.openai.com/v1/images/generations"
    orbweaver_image_model: str = "dall-e-3"
    orbweaver_image_size: str = "1024x1024"
    orbweaver_linter: str = ""  # e.g. ruff check {paths} ; empty uses the Python AST stub
    # Write/StrReplace/NotebookEdit/Delete on an existing file require a prior Read
    # this session and refuse when the file changed on disk since that Read.
    edit_require_read: bool = True
    orbweaver_search_provider: str = ""  # duckduckgo | brave | empty auto
    orbweaver_brave_api_key: str = ""
    brave_search_api_key: str = ""
    orbweaver_host: str = "0.0.0.0"
    orbweaver_port: int = 8080
    orbweaver_allow_http_mint: bool = False
    orbweaver_trust_proxy: bool = False
    embedding_dim: int = 384
    embedding_model: str = "hash://blake2b-384"  # sentence-transformers name when using embed extra
    hindsight_api_url: str = ""  # empty → native chunk memory only
    hindsight_api_key: str = ""
    hindsight_bank_id: str = "personal"

    @property
    def telegram_user_ids(self) -> set[int]:
        if not self.telegram_allowlist.strip():
            return set()
        return {int(x.strip()) for x in self.telegram_allowlist.split(",") if x.strip()}

    @property
    def cors_origins(self) -> list[str]:
        return [o.strip() for o in self.orbweaver_cors_origins.split(",") if o.strip()]

    @property
    def brave_api_key(self) -> str:
        return self.orbweaver_brave_api_key.strip() or self.brave_search_api_key.strip()

    @property
    def search_provider(self) -> str:
        spec = self.orbweaver_search_provider.strip().lower()
        if spec in {"brave", "duckduckgo"}:
            return spec
        if self.brave_api_key:
            return "brave"
        return "duckduckgo"

    event_budget_override: int | None = None

    @property
    def event_budget(self) -> int:
        if self.event_budget_override is not None:
            return self.event_budget_override
        raw = (
            self.context_window
            - self.output_reserve
            - self.static_token_estimate
            - self.orbweaver_pinned_token_cap
            - self.orbweaver_skills_token_cap
        )
        return max(256, raw)


settings = Settings()
