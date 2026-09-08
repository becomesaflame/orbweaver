from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    anthropic_api_key: str = ""
    anthropic_workspace_id: str = ""
    orbweaver_jwt_secret: str = "dev-secret-change-me"
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
    orbweaver_sandbox_additional_readonly: str = ""
    orbweaver_sandbox_additional_readwrite: str = ""
    orbweaver_sandbox_deny_read: str = ""
    orbweaver_sandbox_unix_sockets: str = ""
    orbweaver_sandbox_allowed_domains: str = ""
    orbweaver_sandbox_denied_domains: str = ""
    orbweaver_sandbox_network_default: str = ""
    orbweaver_sandbox_include_default_domains: str = ""
    orbweaver_pinned_token_cap: int = 4000
    context_window: int = 200_000
    output_reserve: int = 16_000
    static_token_estimate: int = 12_000
    compact_ratio: float = 0.85
    orbweaver_compact_model: str = "claude-haiku-4-5"
    compact_micro_keep: int = 5
    compact_tool_result_chars: int = 8000
    compact_max_failures: int = 3
    compact_rehydrate_files: int = 5
    compact_rehydrate_chars_per_file: int = 20000
    compact_rehydrate_token_budget: int = 50000
    compact_notes_max_chars: int = 12000
    database_url: str = "postgresql://orbweaver:orbweaver@localhost:5432/orbweaver"
    orbweaver_store: str = "memory"  # memory | postgres
    workspace_root: str = "."
    telegram_bot_token: str = ""
    telegram_allowlist: str = ""
    orbweaver_image_api_key: str = ""
    orbweaver_image_api_url: str = "https://api.openai.com/v1/images/generations"
    orbweaver_image_model: str = "dall-e-3"
    orbweaver_image_size: str = "1024x1024"
    orbweaver_search_provider: str = ""  # duckduckgo | brave | empty auto
    orbweaver_brave_api_key: str = ""
    brave_search_api_key: str = ""
    orbweaver_host: str = "0.0.0.0"
    orbweaver_port: int = 8080
    orbweaver_allow_http_mint: bool = False
    embedding_dim: int = 384
    embedding_model: str = "hash://blake2b-384"  # sentence-transformers name when using embed extra

    @property
    def telegram_user_ids(self) -> set[int]:
        if not self.telegram_allowlist.strip():
            return set()
        return {int(x.strip()) for x in self.telegram_allowlist.split(",") if x.strip()}

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
        )
        return max(256, raw)


settings = Settings()
