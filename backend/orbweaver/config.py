from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    anthropic_api_key: str = ""
    orbweaver_jwt_secret: str = "dev-secret-change-me"
    orbweaver_model: str = "claude-sonnet-4-6"
    orbweaver_haiku_model: str = "claude-haiku-4-5"
    orbweaver_pinned_token_cap: int = 4000
    context_window: int = 200_000
    output_reserve: int = 16_000
    static_token_estimate: int = 12_000
    compact_ratio: float = 0.85
    database_url: str = "postgresql://orbweaver:orbweaver@localhost:5432/orbweaver"
    orbweaver_store: str = "memory"  # memory | postgres
    workspace_root: str = "."
    telegram_bot_token: str = ""
    telegram_allowlist: str = ""
    orbweaver_host: str = "0.0.0.0"
    orbweaver_port: int = 8080
    embedding_dim: int = 384
    embedding_model: str = "hash://blake2b-384"  # sentence-transformers name when using embed extra

    @property
    def telegram_user_ids(self) -> set[int]:
        if not self.telegram_allowlist.strip():
            return set()
        return {int(x.strip()) for x in self.telegram_allowlist.split(",") if x.strip()}

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
