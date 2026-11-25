"""Application configuration loaded from environment variables and ``.env``.

No credential has a default value; secrets are held as ``SecretStr`` so they do
not appear in ``repr()`` output or accidental log lines.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

#: Default model per provider (see docs/LLM_RESEARCH.md for the selection rationale).
DEFAULT_MODELS: dict[str, str] = {"gemini": "gemini-3.8-flash"}


class Settings(BaseSettings):
    """Typed runtime configuration."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", case_sensitive=False
    )

    # --- application ------------------------------------------------------
    app_name: str = "AI Website Link Analysis API"
    app_env: str = "development"
    port: int = Field(default=4000, ge=1, le=65535)
    log_level: str = "INFO"
    trust_proxy_headers: bool = False  # honour X-Forwarded-For (set true behind Render)
    rate_limit_per_minute: int = Field(default=10, ge=0)  # 0 disables the limiter
    max_request_bytes: int = Field(default=8192, ge=256)

    # --- LLM --------------------------------------------------------------
    llm_provider: str = "gemini"
    llm_model: str = ""  # blank -> DEFAULT_MODELS[llm_provider]
    gemini_api_key: SecretStr | None = None
    llm_timeout: float = Field(default=90.0, gt=0, le=300)
    llm_max_retries: int = Field(default=2, ge=0, le=5)  # retries after the first attempt
    llm_max_output_tokens: int = Field(default=8192, ge=1024, le=65536)
    llm_thinking_level: str = "low"  # low | medium | high; blank omits the parameter
    llm_temperature: float | None = Field(default=None, ge=0, le=2)  # None -> provider default
    llm_prompt_max_chars: int = Field(default=24000, ge=2000)  # cleaned page text sent to the LLM
    llm_input_price_per_mtok: float | None = Field(default=None, ge=0)
    llm_output_price_per_mtok: float | None = Field(default=None, ge=0)

    # --- competitor research (optional) ----------------------------------
    tavily_api_key: SecretStr | None = None
    research_timeout: float = Field(default=15.0, gt=0, le=60)
    research_max_results: int = Field(default=5, ge=1, le=10)

    # --- website extraction ----------------------------------------------
    request_timeout: float = Field(default=20.0, gt=0, le=120)  # overall fetch budget (seconds)
    max_content_length: int = Field(default=50000, ge=1000)  # max extracted text (characters)
    max_redirects: int = Field(default=3, ge=0, le=10)
    max_response_bytes: int = Field(default=3_000_000, ge=10_000)

    # --- caching ----------------------------------------------------------
    cache_ttl_seconds: int = Field(default=900, ge=0)  # 0 disables the in-memory cache
    cache_max_entries: int = Field(default=128, ge=1)

    @field_validator("gemini_api_key", "tavily_api_key", mode="before")
    @classmethod
    def _blank_secret_is_none(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator(
        "llm_temperature", "llm_input_price_per_mtok", "llm_output_price_per_mtok", mode="before"
    )
    @classmethod
    def _blank_number_is_none(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("llm_provider", "llm_thinking_level", mode="after")
    @classmethod
    def _normalize_choice(cls, value: str) -> str:
        return value.strip().lower()

    # ------------------------------------------------------------------ #
    @property
    def effective_llm_model(self) -> str:
        """Configured model, or the provider's default."""
        return self.llm_model.strip() or DEFAULT_MODELS.get(self.llm_provider, "")

    @property
    def has_llm_key(self) -> bool:
        return self.llm_provider == "gemini" and self.gemini_api_key is not None

    @property
    def has_research_key(self) -> bool:
        return self.tavily_api_key is not None

    def secret_values(self) -> list[str]:
        """Non-empty secret values, used by the log redaction filter."""
        secrets = (self.gemini_api_key, self.tavily_api_key)
        return [s.get_secret_value() for s in secrets if s is not None and s.get_secret_value()]


@lru_cache
def get_settings() -> Settings:
    """Cached settings accessor (override in tests via ``cache_clear``)."""
    return Settings()
