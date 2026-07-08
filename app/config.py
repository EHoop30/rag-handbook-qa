"""Runtime configuration, read from environment variables (or a .env file).

Every knob a deployment might change lives here so nothing is hardcoded in
the pipeline. Provider selection ("openai" vs "fake") is what lets the same
code run against real models in production and run offline in tests.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql://rag:rag@localhost:5432/rag"

    # "openai" | "fake"
    embedding_provider: str = "fake"
    openai_api_key: str | None = None
    openai_embedding_model: str = "text-embedding-3-small"

    # "anthropic" | "openai" | "fake"
    llm_provider: str = "fake"
    anthropic_api_key: str | None = None
    anthropic_model: str = "claude-haiku-4-5-20251001"
    openai_chat_model: str = "gpt-4o-mini"

    top_k: int = 4

    # Chunking (characters, not tokens; simple and deterministic for a demo)
    chunk_size: int = 900
    chunk_overlap: int = 150


@lru_cache
def get_settings() -> Settings:
    return Settings()
