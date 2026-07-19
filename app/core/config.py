from functools import lru_cache
from typing import Literal

from pydantic import Field, computed_field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    app_version: str = "0.1.0"
    database_url: str = Field(..., description="URL do Postgres com driver async (postgresql+asyncpg)")

    # --- OpenAI ---
    # Vazia por padrão: só é obrigatória quando embedding_provider="openai"
    # (validado abaixo). Rodar 100% em modo fake não deve exigir uma chave
    # que nunca será usada.
    openai_api_key: str = Field("", description="Chave da API OpenAI")
    openai_chat_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"

    # --- Provider switch ---
    # "fake" pra dev/teste sem custo. "openai" quando tiver chave.
    embedding_provider: Literal["openai", "fake"] = "fake"

    @model_validator(mode="after")
    def _require_openai_key_when_selected(self) -> "Settings":
        if self.embedding_provider == "openai" and not self.openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY é obrigatória quando EMBEDDING_PROVIDER=openai."
            )
        return self

    langsmith_tracing: bool = False
    langsmith_api_key: str = ""
    langsmith_project: str = "docs-rag-api"

    max_upload_size_mb: int = 10
    max_pdf_pages: int = 50
    chunk_size: int = 700
    chunk_overlap: int = 100

    retrieval_top_k: int = 5
    retrieval_quality_threshold: float = 0.7

    # --- Logging ---
    log_level: str = "INFO"
    log_format: Literal["json", "console"] = "console"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def database_url_sync(self) -> str:
        return self.database_url.replace("postgresql+asyncpg", "postgresql+psycopg")


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
