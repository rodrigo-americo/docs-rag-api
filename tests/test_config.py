import pytest
from pydantic import ValidationError

from app.core.config import Settings


def _base_kwargs(**overrides) -> dict:
    kwargs = {"database_url": "postgresql+asyncpg://user:pass@localhost:5432/db"}
    kwargs.update(overrides)
    return kwargs


def test_settings_requires_redis_url_when_queue_backend_is_redis():
    with pytest.raises(ValidationError, match="REDIS_URL"):
        Settings(**_base_kwargs(queue_backend="redis", redis_url=""))


def test_settings_allows_redis_backend_with_redis_url_set():
    settings = Settings(**_base_kwargs(queue_backend="redis", redis_url="redis://localhost:6379/0"))

    assert settings.queue_backend == "redis"


def test_settings_requires_openai_key_when_embedding_provider_is_openai():
    with pytest.raises(ValidationError, match="OPENAI_API_KEY"):
        Settings(**_base_kwargs(embedding_provider="openai", openai_api_key=""))


def test_settings_requires_openai_key_when_chat_provider_is_openai():
    with pytest.raises(ValidationError, match="OPENAI_API_KEY"):
        Settings(**_base_kwargs(chat_provider="openai", openai_api_key=""))


def test_settings_allows_openai_providers_with_key_set():
    settings = Settings(
        **_base_kwargs(embedding_provider="openai", chat_provider="openai", openai_api_key="sk-x")
    )

    assert settings.embedding_provider == "openai"
    assert settings.chat_provider == "openai"


def test_settings_database_url_sync_swaps_driver():
    settings = Settings(
        **_base_kwargs(database_url="postgresql+asyncpg://user:pass@localhost:5432/db")
    )

    assert settings.database_url_sync == "postgresql+psycopg://user:pass@localhost:5432/db"
