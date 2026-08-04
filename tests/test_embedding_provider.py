import pytest

from app.core.config import settings
from app.services.embedding import (
    FakeEmbeddingProvider,
    OpenAIEmbeddingProvider,
    get_embedding_provider,
)


def test_openai_embedding_provider_raises_without_api_key():
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        OpenAIEmbeddingProvider(api_key="", model="text-embedding-3-small")


def test_get_embedding_provider_returns_fake_when_provider_is_fake(monkeypatch):
    monkeypatch.setattr(settings, "embedding_provider", "fake")

    provider = get_embedding_provider()

    assert isinstance(provider, FakeEmbeddingProvider)


def test_get_embedding_provider_returns_openai_when_provider_is_openai(monkeypatch):
    # Mesmo raciocínio de test_chat_provider: __init__ só valida chave
    # não-vazia, sem chamada de rede.
    monkeypatch.setattr(settings, "embedding_provider", "openai")
    monkeypatch.setattr(settings, "openai_api_key", "sk-fake-key-for-unit-test")

    provider = get_embedding_provider()

    assert isinstance(provider, OpenAIEmbeddingProvider)
