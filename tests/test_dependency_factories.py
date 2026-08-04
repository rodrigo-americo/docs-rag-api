"""Testa as dependency functions triviais (get_chat/get_embedding/
get_session_factory) diretamente, contornando os dependency_overrides que
tests/conftest.py sempre aplica no client HTTP — elas nunca são exercitadas
pela suite normal porque sempre são sobrescritas antes de qualquer request."""

from app.api.documents import get_embedding as get_ingest_embedding
from app.api.documents import get_session_factory
from app.api.query import get_chat
from app.api.query import get_embedding as get_query_embedding
from app.core.config import settings
from app.core.db import AsyncSessionLocal
from app.services.chat import FakeChatProvider
from app.services.embedding import FakeEmbeddingProvider


def test_documents_get_embedding_returns_configured_provider(monkeypatch):
    # embedding_provider/chat_provider refletem o .env local (pode ser
    # "openai" se alguém tiver configurado pra gravar cassetes VCR) —
    # fixa "fake" explicitamente pra este teste não depender disso.
    monkeypatch.setattr(settings, "embedding_provider", "fake")
    assert isinstance(get_ingest_embedding(), FakeEmbeddingProvider)


def test_documents_get_session_factory_returns_production_session_local():
    assert get_session_factory() is AsyncSessionLocal


def test_query_get_chat_returns_configured_provider(monkeypatch):
    monkeypatch.setattr(settings, "chat_provider", "fake")
    assert isinstance(get_chat(), FakeChatProvider)


def test_query_get_embedding_returns_configured_provider(monkeypatch):
    monkeypatch.setattr(settings, "embedding_provider", "fake")
    assert isinstance(get_query_embedding(), FakeEmbeddingProvider)
