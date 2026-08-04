"""Testa app.core.db.get_db_session diretamente — tests/conftest.py sempre
sobrescreve essa dependency com _override_get_db_session (que usa o engine
de teste), então o código real de get_db_session nunca é exercitado pelo
resto da suite. Mocka AsyncSessionLocal em vez de usar o banco de dev real
(settings.database_url) que get_db_session usa por padrão."""

from unittest.mock import AsyncMock, MagicMock

import pytest

import app.core.db as db_module
from app.core.db import get_db_session


def _mock_session_factory():
    """AsyncSessionLocal() é um async context manager — MagicMock com
    __aenter__/__aexit__ configurados manualmente, já que
    unittest.mock.AsyncMock não infere isso automaticamente pra chamadas
    encadeadas (AsyncSessionLocal() precisa ser síncrono, só o __aenter__
    resultante é assíncrono)."""
    session = AsyncMock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()

    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=session)
    cm.__aexit__ = AsyncMock(return_value=False)

    factory = MagicMock(return_value=cm)
    return factory, session


async def test_get_db_session_commits_on_success(monkeypatch):
    factory, session = _mock_session_factory()
    monkeypatch.setattr(db_module, "AsyncSessionLocal", factory)

    gen = get_db_session()
    yielded_session = await anext(gen)
    assert yielded_session is session

    with pytest.raises(StopAsyncIteration):
        await anext(gen)

    session.commit.assert_awaited_once()
    session.rollback.assert_not_awaited()


async def test_get_db_session_rolls_back_and_reraises_on_exception(monkeypatch):
    factory, session = _mock_session_factory()
    monkeypatch.setattr(db_module, "AsyncSessionLocal", factory)

    gen = get_db_session()
    await anext(gen)

    with pytest.raises(ValueError, match="erro do request"):
        await gen.athrow(ValueError("erro do request"))

    session.rollback.assert_awaited_once()
    session.commit.assert_not_awaited()
