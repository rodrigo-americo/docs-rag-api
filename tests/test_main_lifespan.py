"""Testa app.main.lifespan diretamente — ASGITransport (usado pelo resto da
suite via a fixture `client`) não dispara o lifespan do FastAPI, então essas
linhas nunca são exercitadas pelos testes de endpoint (ver comentário sobre
app.state.redis_client em app/main.py).

configure_logging()/configure_tracing() são mockados em todo teste aqui:
lifespan() os chama de verdade, e configure_logging() reconfigura o
logging global (logging.getLogger().handlers.clear() + StreamHandler novo
sobre o sys.stdout do momento da chamada) — chamado fora do startup real
da app, isso quebra a captura de log (capsys) dos testes que rodarem
depois na mesma suite, silenciosamente (sem erro, só perdendo a saída)."""

from unittest.mock import patch

from app.core.config import settings
from app.main import app, lifespan


async def test_lifespan_sets_redis_client_to_none_when_queue_backend_is_inline(monkeypatch):
    monkeypatch.setattr(settings, "queue_backend", "inline")

    with patch("app.main.configure_logging"), patch("app.main.configure_tracing"):
        async with lifespan(app):
            assert app.state.redis_client is None


async def test_lifespan_creates_redis_client_when_queue_backend_is_redis(monkeypatch):
    monkeypatch.setattr(settings, "queue_backend", "redis")
    monkeypatch.setattr(settings, "redis_url", "redis://localhost:6379/0")

    with patch("app.main.configure_logging"), patch("app.main.configure_tracing"):
        async with lifespan(app):
            assert app.state.redis_client is not None
