import logging
import os

from app.core.config import settings
from app.core.logging import configure_logging
from app.core.tracing import configure_tracing


def test_configure_tracing_noop_when_disabled(monkeypatch):
    monkeypatch.setattr(settings, "langsmith_tracing", False)
    monkeypatch.delenv("LANGCHAIN_TRACING_V2", raising=False)

    configure_tracing()

    assert "LANGCHAIN_TRACING_V2" not in os.environ


def test_configure_tracing_sets_env_vars_when_enabled(monkeypatch):
    monkeypatch.setattr(settings, "langsmith_tracing", True)
    monkeypatch.setattr(settings, "langsmith_api_key", "ls-fake-key")
    monkeypatch.setattr(settings, "langsmith_project", "test-project")
    monkeypatch.delenv("LANGCHAIN_TRACING_V2", raising=False)

    configure_tracing()

    assert os.environ["LANGCHAIN_TRACING_V2"] == "true"
    assert os.environ["LANGCHAIN_API_KEY"] == "ls-fake-key"
    assert os.environ["LANGCHAIN_PROJECT"] == "test-project"


def test_configure_logging_with_json_format_configures_root_logger(monkeypatch):
    """log_format="json" (branch nunca exercitado — .env local usa
    "console", CI usa "json" mas nunca os dois na mesma suite). Restaura
    os handlers originais do root logger no fim: configure_logging() faz
    root_logger.handlers.clear() + adiciona um StreamHandler novo — sem
    restaurar, isso quebraria capsys de testes que rodarem depois na
    mesma suite (mesmo mecanismo que corrompeu test_query_endpoint.py
    quando lifespan() chamava configure_logging() sem mock, ver
    test_main_lifespan.py)."""
    monkeypatch.setattr(settings, "log_format", "json")
    root_logger = logging.getLogger()
    original_handlers = list(root_logger.handlers)
    try:
        configure_logging()
        assert len(root_logger.handlers) == 1
    finally:
        root_logger.handlers.clear()
        root_logger.handlers.extend(original_handlers)
