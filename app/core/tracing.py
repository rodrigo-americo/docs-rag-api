import os

from app.core.config import settings


def configure_tracing() -> None:
    """Traduz Settings (Pydantic) para as env vars que o LangChain lê
    diretamente do os.environ — LANGCHAIN_TRACING_V2 etc. não existem
    como conceito do Settings, então isso é só uma ponte."""
    if not settings.langsmith_tracing:
        return
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key
    os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project
