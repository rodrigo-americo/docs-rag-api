# app/main.py
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api import documents, health
from app.core.config import settings
from app.core.db import engine
from app.core.logging import configure_logging, get_logger


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Startup: configura logging ANTES de qualquer outra coisa.
    configure_logging()
    log = get_logger(__name__)
    log.info(
        "app.startup",
        version=settings.app_version,
        embedding_provider=settings.embedding_provider,
        log_format=settings.log_format,
    )
    yield
    # Shutdown
    log.info("app.shutdown")
    await engine.dispose()


app = FastAPI(
    title="docs-rag-api",
    description="API REST de RAG sobre PDF/Markdown com pgvector e LangGraph",
    version=settings.app_version,
    lifespan=lifespan,
)

app.include_router(health.router)
app.include_router(documents.router)