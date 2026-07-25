# app/main.py
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from app.api import documents, health, query
from app.core.config import settings
from app.core.db import engine
from app.core.logging import configure_logging, get_logger
from app.core.rate_limit import limiter


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

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

app.include_router(health.router)
app.include_router(documents.router)
app.include_router(query.router)
