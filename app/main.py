from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from fastapi import FastAPI

from app.api import health
from app.core.config import settings
from app.core.db import engine


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Gerencia recursos pelo ciclo de vida da aplicação.

    Startup: nada agora (engine é instanciado no import).
    Shutdown: disposa o pool de conexões do engine.
    """
    yield
    # Shutdown: fecha o pool de conexões. Sem isso, em SIGTERM
    # rápido o pool não devolve conexões, e o banco vê elas como
    # "idle in transaction" até o timeout do Postgres.
    await engine.dispose()


app = FastAPI(
    title="docs-rag-api",
    description="API REST de RAG sobre PDF/Markdown com pgvector e LangGraph",
    version=settings.app_version,
    lifespan=lifespan,
)

app.include_router(health.router)