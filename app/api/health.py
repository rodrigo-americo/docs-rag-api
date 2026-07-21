from typing import Literal

from fastapi import APIRouter, Depends, Response, status
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from app.core.db import engine
from app.core.logging import get_logger

router = APIRouter(tags=["health"])

log = get_logger(__name__)


class HealthResponse(BaseModel):
    status: Literal["ok"]


class ReadyResponse(BaseModel):
    status: Literal["ok", "degraded"]
    checks: dict[str, str]


def get_engine() -> AsyncEngine:
    """Dependency pra injetar engine — facilita override em testes."""
    return engine


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness probe",
    description=(
        "Retorna 200 se o processo está respondendo HTTP. "
        "Sem dependências externas — usado como liveness probe."
    ),
)
async def health() -> HealthResponse:
    return HealthResponse(status="ok")


@router.get(
    "/ready",
    response_model=ReadyResponse,
    summary="Readiness probe",
    description=(
        "Retorna 200 se o serviço pode atender requests do caminho crítico. "
        "Checa banco. OpenAI fica de fora — falha de OpenAI degrada features "
        "específicas, não tira a API do rotation."
    ),
)
async def ready(
    response: Response,
    engine: AsyncEngine = Depends(get_engine),
) -> ReadyResponse:
    checks: dict[str, str] = {}

    # --- Banco ---
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        checks["database"] = "ok"
    except Exception as exc:
        # Capturamos ampla porque qualquer falha aqui é "banco indisponível"
        # do ponto de vista do probe. Resposta HTTP só leva o nome da classe
        # (evita vazar detalhes de conexão); mensagem completa vai pro log.
        log.warning("ready.database_check_failed", error=str(exc))
        checks["database"] = f"fail: {type(exc).__name__}"

    all_ok = all(v == "ok" for v in checks.values())
    if not all_ok:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return ReadyResponse(status="degraded", checks=checks)

    return ReadyResponse(status="ok", checks=checks)
