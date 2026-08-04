from typing import Literal

from fastapi import APIRouter, Depends, Request, Response, status
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from app.core.config import settings
from app.core.db import engine
from app.core.logging import get_logger
from app.core.metrics import ingest_queue_depth, registry
from app.core.queue import queue_depths

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


@router.get(
    "/metrics",
    summary="Métricas Prometheus",
    description=(
        "Expõe fila (profundidade pending/processing), latência de ingest "
        "(histograma) e contagem de sucesso/falha, no formato de exposição "
        "do Prometheus. Sem autenticação — mesma postura de /health e /ready, "
        "assume rede interna não exposta publicamente."
    ),
)
async def metrics(request: Request) -> Response:
    # Gauge de fila é "puxado" aqui, não mantido atualizado em background:
    # profundidade da lista Redis é estado atual, não um evento que a
    # aplicação observa acontecer (diferente de latência/contagem, que são
    # incrementados em app/services/ingest.py na hora que o processamento
    # termina). Em queue_backend="inline" não há Redis — fica de fora do
    # scrape em vez de reportar um valor sem sentido.
    if settings.queue_backend == "redis" and request.app.state.redis_client is not None:
        depths = await queue_depths(request.app.state.redis_client)
        for queue_name, depth in depths.items():
            ingest_queue_depth.labels(queue=queue_name).set(depth)

    return Response(content=generate_latest(registry), media_type=CONTENT_TYPE_LATEST)
