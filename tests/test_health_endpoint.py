from sqlalchemy.ext.asyncio import create_async_engine

from app.api.health import get_engine
from app.main import app


async def test_health_returns_ok(client):
    response = await client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


async def test_ready_returns_ok_when_database_is_reachable(client):
    response = await client.get("/ready")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["checks"]["database"] == "ok"


async def test_ready_returns_degraded_when_database_is_unreachable(client):
    # Engine apontando pra uma porta que não existe — connect() falha rápido
    # (connection refused), sem depender de derrubar o Postgres de verdade
    # pra exercitar o branch de erro.
    broken_engine = create_async_engine(
        "postgresql+asyncpg://docsrag:docsrag@localhost:1/docsrag_nao_existe"
    )
    app.dependency_overrides[get_engine] = lambda: broken_engine
    try:
        response = await client.get("/ready")
    finally:
        del app.dependency_overrides[get_engine]
        await broken_engine.dispose()

    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "degraded"
    assert body["checks"]["database"].startswith("fail:")
