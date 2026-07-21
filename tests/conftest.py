import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from app.core.config import settings
from app.core.db import get_db_session
from app.main import app

# NullPool: sem conexões persistentes entre requests. Cada operação abre e
# fecha sua própria conexão dentro do event loop atual — evita o clássico
# "Future attached to a different loop" que aparece quando um pool global
# (criado em app.core.db, no import-time) sobrevive entre loops de teste.
_test_engine = create_async_engine(settings.database_url, poolclass=NullPool)
_TestSessionLocal = async_sessionmaker(bind=_test_engine, expire_on_commit=False, autoflush=False)


async def _override_get_db_session():
    async with _TestSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


app.dependency_overrides[get_db_session] = _override_get_db_session


@pytest_asyncio.fixture(autouse=True)
async def _clean_tables():
    """Limpa documents/chunks antes de cada teste — isolamento entre testes
    que rodam contra o Postgres real (não há banco só de testes ainda)."""
    async with _test_engine.begin() as conn:
        await conn.execute(text("TRUNCATE TABLE chunks, documents CASCADE"))
    yield


@pytest_asyncio.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.fixture
def sample_txt_bytes() -> bytes:
    return (
        "O prazo de entrega acordado é de 10 dias úteis a partir da assinatura "
        "do contrato. Qualquer atraso deve ser comunicado com antecedência "
        "mínima de 48 horas.\n\n"
        "O pagamento deve ser realizado em até 5 dias úteis após a entrega."
    ).encode()
