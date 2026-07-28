import asyncio
import time
from collections.abc import Awaitable, Callable

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from app.api.documents import get_session_factory
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

# process_document (background task) precisa da mesma proteção contra
# "Event loop is closed" que _override_get_db_session já dá ao fluxo normal
# de request — usa o engine de teste (NullPool), não o de produção.
app.dependency_overrides[get_session_factory] = lambda: _TestSessionLocal

# ASGITransport faz todas as requisições de teste compartilharem o mesmo IP
# (get_remote_address), então o rate limit por IP derrubaria a suite se
# ficasse ligado aqui — desliga só no processo de teste.
app.state.limiter.enabled = False


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


def get_test_session_factory():
    """Expõe a session factory de teste (NullPool) pra quem precisar abrir
    sessões fora do ciclo normal de dependency injection — ex: testes que
    chamam IngestService/consultam o banco diretamente, sem passar pelo
    client HTTP. Usar AsyncSessionLocal (produção) nesses casos reproduziria
    o mesmo "Event loop is closed" que esta engine de teste existe pra
    evitar."""
    return _TestSessionLocal


async def wait_until(
    condition: Callable[[], Awaitable[bool]],
    timeout: float = 5.0,
    interval: float = 0.1,
) -> None:
    """Espera `condition()` virar True, checando a cada `interval` segundos.

    Necessário pra testes que exercitam um worker de verdade, rodando fora
    do processo de teste: diferente do ASGITransport (que roda BackgroundTasks
    de forma síncrona, antes do POST retornar), um worker real consumindo de
    um Redis real processa de forma genuinamente assíncrona — não tem
    atalho, o teste precisa checar repetidamente até o estado mudar ou
    desistir depois de `timeout` segundos.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if await condition():
            return
        await asyncio.sleep(interval)
    raise TimeoutError(f"condição não satisfeita após {timeout}s")


@pytest.fixture
def sample_txt_bytes() -> bytes:
    return (
        "O prazo de entrega acordado é de 10 dias úteis a partir da assinatura "
        "do contrato. Qualquer atraso deve ser comunicado com antecedência "
        "mínima de 48 horas.\n\n"
        "O pagamento deve ser realizado em até 5 dias úteis após a entrega."
    ).encode()
