import asyncio
import time
from collections.abc import Awaitable, Callable

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from app.api.documents import get_embedding as get_embedding_for_ingest
from app.api.documents import get_session_factory
from app.api.query import get_chat, get_embedding
from app.core.config import settings
from app.core.db import get_db_session
from app.main import app
from app.services.chat import FakeChatProvider, OpenAIChatProvider
from app.services.embedding import FakeEmbeddingProvider, OpenAIEmbeddingProvider

# A suite faz TRUNCATE a cada teste (_clean_tables abaixo) — rodar isso
# contra settings.database_url apagaria dados de dev/produção sem aviso.
# TEST_DATABASE_URL é obrigatória e precisa apontar pra um banco diferente;
# falha alto aqui em vez de silenciosamente truncar o banco errado.
if not settings.test_database_url:
    raise RuntimeError(
        "TEST_DATABASE_URL não configurada — necessária pra rodar a suite de "
        "teste sem apagar o banco de dev (que TRUNCATE a cada teste tocaria "
        "se caísse em settings.database_url). Ver docs/testes.md."
    )
if settings.test_database_url == settings.database_url:
    raise RuntimeError(
        "TEST_DATABASE_URL é igual a DATABASE_URL — a suite de teste faz "
        "TRUNCATE a cada teste e apagaria o banco de dev. Aponte "
        "TEST_DATABASE_URL para um banco separado (ver docs/testes.md)."
    )

# NullPool: sem conexões persistentes entre requests. Cada operação abre e
# fecha sua própria conexão dentro do event loop atual — evita o clássico
# "Future attached to a different loop" que aparece quando um pool global
# (criado em app.core.db, no import-time) sobrevive entre loops de teste.
_test_engine = create_async_engine(settings.test_database_url, poolclass=NullPool)
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

# Sempre fake em teste, independente do que estiver no .env local
# (CHAT_PROVIDER=openai/EMBEDDING_PROVIDER=openai) — evita custo real de
# OpenAI e conexões de rede penduradas a cada rodada da suite. Testes que
# precisarem do provider real de verdade (ex: contrato de resposta da API)
# fazem override pontual com uma fixture própria.
# get_embedding é declarado separadamente em app.api.query e app.api.documents
# (um Depends por router) — os dois precisam de override, não são a mesma função.
app.dependency_overrides[get_chat] = lambda: FakeChatProvider()
app.dependency_overrides[get_embedding] = lambda: FakeEmbeddingProvider()
app.dependency_overrides[get_embedding_for_ingest] = lambda: FakeEmbeddingProvider()


@pytest_asyncio.fixture(scope="session", autouse=True)
async def _dispose_test_engine():
    """Descarta _test_engine ao fim da suite inteira.

    Sem isso, as conexões TCP que o asyncpg abre por trás do NullPool
    (uma por operação, nunca reaproveitada — daí NullPool) não são
    fechadas de forma determinística: sobrevivem até o garbage collector
    do Python rodar, o que pode acontecer só depois do pytest já estar no
    próprio processo de shutdown. filterwarnings=["error"] (pyproject.toml)
    promove o ResourceWarning resultante a erro fatal nesse momento,
    derrubando a suite inteira de forma intermitente mesmo com todo teste
    tendo passado — reproduzido tanto em Windows quanto em Linux (dentro
    do container `test`, ver docs/testes.md), não é peculiaridade de SO.
    """
    yield
    await _test_engine.dispose()


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


@pytest.fixture(scope="module")
def vcr_config() -> dict:
    """Config do pytest-recording (vcrpy) pra testes marcados com @pytest.mark.vcr.

    filter_headers evita gravar a API key real dentro do cassete — sem isso,
    tests/cassettes/*.yaml (versionado no repo, embora ignorado via
    .gitignore aqui) carregaria o segredo em texto puro.
    """
    return {
        "filter_headers": [("authorization", "REDACTED"), ("openai-organization", "REDACTED")],
        "record_mode": "once",
    }


@pytest.fixture
def real_embedding_provider() -> OpenAIEmbeddingProvider:
    """Provider real de embedding, só para testes @pytest.mark.vcr — cobre o
    contrato de resposta de verdade da API da OpenAI (ver
    docs/roadmap.md, trilha 'infraestrutura de teste').

    api_key usa um fallback fake quando settings.openai_api_key está vazia
    (ex: CI, que deliberadamente roda com OPENAI_API_KEY="" pra nunca gastar
    de verdade) — o __init__ do provider só valida que a chave não é vazia,
    e a chamada HTTP real nunca sai porque o VCR intercepta e reproduz o
    cassete já gravado antes dela acontecer."""
    return OpenAIEmbeddingProvider(
        api_key=settings.openai_api_key or "sk-fake-key-for-vcr-replay",
        model=settings.openai_embedding_model,
    )


@pytest.fixture
def real_chat_provider() -> OpenAIChatProvider:
    """Provider real de chat, só para testes @pytest.mark.vcr — mesmo
    raciocínio de real_embedding_provider."""
    return OpenAIChatProvider(
        api_key=settings.openai_api_key or "sk-fake-key-for-vcr-replay",
        model=settings.openai_chat_model,
    )


@pytest.fixture
def sample_txt_bytes() -> bytes:
    return (
        "O prazo de entrega acordado é de 10 dias úteis a partir da assinatura "
        "do contrato. Qualquer atraso deve ser comunicado com antecedência "
        "mínima de 48 horas.\n\n"
        "O pagamento deve ser realizado em até 5 dias úteis após a entrega."
    ).encode()
