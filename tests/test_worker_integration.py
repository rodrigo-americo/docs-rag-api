import asyncio
import contextlib
import uuid

import pytest

from app.core.config import settings
from app.core.queue import RedisQueueClient, get_redis_client
from app.models.document import Document, DocumentStatus
from app.services.embedding import get_embedding_provider
from app.services.ingest import IngestService
from app.worker import run
from tests.conftest import get_test_session_factory, wait_until

# Sessões abertas diretamente aqui (fora do client HTTP) precisam do mesmo
# engine NullPool que o resto da suite usa — não app.core.db.AsyncSessionLocal
# (produção), que fica preso ao event loop do primeiro teste que o usar e
# quebra com "Event loop is closed" no teste seguinte (loop novo por teste).
TestSessionLocal = get_test_session_factory()

pytestmark = pytest.mark.skipif(
    settings.queue_backend != "redis",
    reason="requer QUEUE_BACKEND=redis com um Redis real — não roda em modo inline",
)


@pytest.fixture
async def worker_task():
    """Sobe o worker de verdade (o mesmo app.worker.run usado em produção)
    como uma task em paralelo, rodando contra o Redis real da suite — não
    é um fake nem um mock, é o código de produção mesmo. Cancela ao fim do
    teste: o cancelamento propaga pro `finally` de run(), que fecha a
    conexão Redis e o engine do worker de forma limpa (mesmo caminho que
    um `docker stop` exercitaria em produção).
    """
    task = asyncio.create_task(run())
    yield task
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


async def test_message_travels_through_real_redis_to_a_real_worker(worker_task, sample_txt_bytes):
    """Prova que a mensageria é real: a API (aqui, IngestService.ingest
    direto, que é exatamente o que o endpoint chama) enfileira via
    RedisQueueClient — sem atalho de BackgroundTasks/InlineQueueClient —,
    e um worker rodando à parte consome do Redis e processa. Sem isso,
    'issue de mensageria' seria só uma alegação, não algo verificado.
    """
    async with TestSessionLocal() as session:
        service = IngestService(session=session, embedding_provider=get_embedding_provider())
        result = await service.ingest(content=sample_txt_bytes, filename="worker_integration.txt")

    redis_client = get_redis_client()
    try:
        queue_client = RedisQueueClient(redis_client)
        await queue_client.enqueue(result.document_id)

        async def _is_done() -> bool:
            async with TestSessionLocal() as session:
                doc = await session.get(Document, result.document_id)
                # PENDING e PROCESSING são estados de trânsito — espera até
                # um estado terminal, senão a condição vira verdadeira cedo
                # demais (assim que o worker seta PROCESSING, mas ainda não
                # terminou o resto do trabalho).
                return doc.status in (DocumentStatus.INDEXED, DocumentStatus.FAILED)

        await wait_until(_is_done, timeout=10)
    finally:
        await redis_client.aclose()

    async with TestSessionLocal() as session:
        doc = await session.get(Document, result.document_id)
        assert doc.status == DocumentStatus.INDEXED


async def test_document_that_always_fails_is_retried_then_marked_failed(worker_task):
    """Mesmo teste de fila real, mas pelo caminho de falha: prova que o
    retry_count sobe a cada tentativa e que o worker desiste definitivamente
    ao esgotar settings.ingest_max_retries, em vez de reenfileirar pra
    sempre um documento que nunca vai processar com sucesso.
    """
    unique_content = f"conteudo invalido {uuid.uuid4()}".encode()
    async with TestSessionLocal() as session:
        service = IngestService(session=session, embedding_provider=get_embedding_provider())
        result = await service.ingest(content=unique_content, filename="sempre_falha.exe")

    redis_client = get_redis_client()
    try:
        queue_client = RedisQueueClient(redis_client)
        await queue_client.enqueue(result.document_id)

        async def _is_terminal() -> bool:
            async with TestSessionLocal() as session:
                doc = await session.get(Document, result.document_id)
                return doc.retry_count >= settings.ingest_max_retries

        await wait_until(_is_terminal, timeout=10)
    finally:
        await redis_client.aclose()

    async with TestSessionLocal() as session:
        doc = await session.get(Document, result.document_id)
        assert doc.status == DocumentStatus.FAILED
        assert doc.retry_count == settings.ingest_max_retries
        assert doc.last_error is not None
