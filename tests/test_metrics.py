import uuid

import pytest

from app.core.config import settings
from app.core.metrics import ingest_processed_total, ingest_processing_seconds
from app.core.queue import ack, dequeue, enqueue, get_redis_client, queue_depths
from app.core.storage import delete_upload
from app.main import app
from app.models.document import DocumentStatus
from app.services.embedding import FakeEmbeddingProvider
from app.services.ingest import IngestService
from tests.conftest import get_test_session_factory

TestSessionLocal = get_test_session_factory()


def _counter_value(status: str) -> float:
    return ingest_processed_total.labels(status=status)._value.get()


async def test_process_document_success_increments_counter_and_observes_latency(
    sample_txt_bytes,
):
    before_ok = _counter_value(DocumentStatus.INDEXED)
    before_count = ingest_processing_seconds._sum.get()

    async with TestSessionLocal() as session:
        service = IngestService(session=session, embedding_provider=FakeEmbeddingProvider())
        result = await service.ingest(content=sample_txt_bytes, filename="metrics_ok.txt")

    service = IngestService(
        embedding_provider=FakeEmbeddingProvider(), session_factory=TestSessionLocal
    )
    status = await service.process_document(result.document_id)

    assert status == DocumentStatus.INDEXED
    assert _counter_value(DocumentStatus.INDEXED) == before_ok + 1
    # Não afirmamos um valor exato de latência (não-determinístico) — só que
    # o histograma registrou uma nova observação (soma mudou).
    assert ingest_processing_seconds._sum.get() > before_count


async def test_process_document_failure_increments_failed_counter(sample_txt_bytes):
    before_failed = _counter_value(DocumentStatus.FAILED)

    async with TestSessionLocal() as session:
        service = IngestService(session=session, embedding_provider=FakeEmbeddingProvider())
        result = await service.ingest(content=sample_txt_bytes, filename="metrics_fail.txt")

    delete_upload(result.document_id)

    service = IngestService(
        embedding_provider=FakeEmbeddingProvider(), session_factory=TestSessionLocal
    )
    status = await service.process_document(result.document_id)

    assert status == DocumentStatus.FAILED
    assert _counter_value(DocumentStatus.FAILED) == before_failed + 1


async def test_metrics_endpoint_exposes_prometheus_text_format(client):
    response = await client.get("/metrics")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    body = response.text
    assert "ingest_processed_total" in body
    assert "ingest_processing_seconds" in body


async def test_metrics_endpoint_skips_queue_gauge_when_redis_client_is_none(client):
    """app.state.redis_client é None sempre que o lifespan não roda
    (ASGITransport de teste) ou quando queue_backend="inline" em produção —
    o endpoint não deve tentar consultar Redis nem quebrar, só omite o
    gauge de fila do scrape. Força None explicitamente em vez de depender
    de settings.queue_backend (variável de ambiente externa à suite, que
    tests/test_worker_integration.py e o teste seguinte já mudam quando
    QUEUE_BACKEND=redis está setado no processo)."""
    original = app.state.redis_client
    app.state.redis_client = None
    try:
        response = await client.get("/metrics")
    finally:
        app.state.redis_client = original

    assert response.status_code == 200


@pytest.mark.skipif(
    settings.queue_backend != "redis",
    reason="requer QUEUE_BACKEND=redis com um Redis real — não roda em modo inline",
)
async def test_metrics_endpoint_populates_queue_gauge_when_redis_backend_is_active(client):
    """Sobrescreve app.state.redis_client direto (ASGITransport não dispara
    o lifespan de app/main.py, que é quem normalmente cria esse client) —
    mesma necessidade que app/api/documents.py já tem ao consumir esse
    state em ingest_document."""
    redis_client = get_redis_client()
    app.state.redis_client = redis_client
    try:
        response = await client.get("/metrics")
        assert response.status_code == 200
        body = response.text
        assert 'ingest_queue_depth{queue="pending"}' in body
        assert 'ingest_queue_depth{queue="processing"}' in body
    finally:
        app.state.redis_client = None
        await redis_client.aclose()


async def test_queue_depths_reads_pending_and_processing_lists():
    redis_client = get_redis_client()
    try:
        doc_id = uuid.uuid4()
        await enqueue(redis_client, doc_id)

        depths = await queue_depths(redis_client)
        assert depths["pending"] >= 1

        result = await dequeue(redis_client, timeout=1)
        assert result is not None
        _, raw_payload = result

        depths = await queue_depths(redis_client)
        assert depths["processing"] >= 1

        await ack(redis_client, raw_payload)
    finally:
        await redis_client.aclose()
