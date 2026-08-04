import asyncio
import uuid
from collections.abc import Callable

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import settings
from app.core.logging import configure_logging, get_logger
from app.core.queue import ack, dequeue, enqueue, get_redis_client
from app.core.storage import delete_upload
from app.models.document import DocumentStatus
from app.services.embedding import get_embedding_provider
from app.services.ingest import IngestService

log = get_logger(__name__)


async def run(session_factory: Callable[[], AsyncSession] | None = None) -> None:
    """`session_factory=None` (produção, `app.worker.__main__`): cria
    engine/session factory próprios a partir de `settings.database_url` —
    não reusa `app.core.db.AsyncSessionLocal`, criado no event loop de quem
    importou aquele módulo primeiro (poderia não ser o loop do
    `asyncio.run()` abaixo). Dono desse engine é esta função, por isso
    descarta no `finally`.

    `session_factory` explícito (só `tests/test_worker_integration.py`,
    que sobe este `run()` como task no mesmo processo/loop do pytest):
    injeta a session factory de teste (`TEST_DATABASE_URL`, não
    `settings.database_url`) — o worker de teste não é dono desse engine
    (o `conftest.py` é, e precisa dele vivo entre testes), então não o
    descarta.
    """
    configure_logging()
    log.info("worker.startup", queue=settings.ingest_queue_name)

    engine = None
    if session_factory is None:
        engine = create_async_engine(settings.database_url, pool_pre_ping=True)
        session_factory = async_sessionmaker(bind=engine, expire_on_commit=False, autoflush=False)

    service = IngestService(
        embedding_provider=get_embedding_provider(), session_factory=session_factory
    )
    redis_client = get_redis_client()

    try:
        while True:
            result = await dequeue(redis_client, timeout=5)
            if result is None:
                continue
            message, raw_payload = result
            document_id = uuid.UUID(message["document_id"])
            attempt = message["attempt"]

            status = await service.process_document(document_id)

            if status != DocumentStatus.FAILED:
                delete_upload(document_id)
                await ack(redis_client, raw_payload)
            elif attempt + 1 >= settings.ingest_max_retries:
                log.warning("worker.giving_up", document_id=str(document_id), attempt=attempt)
                delete_upload(document_id)
                await ack(redis_client, raw_payload)
            else:
                log.info("worker.retrying", document_id=str(document_id), next_attempt=attempt + 1)
                await enqueue(redis_client, document_id, attempt=attempt + 1)
                await ack(redis_client, raw_payload)
    finally:
        await redis_client.aclose()
        if engine is not None:
            await engine.dispose()


if __name__ == "__main__":
    asyncio.run(run())
