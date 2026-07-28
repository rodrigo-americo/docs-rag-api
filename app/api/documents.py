import uuid
from collections.abc import Callable

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
    status,
)
from redis.exceptions import RedisError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.db import AsyncSessionLocal, get_db_session
from app.core.exceptions import DocumentTooLargeError, IngestError
from app.core.logging import get_logger
from app.core.queue import InlineQueueClient, QueueClient, RedisQueueClient
from app.core.rate_limit import limiter
from app.models.document import Document
from app.schemas.documents import DocumentSummary, IngestResponse
from app.services.embedding import EmbeddingProvider, get_embedding_provider
from app.services.ingest import IngestService

router = APIRouter(prefix="/documents", tags=["documents"])

log = get_logger(__name__)


def get_embedding() -> EmbeddingProvider:
    """Dependency pra injetar o embedding provider — facilita override em testes."""
    return get_embedding_provider()


def get_session_factory() -> Callable[[], AsyncSession]:
    """Dependency pra injetar a factory de sessão usada em background tasks
    (process_document abre sua própria sessão, fora do ciclo do request).
    Sobrescrita em teste pra apontar pro engine de teste, evitando que a
    background task use o pool de conexões de produção preso a um event
    loop de teste diferente do atual."""
    return AsyncSessionLocal


@router.get(
    "",
    response_model=list[DocumentSummary],
    summary="Lista de documentos indexados",
)
async def list_documents(
    session: AsyncSession = Depends(get_db_session),
) -> list[DocumentSummary]:
    stmt = select(Document).order_by(Document.created_at.desc())
    result = await session.execute(stmt)
    documents = result.scalars().all()

    return [
        DocumentSummary(
            id=doc.id,
            title=doc.title,
            source_filename=doc.source_filename,
            status=doc.status,
            created_at=doc.created_at,
        )
        for doc in documents
    ]


@router.get(
    "/{document_id}",
    response_model=DocumentSummary,
    summary="Consulta um documento e seu status de processamento",
)
async def get_document(
    document_id: uuid.UUID,
    session: AsyncSession = Depends(get_db_session),
) -> DocumentSummary:
    document = await session.get(Document, document_id)
    if document is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Documento {document_id} não encontrado",
        )

    return DocumentSummary(
        id=document.id,
        title=document.title,
        source_filename=document.source_filename,
        status=document.status,
        created_at=document.created_at,
    )


@router.delete(
    "/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Remove um documento e seus chunks",
)
async def delete_document(
    document_id: uuid.UUID,
    session: AsyncSession = Depends(get_db_session),
) -> None:
    document = await session.get(Document, document_id)
    if document is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Documento {document_id} não encontrado",
        )

    await session.delete(document)


@router.post(
    "/ingest",
    response_model=IngestResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Ingere um arquivo PDF, MD ou TXT",
)
@limiter.limit(settings.rate_limit_ingest)
async def ingest_document(
    request: Request,
    file: UploadFile = File(...),
    title: str | None = Form(None),
    session: AsyncSession = Depends(get_db_session),
    embedding: EmbeddingProvider = Depends(get_embedding),
    session_factory: Callable[[], AsyncSession] = Depends(get_session_factory),
) -> IngestResponse:
    # 1. Lê conteúdo com limite estrito — protege contra DoS de upload grande.
    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    content = await _read_with_limit(file, max_bytes)

    # 2. Delega tudo pra service. Endpoint só traduz exceções de domínio em HTTP.
    service = IngestService(
        session=session, embedding_provider=embedding, session_factory=session_factory
    )
    try:
        result = await service.ingest(
            content=content,
            filename=file.filename or "unknown",
            title=title,
        )
    except DocumentTooLargeError as exc:
        log.warning(
            "ingest.rejected",
            reason="too_large",
            size_bytes=exc.size_bytes,
            max_bytes=exc.max_bytes,
        )
        raise exc.to_http() from exc
    except IngestError as exc:
        # Catch-all: qualquer subclasse de IngestError já carrega seu próprio
        # status_code (ver app/core/exceptions.py). Novas exceções de ingest
        # não exigem tocar neste endpoint.
        raise exc.to_http() from exc

    # 3. Só enfileira o processamento pesado se for um documento novo — uma
    #    duplicata já foi (ou está sendo) processada, reenfileirar geraria
    #    chamada de embedding desperdiçada e cairia na unique constraint.
    if not result.is_duplicate:
        queue_client: QueueClient
        if settings.queue_backend == "redis":
            # Client de vida longa, criado uma vez no lifespan (app/main.py)
            # — não abrimos/fechamos conexão Redis a cada request. None
            # só acontece se o lifespan não rodou (ex: ASGITransport de
            # teste) — tratado como fila indisponível, mesmo caso do
            # RedisError abaixo.
            redis_client = request.app.state.redis_client
            if redis_client is None:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Fila de processamento indisponível — tente novamente em instantes",
                )
            queue_client = RedisQueueClient(redis_client)
            try:
                await queue_client.enqueue(result.document_id)
            except RedisError as exc:
                log.warning("ingest.enqueue_failed", document_id=str(result.document_id))
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Fila de processamento indisponível — tente novamente em instantes",
                ) from exc
        else:
            queue_client = InlineQueueClient(service)
            await queue_client.enqueue(result.document_id)

    return IngestResponse(
        document_id=result.document_id,
        title=result.title,
        chunks_created=result.chunks_created,
        status=result.status,
    )


async def _read_with_limit(file: UploadFile, max_bytes: int) -> bytes:
    """Lê o upload em chunks, abortando se passar do limite.

    Sem isso, um POST de 1GB ainda escreveria 1GB em disco temp (via
    python-multipart) antes de chegar no nosso código. Esta função aborta
    a leitura cedo.
    """
    CHUNK = 64 * 1024  # 64KB por leitura
    buf = bytearray()
    while True:
        chunk = await file.read(CHUNK)
        if not chunk:
            break
        buf.extend(chunk)
        if len(buf) > max_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                detail=f"Arquivo excede limite de {max_bytes} bytes",
            )
    return bytes(buf)
