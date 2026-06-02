# app/api/documents.py
"""Endpoint /documents — thin, delega tudo pra IngestService."""
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.db import get_db_session
from app.core.exceptions import (
    DocumentTooLargeError,
    DocumentTooManyPagesError,
    UnsupportedFileTypeError,
)
from app.schemas.documents import IngestResponse
from app.services.embedding import EmbeddingProvider, get_embedding_provider
from app.services.ingest import IngestService
from app.services.pdf_parser import PdfParseError
from app.core.logging import get_logger

router = APIRouter(prefix="/documents", tags=["documents"])

log = get_logger(__name__)

def get_embedding() -> EmbeddingProvider:
    """Dependency pra injetar o embedding provider — facilita override em testes."""
    return get_embedding_provider()


@router.post(
    "/ingest",
    response_model=IngestResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Ingere um arquivo PDF, MD ou TXT",
)
async def ingest_document(
    file: UploadFile = File(...),
    title: str | None = Form(None),
    session: AsyncSession = Depends(get_db_session),
    embedding: EmbeddingProvider = Depends(get_embedding),
) -> IngestResponse:
    # 1. Lê conteúdo com limite estrito — protege contra DoS de upload grande.
    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    content = await _read_with_limit(file, max_bytes)

    # 2. Delega tudo pra service. Endpoint só traduz exceções de domínio em HTTP.
    service = IngestService(session=session, embedding_provider=embedding)
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
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=str(exc),
        ) from exc
    except DocumentTooManyPagesError as exc:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=str(exc),
        ) from exc
    except UnsupportedFileTypeError as exc:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=str(exc),
        ) from exc
    except PdfParseError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    return IngestResponse(
        document_id=result.document_id,
        title=result.title,
        chunks_created=result.chunks_created,
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
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"Arquivo excede limite de {max_bytes} bytes",
            )
    return bytes(buf)
