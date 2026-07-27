import hashlib
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from openai import OpenAIError
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.db import AsyncSessionLocal
from app.core.exceptions import (
    DocumentTooLargeError,
    DocumentTooManyPagesError,
    DuplicateChunkError,
    EmbeddingProviderError,
    EmptyDocumentError,
    SuspiciousContentError,
    UnsupportedFileTypeError,
)
from app.core.logging import get_logger
from app.models.document import Chunk, Document, DocumentStatus
from app.services.chunking import chunk_text
from app.services.embedding import EmbeddingProvider
from app.services.injection_detection import find_suspicious_pattern
from app.services.pdf_parser import parse_pdf


@dataclass
class IngestResult:
    document_id: uuid.UUID
    title: str
    chunks_created: int
    status: str
    content_hash: str
    is_duplicate: bool


log = get_logger(__name__)


class IngestService:
    """Serviço de ingest. Recebe dependências via construtor pra ser
    facilmente testável (passar mocks/fakes nos testes)."""

    def __init__(
        self,
        session: AsyncSession,
        embedding_provider: EmbeddingProvider,
        session_factory: Callable[[], AsyncSession] = AsyncSessionLocal,
    ) -> None:
        self._session = session
        self._embedding = embedding_provider
        # process_document() roda em background, fora do ciclo de vida da
        # `session` do request — precisa abrir a sua própria. Injetável
        # (em vez de importar AsyncSessionLocal fixo lá dentro) pra que
        # testes possam apontar pro mesmo engine de teste (NullPool) que o
        # resto da suite usa, evitando "Event loop is closed" entre testes.
        self._session_factory = session_factory

    async def ingest(self,
            content: bytes,
            filename: str,
            title: str | None = None,
        ) -> IngestResult:
        size = len(content)
        max_bytes = settings.max_upload_size_mb * 1024 * 1024
        if size > max_bytes:
            raise DocumentTooLargeError(size, max_bytes)
        content_hash = hashlib.sha256(content).hexdigest()
        stmt = select(Document).where(Document.content_sha256 == content_hash)
        result = await self._session.execute(stmt)
        existing = result.scalar_one_or_none()
        if (existing):
            stmt = select(func.count()).select_from(Chunk).where(Chunk.document_id == existing.id)
            chunks_count = await self._session.scalar(stmt) | 0
            return IngestResult(
                document_id=existing.id,
                title=existing.title,
                chunks_created=chunks_count,
                status=existing.status,
                content_hash=existing.content_sha256,
                is_duplicate=True,
            )
        document = Document(
            id=uuid.uuid4(),
            title=title or Path(filename).stem,
            source_filename=filename,
            status=DocumentStatus.PENDING,
            content_sha256=content_hash,
        )
        self._session.add(document)
        try:
            await self._session.flush()
        except IntegrityError as exc:
            raise DuplicateChunkError(filename) from exc
        await self._session.commit()
        return IngestResult(
            document_id=document.id,
            title=document.title,
            chunks_created=0,
            status=DocumentStatus.PENDING,
            content_hash=content_hash,
            is_duplicate=False,
        )

    async def process_document(
        self,
        document_id: uuid.UUID,
        content: bytes,
        filename: str,
    ) -> None:
        """Fase pesada do ingest: roda em background, numa sessão própria
        (a sessão do request original já fechou quando isso executa).
        Assume que o Document com `document_id` já existe (status=PENDING),
        criado pela fase leve em `ingest()`.
        """
        async with self._session_factory() as session:
            try:
                doc = await session.get(Document, document_id)
                if doc is None:
                    # Não há request HTTP esperando resposta aqui — não faz
                    # sentido levantar exceção. Só loga e desiste: o
                    # documento pode ter sido deletado enquanto esperava
                    # processamento (corrida rara, mas possível).
                    log.warning("ingest.process_document_not_found", document_id=str(document_id))
                    return

                suffix = Path(filename).suffix.lower()
                if suffix == ".pdf":
                    text, num_pages = parse_pdf(content)
                    if num_pages > settings.max_pdf_pages:
                        raise DocumentTooManyPagesError(num_pages, settings.max_pdf_pages)
                elif suffix in (".md", ".markdown", ".txt"):
                    text = content.decode("utf-8", errors="replace")
                else:
                    raise UnsupportedFileTypeError(suffix)

                chunks_data = chunk_text(text)
                if not chunks_data:
                    raise EmptyDocumentError(filename)

                for chunk_data in chunks_data:
                    matched = find_suspicious_pattern(chunk_data.content)
                    if matched:
                        log.warning(
                            "ingest.rejected",
                            reason="suspicious_content",
                            filename=filename,
                            matched_text=matched,
                        )
                        raise SuspiciousContentError(filename, matched)

                contents = [c.content for c in chunks_data]
                try:
                    embeddings = await self._embedding.embed_texts(contents)
                except OpenAIError as exc:
                    log.warning(
                        "ingest.embedding_failed",
                        filename=filename,
                        error=str(exc),
                    )
                    raise EmbeddingProviderError(str(exc)) from exc

                chunk_models = [
                    Chunk(
                        id=uuid.uuid4(),
                        document_id=doc.id,
                        chunk_index=chunk_data.index,
                        content=chunk_data.content,
                        embedding=embedding,
                    )
                    for chunk_data, embedding in zip(chunks_data, embeddings, strict=True)
                ]
                session.add_all(chunk_models)
                doc.status = DocumentStatus.INDEXED
                try:
                    await session.flush()
                except IntegrityError as exc:
                    # Cobre a unique constraint (document_id, chunk_index) — só
                    # alcançável por retry ou corrida, nunca por input do cliente.
                    raise DuplicateChunkError(filename) from exc

                await session.commit()
                log.info("ingest.process_document_done", document_id=str(document_id))
            except Exception as exc:
                await session.rollback()
                log.warning(
                    "ingest.process_document_failed",
                    document_id=str(document_id),
                    error=str(exc),
                )
                await self._mark_failed(document_id)

    async def _mark_failed(self, document_id: uuid.UUID) -> None:
        """Sessão própria e separada: a sessão de `process_document` pode
        estar em estado inválido após o rollback do erro que a levou aqui."""
        async with self._session_factory() as session:
            doc = await session.get(Document, document_id)
            if doc is not None:
                doc.status = DocumentStatus.FAILED
                await session.commit()