import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.exceptions import (
    DocumentTooLargeError,
    DocumentTooManyPagesError,
    DuplicateChunkError,
    EmptyDocumentError,
    UnsupportedFileTypeError,
)
from app.core.logging import get_logger
from app.models.document import Chunk, Document
from app.services.chunking import chunk_text
from app.services.embedding import EmbeddingProvider
from app.services.pdf_parser import parse_pdf


@dataclass
class IngestResult:
    document_id: uuid.UUID
    title: str
    chunks_created: int

log = get_logger(__name__)

class IngestService:
    """Serviço de ingest. Recebe dependências via construtor pra ser
    facilmente testável (passar mocks/fakes nos testes)."""

    def __init__(
        self,
        session: AsyncSession,
        embedding_provider: EmbeddingProvider,
    ) -> None:
        self._session = session
        self._embedding = embedding_provider

    async def ingest(
        self,
        content: bytes,
        filename: str,
        title: str | None = None,
    ) -> IngestResult:
        """Ingere um arquivo completo: valida, parseia, chunka, embeda, persiste.

        A transação é responsabilidade do caller (dependency get_db_session
        commita no fim do request). Aqui só fazemos `add` — o flush/commit
        é centralizado.
        """
        start = time.perf_counter()
        log.info(
            "ingest.start",
            filename=filename,
            size_bytes=len(content),
        )
        # 1. Validação de tamanho (já validada no endpoint, mas defesa em
        #    profundidade — service não confia no caller).
        size = len(content)
        max_bytes = settings.max_upload_size_mb * 1024 * 1024
        if size > max_bytes:
            raise DocumentTooLargeError(size, max_bytes)

        # 2. Determina tipo pela extensão (simplório mas suficiente pro escopo).
        suffix = Path(filename).suffix.lower()
        if suffix == ".pdf":
            text, num_pages = parse_pdf(content)
            if num_pages > settings.max_pdf_pages:
                raise DocumentTooManyPagesError(num_pages, settings.max_pdf_pages)
        elif suffix in (".md", ".markdown", ".txt"):
            text = content.decode("utf-8", errors="replace")
        else:
            raise UnsupportedFileTypeError(suffix)

        # 3. Chunking (CPU local, rápido).
        chunks_data = chunk_text(text)
        if not chunks_data:
            raise EmptyDocumentError(filename)
        # 4. Embedding em batch — FORA da transação. API externa nunca
        #    dentro de lock de banco.
        log.info(
            "ingest.embedding_start",
            filename=filename,
            chunks=len(chunks_data),
        )
        embed_start = time.perf_counter()
        contents = [c.content for c in chunks_data]
        embeddings = await self._embedding.embed_texts(contents)
        log.info(
            "ingest.embedding_done",
            filename=filename,
            chunks=len(chunks_data),
            duration_ms=round((time.perf_counter() - embed_start) * 1000),
        )

        # 5. Persistência: cria Document + Chunks. O commit é centralizado
        #    na dependency get_db_session.
        document = Document(
            id=uuid.uuid4(),
            title=title or Path(filename).stem,
            source_filename=filename,
        )
        self._session.add(document)

        chunk_models = [
            Chunk(
                id=uuid.uuid4(),
                document_id=document.id,
                chunk_index=chunk_data.index,
                content=chunk_data.content,
                embedding=embedding,
            )
            for chunk_data, embedding in zip(chunks_data, embeddings, strict=True)
        ]
        self._session.add_all(chunk_models)
        log.info(
            "ingest.done",
            document_id=str(document.id),
            filename=filename,
            chunks=len(chunk_models),
            duration_ms=round((time.perf_counter() - start) * 1000),
        )
        # Flush força INSERT mas NÃO commita — útil pra pegar erros de
        # constraint cedo, antes do commit do request.
        try:
            await self._session.flush()
        except IntegrityError as exc:
            # Cobre a unique constraint (document_id, chunk_index) — só
            # alcançável por retry ou corrida, nunca por input do cliente.
            raise DuplicateChunkError(filename) from exc

        return IngestResult(
            document_id=document.id,
            title=document.title,
            chunks_created=len(chunk_models),
        )
