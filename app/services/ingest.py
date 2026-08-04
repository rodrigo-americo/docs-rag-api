import hashlib
import time
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
    DuplicateChunkError,
    EmbeddingProviderError,
    EmptyDocumentError,
    SuspiciousContentError,
)
from app.core.logging import get_logger
from app.core.metrics import ingest_processed_total, ingest_processing_seconds
from app.core.storage import delete_upload, read_upload, save_upload
from app.models.document import Chunk, Document, DocumentStatus
from app.services.chunking import TextChunk, chunk_text
from app.services.embedding import EmbeddingProvider
from app.services.injection_detection import find_suspicious_pattern
from app.services.parsers import extract_text


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
        embedding_provider: EmbeddingProvider,
        session: AsyncSession | None = None,
        session_factory: Callable[[], AsyncSession] = AsyncSessionLocal,
    ) -> None:
        # session só é usada por ingest() (fase leve, dentro de um request
        # HTTP). process_document() (fase pesada, chamada pelo worker) nunca
        # a usa — só session_factory, pra abrir sua própria sessão. Por isso
        # é opcional: quem só chama process_document (o worker) não precisa
        # fingir que tem uma sessão de request.
        self._session = session
        self._embedding = embedding_provider
        # process_document() roda em background, fora do ciclo de vida da
        # `session` do request — precisa abrir a sua própria. Injetável
        # (em vez de importar AsyncSessionLocal fixo lá dentro) pra que
        # testes possam apontar pro mesmo engine de teste (NullPool) que o
        # resto da suite usa, evitando "Event loop is closed" entre testes.
        self._session_factory = session_factory

    async def ingest(
        self,
        content: bytes,
        filename: str,
        title: str | None = None,
    ) -> IngestResult:
        """Fase leve do ingest: valida tamanho, calcula hash, checa
        duplicata e cria o Document (status=pending). Roda dentro do
        request — nada de parse/embedding aqui, isso é process_document().
        """
        assert self._session is not None, "ingest() exige session (passada no construtor)"
        size = len(content)
        max_bytes = settings.max_upload_size_mb * 1024 * 1024
        if size > max_bytes:
            raise DocumentTooLargeError(size, max_bytes)

        content_hash = hashlib.sha256(content).hexdigest()

        existing = await self._find_duplicate(content_hash)
        if existing is not None:
            chunks_count = await self._count_chunks(existing.id)
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
        save_upload(document.id, content)
        return IngestResult(
            document_id=document.id,
            title=document.title,
            chunks_created=0,
            status=DocumentStatus.PENDING,
            content_hash=content_hash,
            is_duplicate=False,
        )

    async def _find_duplicate(self, content_hash: str) -> Document | None:
        stmt = select(Document).where(Document.content_sha256 == content_hash)
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()

    async def _count_chunks(self, document_id: uuid.UUID) -> int:
        stmt = select(func.count()).select_from(Chunk).where(Chunk.document_id == document_id)
        return await self._session.scalar(stmt) or 0

    async def process_document(
        self,
        document_id: uuid.UUID,
    ) -> DocumentStatus | None:
        """Fase pesada do ingest: roda em background, numa sessão própria
        (a sessão do request original já fechou quando isso executa).
        Assume que o Document com `document_id` já existe (status=PENDING),
        criado pela fase leve em `ingest()`. filename vem do próprio
        Document (source_filename) — não precisa viajar como parâmetro,
        já que quem chama (worker) só tem o document_id vindo da fila.

        Devolve o status final (INDEXED ou FAILED) pra quem chamou decidir
        o que fazer com a mensagem da fila (ack vs retry) sem precisar
        reconsultar o banco. None só no caso raro do Document não existir
        mais (ver abaixo).

        Mede ingest_processing_seconds e incrementa ingest_processed_total
        (métricas Prometheus, ver app/core/metrics.py) em torno do trabalho
        pesado (parse+embedding+persist) — não do caso "não encontrado" acima,
        que nunca chega a processar nada.
        """
        async with self._session_factory() as session:
            started_at = time.perf_counter()
            try:
                doc = await session.get(Document, document_id)
                if doc is None:
                    # Não há request HTTP esperando resposta aqui — não faz
                    # sentido levantar exceção. Só loga e desiste: o
                    # documento pode ter sido deletado enquanto esperava
                    # processamento (corrida rara, mas possível).
                    log.warning("ingest.process_document_not_found", document_id=str(document_id))
                    return None

                filename = doc.source_filename or "unknown"

                doc.status = DocumentStatus.PROCESSING
                await session.commit()

                content = read_upload(document_id)
                text = extract_text(content, filename)
                chunks_data = chunk_text(text)
                if not chunks_data:
                    raise EmptyDocumentError(filename)

                self._check_suspicious_content(chunks_data, filename)
                embeddings = await self._generate_embeddings(chunks_data, filename)

                self._persist_chunks(session, doc, chunks_data, embeddings)
                doc.status = DocumentStatus.INDEXED
                try:
                    await session.flush()
                except IntegrityError as exc:
                    # Cobre a unique constraint (document_id, chunk_index) — só
                    # alcançável por retry ou corrida, nunca por input do cliente.
                    raise DuplicateChunkError(filename) from exc

                await session.commit()
                log.info("ingest.process_document_done", document_id=str(document_id))
                delete_upload(document_id)
                ingest_processing_seconds.observe(time.perf_counter() - started_at)
                ingest_processed_total.labels(status=DocumentStatus.INDEXED).inc()
                return DocumentStatus.INDEXED
            except Exception as exc:
                await session.rollback()
                log.warning(
                    "ingest.process_document_failed",
                    document_id=str(document_id),
                    error=str(exc),
                )
                await self._mark_failed(document_id, str(exc))
                ingest_processing_seconds.observe(time.perf_counter() - started_at)
                ingest_processed_total.labels(status=DocumentStatus.FAILED).inc()
                # NÃO apaga o arquivo aqui: quem chama pode reenfileirar pra
                # nova tentativa (ver app/worker.py), e o arquivo precisa
                # continuar disponível pra isso. Só apaga em caso de sucesso
                # (acima) ou quando quem chama decide que a falha é
                # definitiva (esgotou as tentativas).
                return DocumentStatus.FAILED

    def _check_suspicious_content(self, chunks_data: list[TextChunk], filename: str) -> None:
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

    async def _generate_embeddings(
        self, chunks_data: list[TextChunk], filename: str
    ) -> list[list[float]]:
        contents = [c.content for c in chunks_data]
        try:
            return await self._embedding.embed_texts(contents)
        except OpenAIError as exc:
            log.warning(
                "ingest.embedding_failed",
                filename=filename,
                error=str(exc),
            )
            raise EmbeddingProviderError(str(exc)) from exc

    def _persist_chunks(
        self,
        session: AsyncSession,
        document: Document,
        chunks_data: list[TextChunk],
        embeddings: list[list[float]],
    ) -> list[Chunk]:
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
        session.add_all(chunk_models)
        return chunk_models

    async def _mark_failed(self, document_id: uuid.UUID, error: str) -> None:
        """Sessão própria e separada: a sessão de `process_document` pode
        estar em estado inválido após o rollback do erro que a levou aqui.

        Por enquanto, toda falha vai direto pra FAILED (definitivo) — não há
        requeue automático ainda, porque isso só faz sentido quando existir
        uma fila de verdade decidindo se vale tentar de novo. retry_count e
        last_error já são rastreados desde já, preparando o terreno pra
        quando o worker (issue #7) passar a decidir "tenta de novo" vs
        "desiste" usando esses mesmos campos.
        """
        async with self._session_factory() as session:
            doc = await session.get(Document, document_id)
            if doc is None:
                return
            doc.retry_count += 1
            doc.last_error = error
            doc.status = DocumentStatus.FAILED
            await session.commit()
