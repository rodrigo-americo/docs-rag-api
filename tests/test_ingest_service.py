import uuid
from unittest.mock import AsyncMock, patch

import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from app.core.exceptions import DocumentTooLargeError
from app.models.document import Chunk, Document, DocumentStatus
from app.services.embedding import FakeEmbeddingProvider
from app.services.ingest import IngestService
from tests.conftest import get_test_session_factory

TestSessionLocal = get_test_session_factory()


async def test_ingest_raises_when_content_exceeds_max_upload_size(monkeypatch):
    from app.core.config import settings

    monkeypatch.setattr(settings, "max_upload_size_mb", 0)
    async with TestSessionLocal() as session:
        service = IngestService(session=session, embedding_provider=FakeEmbeddingProvider())

        with pytest.raises(DocumentTooLargeError):
            await service.ingest(content=b"qualquer conteudo", filename="doc.txt")


async def test_ingest_returns_existing_document_on_duplicate_content(sample_txt_bytes):
    async with TestSessionLocal() as session:
        service = IngestService(session=session, embedding_provider=FakeEmbeddingProvider())
        first = await service.ingest(content=sample_txt_bytes, filename="original.txt")

    async with TestSessionLocal() as session:
        service = IngestService(session=session, embedding_provider=FakeEmbeddingProvider())
        second = await service.ingest(content=sample_txt_bytes, filename="reenvio.txt")

    assert second.is_duplicate is True
    assert second.document_id == first.document_id
    # Título e nome do arquivo do reenvio são ignorados — o documento já
    # existente (do primeiro ingest) é o que é devolvido.
    assert second.title == first.title


async def test_ingest_duplicate_reports_chunks_already_created(sample_txt_bytes):
    """chunks_created no retorno de duplicata reflete o que já foi indexado
    pelo ingest original, não 0 — diferente do ingest novo (Fase A, antes
    do processamento pesado), aqui o documento já pode estar 'indexed'."""
    async with TestSessionLocal() as session:
        service = IngestService(session=session, embedding_provider=FakeEmbeddingProvider())
        first = await service.ingest(content=sample_txt_bytes, filename="original.txt")

    async with TestSessionLocal() as session:
        service = IngestService(
            session=session,
            embedding_provider=FakeEmbeddingProvider(),
            session_factory=TestSessionLocal,
        )
        status = await service.process_document(first.document_id)
        assert status == DocumentStatus.INDEXED

    async with TestSessionLocal() as session:
        service = IngestService(session=session, embedding_provider=FakeEmbeddingProvider())
        duplicate = await service.ingest(content=sample_txt_bytes, filename="reenvio.txt")

    assert duplicate.is_duplicate is True
    assert duplicate.chunks_created > 0
    assert duplicate.status == DocumentStatus.INDEXED


async def test_process_document_returns_none_when_document_does_not_exist():
    service = IngestService(
        embedding_provider=FakeEmbeddingProvider(), session_factory=TestSessionLocal
    )

    result = await service.process_document(uuid.uuid4())

    assert result is None


async def test_process_document_marks_failed_and_records_error_on_unexpected_exception(
    sample_txt_bytes,
):
    """Cobre o except Exception genérico de process_document (não um dos
    IngestError conhecidos) — aqui provocado ao apagar o arquivo do volume
    depois do Document já ter sido criado, então read_upload() explode com
    FileNotFoundError, um erro que process_document não antecipa
    especificamente mas ainda precisa marcar como failed."""
    async with TestSessionLocal() as session:
        service = IngestService(session=session, embedding_provider=FakeEmbeddingProvider())
        result = await service.ingest(content=sample_txt_bytes, filename="sem_arquivo.txt")

    from app.core.storage import delete_upload

    delete_upload(result.document_id)

    service = IngestService(
        embedding_provider=FakeEmbeddingProvider(), session_factory=TestSessionLocal
    )
    status = await service.process_document(result.document_id)

    assert status == DocumentStatus.FAILED

    async with TestSessionLocal() as session:
        doc = await session.get(Document, result.document_id)
        assert doc.status == DocumentStatus.FAILED
        assert doc.retry_count == 1
        assert doc.last_error is not None


async def test_process_document_raises_duplicate_chunk_error_on_integrity_error(
    sample_txt_bytes,
):
    """Cobre a unique constraint (document_id, chunk_index) em
    process_document — só alcançável simulando uma corrida: insere um
    Chunk com chunk_index=0 manualmente antes do processamento normal
    rodar, forçando a mesma posição a colidir."""
    async with TestSessionLocal() as session:
        service = IngestService(session=session, embedding_provider=FakeEmbeddingProvider())
        result = await service.ingest(content=sample_txt_bytes, filename="colisao.txt")

    async with TestSessionLocal() as session:
        session.add(
            Chunk(
                id=uuid.uuid4(),
                document_id=result.document_id,
                chunk_index=0,
                content="chunk pré-existente colidindo com o índice 0",
                embedding=FakeEmbeddingProvider()._embed_one("colisão"),
            )
        )
        await session.commit()

    service = IngestService(
        embedding_provider=FakeEmbeddingProvider(), session_factory=TestSessionLocal
    )
    status = await service.process_document(result.document_id)

    assert status == DocumentStatus.FAILED

    async with TestSessionLocal() as session:
        doc = await session.get(Document, result.document_id)
        assert doc.status == DocumentStatus.FAILED
        assert "duplicado" in doc.last_error.lower() or "conflito" in doc.last_error.lower()
        # Nenhum chunk novo foi persistido — o flush falhou e o rollback
        # desfez tudo, sobrando só o chunk inserido manualmente acima.
        chunks = (
            (await session.execute(select(Chunk).where(Chunk.document_id == result.document_id)))
            .scalars()
            .all()
        )
        assert len(chunks) == 1


async def test_ingest_raises_duplicate_chunk_error_on_integrity_error_during_flush():
    """Corrida rara: dois requests com o mesmo conteúdo passam por
    _find_duplicate() antes de qualquer um commitar, e o segundo bate na
    unique constraint de content_sha256 só no flush — não simulável
    deterministicamente com dois requests reais, então mocka
    session.flush() pra forçar o mesmo IntegrityError que o banco levantaria."""
    async with TestSessionLocal() as session:
        service = IngestService(session=session, embedding_provider=FakeEmbeddingProvider())
        with patch.object(
            session, "flush", AsyncMock(side_effect=IntegrityError("stmt", {}, Exception("dup")))
        ):
            from app.core.exceptions import DuplicateChunkError

            with pytest.raises(DuplicateChunkError):
                await service.ingest(content=b"conteudo unico demais", filename="corrida.txt")


async def test_process_document_marks_failed_when_chunking_produces_no_chunks():
    """EmptyDocumentError: um .txt só com espaços/quebras de linha decodifica
    normalmente como UTF-8 válido, mas chunk_text() sobre esse texto produz
    lista vazia — chega em process_document() sem erro de parse, só sem
    conteúdo aproveitável pra indexar."""
    async with TestSessionLocal() as session:
        service = IngestService(session=session, embedding_provider=FakeEmbeddingProvider())
        result = await service.ingest(content=b"   \n\n   ", filename="so_espacos.txt")

    service = IngestService(
        embedding_provider=FakeEmbeddingProvider(), session_factory=TestSessionLocal
    )
    status = await service.process_document(result.document_id)

    assert status == DocumentStatus.FAILED

    async with TestSessionLocal() as session:
        doc = await session.get(Document, result.document_id)
        assert doc.status == DocumentStatus.FAILED
        assert "conteúdo" in doc.last_error.lower()


async def test_mark_failed_is_a_noop_when_document_no_longer_exists():
    """_mark_failed é chamado de dentro do except de process_document, numa
    sessão própria — se o Document sumiu entre o erro acontecer e
    _mark_failed rodar (ex: DELETE concorrente), não há nada pra marcar, só
    retorna sem erro. Chamado direto (não via process_document) porque
    reproduzir essa corrida de verdade exigiria um DELETE no meio da
    execução do método."""
    service = IngestService(
        embedding_provider=FakeEmbeddingProvider(), session_factory=TestSessionLocal
    )

    # Não levanta exceção — é a única asserção possível pra um no-op.
    await service._mark_failed(uuid.uuid4(), "erro que nunca será registrado")
