"""Ingere os documentos de avaliação e lista os chunks gerados com seus IDs.

Rodar uma vez antes de montar o dataset de perguntas (evals/dataset.json) —
os expected_chunk_id só existem depois que os documentos são ingeridos.

Uso: uv run python -m evals.setup_dataset
"""

import asyncio
import json
from pathlib import Path

from sqlalchemy import select

from app.core.db import AsyncSessionLocal
from app.models.document import Chunk, Document
from app.services.embedding import get_embedding_provider
from app.services.ingest import IngestService

DOCUMENTS_DIR = Path(__file__).parent / "documents"
OUTPUT_PATH = Path(__file__).parent / "chunks_reference.json"


async def main() -> None:
    embedding_provider = get_embedding_provider()

    async with AsyncSessionLocal() as session:
        service = IngestService(session=session, embedding_provider=embedding_provider)

        for file_path in sorted(DOCUMENTS_DIR.glob("*.pdf")):
            content = file_path.read_bytes()
            result = await service.ingest(content=content, filename=file_path.name)
            # ingest() só cria o Document (status=PENDING) e salva o upload —
            # é a fase leve, pensada pra rodar dentro de um request HTTP que
            # depois enfileira o processamento pro worker (app/worker.py).
            # Aqui não há fila: chamamos process_document() direto, síncrono,
            # pra gerar chunks e embeddings de verdade sem precisar subir
            # Redis + worker só pra montar o dataset de avaliação.
            status = await service.process_document(result.document_id)
            print(f"Ingerido: {file_path.name} -> status={status}")

        stmt = (
            select(Chunk, Document.title)
            .join(Document, Chunk.document_id == Document.id)
            .order_by(Document.title, Chunk.chunk_index)
        )
        rows = (await session.execute(stmt)).all()

    reference = [
        {
            "document_title": title,
            "chunk_id": str(chunk.id),
            "chunk_index": chunk.chunk_index,
            "content": chunk.content,
        }
        for chunk, title in rows
    ]

    OUTPUT_PATH.write_text(json.dumps(reference, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n{len(reference)} chunks escritos em {OUTPUT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
