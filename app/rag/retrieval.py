import uuid
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.document import Chunk, Document


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    document_title: str
    content: str
    similarity: float


async def search_similar_chunks(
    session: AsyncSession,
    query_embedding: list[float],
    top_k: int,
) -> list[RetrievedChunk]:
    """Busca os chunks mais similares ao embedding da query, via pgvector.

    Recebe o vetor já embedado — não o texto da pergunta. Quem chama esta
    função é responsável por rodar embedding_provider.embed_query antes.
    """
    distance = Chunk.embedding.cosine_distance(query_embedding)
    stmt = (
        select(Chunk, Document.title, distance.label("distance"))
        .join(Document, Chunk.document_id == Document.id)
        .order_by(distance)
        .limit(top_k)
    )
    result = await session.execute(stmt)
    return [
        RetrievedChunk(
            chunk_id=chunk.id,
            document_id=chunk.document_id,
            document_title=title,
            content=chunk.content,
            similarity=1 - distance,
        )
        for chunk, title, distance in result.all()
    ]
