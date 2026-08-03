import uuid
from dataclasses import dataclass, replace

from sqlalchemy import Text, func, select
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


async def search_bm25_chunks(
    session: AsyncSession,
    query_text: str,
    top_k: int,
) -> list[RetrievedChunk]:
    """Busca os chunks com melhor match de termo exato à query, via full-text
    search do Postgres (tsvector/tsquery).

    Recebe o texto da pergunta, não um embedding — o ranking aqui é por
    relevância lexical (ts_rank_cd), não por distância vetorial.
    """
    # websearch_to_tsquery/plainto_tsquery unem os termos com AND — bom pra
    # buscas curtas, mas em perguntas longas em linguagem natural (com
    # palavras genéricas tipo "segundo", "glossário") nenhum chunk contém
    # todos os termos ao mesmo tempo e a busca não retorna nada. OR entre
    # os termos (mesma tokenização/stemming do dicionário 'portuguese', só
    # troca o operador) garante que um único termo específico batendo (ex:
    # "CRI") já produz match, com ts_rank_cd ranqueando mais alto quem bate
    # mais termos — é esse comportamento que faz a busca por termo exato
    # valer a pena sobre o dense sozinho.
    tsquery = func.to_tsquery(
        "portuguese",
        func.regexp_replace(
            func.websearch_to_tsquery("portuguese", query_text).cast(Text),
            r"\s*&\s*",
            " | ",
            "g",
        ),
    )
    rank = func.ts_rank_cd(Chunk.content_tsv, tsquery)
    stmt = (
        select(Chunk, Document.title, rank.label("rank"))
        .join(Document, Chunk.document_id == Document.id)
        .where(Chunk.content_tsv.op("@@")(tsquery))
        .order_by(rank.desc())
        .limit(top_k)
    )
    result = await session.execute(stmt)
    return [
        RetrievedChunk(
            chunk_id=chunk.id,
            document_id=chunk.document_id,
            document_title=title,
            content=chunk.content,
            similarity=rank_value,
        )
        for chunk, title, rank_value in result.all()
    ]


def reciprocal_rank_fusion(
    *rankings: list[RetrievedChunk],
    k: int = 60,
) -> list[RetrievedChunk]:
    """Funde múltiplos rankings de chunks num único, por posição — não por
    score bruto. Cosine similarity e ts_rank_cd vivem em escalas diferentes
    e não são comparáveis diretamente; RRF ignora a escala e usa só a
    posição de cada chunk em cada lista, então não precisa de calibração.

    k=60 é o valor padrão usado na literatura (Cormack et al., 2009) —
    achata o peso das posições mais altas o suficiente pra um chunk só bem
    colocado numa lista não dominar sozinho o resultado.
    """
    scores: dict[uuid.UUID, float] = {}
    chunks_by_id: dict[uuid.UUID, RetrievedChunk] = {}

    for ranking in rankings:
        for position, chunk in enumerate(ranking):
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + 1 / (k + position + 1)
            chunks_by_id[chunk.chunk_id] = chunk

    fused_order = sorted(scores, key=lambda chunk_id: scores[chunk_id], reverse=True)
    return [
        replace(chunks_by_id[chunk_id], similarity=scores[chunk_id]) for chunk_id in fused_order
    ]
