import uuid

from app.rag.graph import decide_after_retrieve
from app.rag.retrieval import RetrievedChunk
from app.rag.state import RagState


def _state(**overrides) -> RagState:
    base: RagState = {
        "question": "pergunta",
        "top_k": 5,
        "rewritten_question": None,
        "retrieved_chunks": [],
        "retry_count": 0,
        "answer": None,
        "answerable": None,
        "citations": [],
    }
    base.update(overrides)
    return base


def _chunk(similarity: float) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        document_title="doc",
        content="conteúdo",
        similarity=similarity,
    )


def test_decide_after_retrieve_goes_to_generate_when_retry_limit_reached(monkeypatch):
    from app.core.config import settings

    monkeypatch.setattr(settings, "max_rewrite_attempts", 2)

    result = decide_after_retrieve(_state(retry_count=2, retrieved_chunks=[]))

    assert result == "generate_answer"


def test_decide_after_retrieve_rewrites_when_no_chunks_retrieved():
    result = decide_after_retrieve(_state(retrieved_chunks=[]))

    assert result == "rewrite_query"


def test_decide_after_retrieve_rewrites_when_best_similarity_below_threshold(monkeypatch):
    from app.core.config import settings

    monkeypatch.setattr(settings, "hybrid_search_enabled", False)
    monkeypatch.setattr(settings, "retrieval_quality_threshold", 0.6)

    result = decide_after_retrieve(_state(retrieved_chunks=[_chunk(0.3)]))

    assert result == "rewrite_query"


def test_decide_after_retrieve_generates_when_best_similarity_meets_threshold(monkeypatch):
    from app.core.config import settings

    monkeypatch.setattr(settings, "hybrid_search_enabled", False)
    monkeypatch.setattr(settings, "retrieval_quality_threshold", 0.6)

    result = decide_after_retrieve(_state(retrieved_chunks=[_chunk(0.9)]))

    assert result == "generate_answer"


async def test_query_uses_dense_only_retrieval_when_hybrid_search_disabled(
    client, sample_txt_bytes, monkeypatch
):
    """hybrid_search_enabled=False: retrieve() usa só search_similar_chunks,
    sem rodar search_bm25_chunks nem reciprocal_rank_fusion (ver branch em
    app/rag/graph.py::make_retrieve_node). Assert é sobre o comportamento
    observável (query responde normalmente), já que o caminho interno
    escolhido não é exposto na resposta HTTP."""
    from app.core.config import settings

    await client.post(
        "/documents/ingest",
        files={"file": ("contrato.txt", sample_txt_bytes, "text/plain")},
    )

    monkeypatch.setattr(settings, "hybrid_search_enabled", False)
    response = await client.post("/query", json={"question": "Qual o prazo de entrega?"})

    assert response.status_code == 200
    assert len(response.json()["citations"]) > 0
