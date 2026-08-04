import pytest
from pydantic import BaseModel

from app.api.query import get_chat, get_embedding
from app.main import app


class _RefusingChatProvider:
    """Sempre responde answerable=False, sem depender de conteúdo real
    recuperado — força o branch security.answer_refused em query.py,
    diferente de FakeChatProvider (sempre True para campos bool)."""

    async def complete(self, prompt: str, *, system: str | None = None) -> str:
        return "reformulação qualquer"

    async def complete_structured(self, prompt: str, *, system: str, schema: type[BaseModel]):
        return schema(answerable=False, answer="Não encontrei essa informação nos documentos.")


@pytest.mark.vcr
async def test_query_returns_answer_with_real_openai(
    client, sample_txt_bytes, real_embedding_provider, real_chat_provider
):
    """Cobre o contrato real de resposta da OpenAI para o fluxo de /query
    (embedding da pergunta + chat estruturado via with_structured_output) —
    diferente do resto da suite (Fake*Provider), gravado uma vez via VCR
    (ver docs/roadmap.md, trilha 'infraestrutura de teste'). Ingest usa
    embedding fake de propósito: só o /query em si precisa do contrato
    real aqui, e manter o ingest fake mantém o cassete pequeno."""
    await client.post(
        "/documents/ingest",
        files={"file": ("contrato.txt", sample_txt_bytes, "text/plain")},
    )

    previous_embedding = app.dependency_overrides[get_embedding]
    previous_chat = app.dependency_overrides[get_chat]
    app.dependency_overrides[get_embedding] = lambda: real_embedding_provider
    app.dependency_overrides[get_chat] = lambda: real_chat_provider
    try:
        response = await client.post("/query", json={"question": "Qual o prazo de entrega?"})
    finally:
        app.dependency_overrides[get_embedding] = previous_embedding
        app.dependency_overrides[get_chat] = previous_chat

    assert response.status_code == 200
    body = response.json()
    assert body["answer"]
    assert body["answerable"] is True
    assert len(body["citations"]) > 0
    citation = body["citations"][0]
    assert {"document_id", "chunk_id", "snippet", "score"} <= citation.keys()


async def test_query_returns_answer_and_citations(client, sample_txt_bytes):
    await client.post(
        "/documents/ingest",
        files={"file": ("contrato.txt", sample_txt_bytes, "text/plain")},
    )

    response = await client.post("/query", json={"question": "Qual o prazo de entrega?"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"]
    assert body["answerable"] is not None
    assert len(body["citations"]) > 0
    citation = body["citations"][0]
    assert {"document_id", "chunk_id", "snippet", "score"} <= citation.keys()
    assert body["trace_id"] is None


async def test_query_respects_top_k_override(client, sample_txt_bytes):
    await client.post(
        "/documents/ingest",
        files={"file": ("contrato.txt", sample_txt_bytes, "text/plain")},
    )

    response = await client.post(
        "/query", json={"question": "Qual o prazo de entrega?", "top_k": 1}
    )

    assert response.status_code == 200
    assert len(response.json()["citations"]) <= 1


async def test_query_without_any_documents_returns_no_citations(client, capsys):
    response = await client.post("/query", json={"question": "Qualquer pergunta"})

    assert response.status_code == 200
    body = response.json()
    assert body["citations"] == []

    # Sem documentos, retrieved_chunks fica vazio em toda tentativa —
    # o grafo bate max_rewrite_attempts, o que deve gerar o log de segurança.
    # structlog escreve direto em stdout (não passa pelo stdlib logging),
    # então capsys em vez de caplog.
    assert "security.rewrite_limit_reached" in capsys.readouterr().out
    assert body["answerable"] is not None


async def test_query_missing_question_returns_422(client):
    response = await client.post("/query", json={})

    assert response.status_code == 422


async def test_query_logs_security_event_when_answer_is_refused(client, sample_txt_bytes, capsys):
    await client.post(
        "/documents/ingest",
        files={"file": ("contrato.txt", sample_txt_bytes, "text/plain")},
    )

    previous_chat = app.dependency_overrides[get_chat]
    app.dependency_overrides[get_chat] = lambda: _RefusingChatProvider()
    try:
        response = await client.post("/query", json={"question": "Qual o prazo de entrega?"})
    finally:
        app.dependency_overrides[get_chat] = previous_chat

    assert response.status_code == 200
    assert response.json()["answerable"] is False
    assert "security.answer_refused" in capsys.readouterr().out


async def test_query_includes_trace_id_when_langsmith_tracing_enabled(
    client, sample_txt_bytes, monkeypatch
):
    """LANGSMITH_TRACING=true dispara o wrapper langsmith.trace() em vez do
    ainvoke() direto — mocka app.api.query.trace em vez de deixá-lo rodar
    de verdade: mesmo sem LANGSMITH_API_KEY configurada, o client real do
    LangSmith abre uma conexão HTTP em background pra tentar enviar o
    trace (fire-and-forget, não bloqueia o request) — essa conexão não
    fecha antes do fim do teste e deixa um socket pendurado que
    filterwarnings=["error"] (pyproject.toml) promove a erro fatal no
    cleanup do pytest, derrubando a suite de forma intermitente. O mock
    aqui só precisa devolver um objeto com `.id`, que é tudo que o
    endpoint usa do run_tree retornado por trace()."""
    import uuid
    from contextlib import contextmanager
    from unittest.mock import patch

    from app.core.config import settings

    await client.post(
        "/documents/ingest",
        files={"file": ("contrato.txt", sample_txt_bytes, "text/plain")},
    )

    fake_run_tree = type("RunTree", (), {"id": uuid.uuid4()})()

    @contextmanager
    def _fake_trace(*args, **kwargs):
        yield fake_run_tree

    monkeypatch.setattr(settings, "langsmith_tracing", True)
    with patch("app.api.query.trace", _fake_trace):
        response = await client.post("/query", json={"question": "Qual o prazo de entrega?"})

    assert response.status_code == 200
    assert response.json()["trace_id"] == str(fake_run_tree.id)
