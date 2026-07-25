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
