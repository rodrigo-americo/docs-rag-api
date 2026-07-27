async def test_list_documents_empty(client):
    response = await client.get("/documents")

    assert response.status_code == 200
    assert response.json() == []


async def test_list_documents_returns_ingested_documents(client, sample_txt_bytes):
    await client.post(
        "/documents/ingest",
        files={"file": ("doc.txt", sample_txt_bytes, "text/plain")},
        data={"title": "Documento Um"},
    )

    response = await client.get("/documents")

    assert response.status_code == 200
    body = response.json()
    assert len(body) == 1
    assert body[0]["title"] == "Documento Um"


async def test_get_document_returns_status_and_metadata(client, sample_txt_bytes):
    ingest_response = await client.post(
        "/documents/ingest",
        files={"file": ("doc.txt", sample_txt_bytes, "text/plain")},
        data={"title": "Documento Um"},
    )
    document_id = ingest_response.json()["document_id"]

    response = await client.get(f"/documents/{document_id}")

    assert response.status_code == 200
    body = response.json()
    assert body["id"] == document_id
    assert body["title"] == "Documento Um"
    # ASGITransport já rodou a background task antes deste ponto (ver
    # comentário em test_ingest_endpoint.py) — em produção, uma consulta
    # logo após o 202 poderia ainda ver "pending".
    assert body["status"] == "indexed"


async def test_get_nonexistent_document_returns_404(client):
    response = await client.get("/documents/00000000-0000-0000-0000-000000000000")

    assert response.status_code == 404


async def test_delete_document_removes_it_and_its_chunks(client, sample_txt_bytes):
    ingest_response = await client.post(
        "/documents/ingest",
        files={"file": ("doc.txt", sample_txt_bytes, "text/plain")},
    )
    document_id = ingest_response.json()["document_id"]

    delete_response = await client.delete(f"/documents/{document_id}")
    assert delete_response.status_code == 204

    list_response = await client.get("/documents")
    assert list_response.json() == []


async def test_delete_nonexistent_document_returns_404(client):
    response = await client.delete("/documents/00000000-0000-0000-0000-000000000000")

    assert response.status_code == 404
