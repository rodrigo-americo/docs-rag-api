async def test_ingest_txt_succeeds(client, sample_txt_bytes):
    response = await client.post(
        "/documents/ingest",
        files={"file": ("contrato.txt", sample_txt_bytes, "text/plain")},
        data={"title": "Contrato de Teste"},
    )

    assert response.status_code == 202
    body = response.json()
    assert body["title"] == "Contrato de Teste"
    assert body["chunks_created"] >= 1
    assert body["status"] == "indexed"


async def test_ingest_without_title_uses_filename_stem(client, sample_txt_bytes):
    response = await client.post(
        "/documents/ingest",
        files={"file": ("meu_documento.txt", sample_txt_bytes, "text/plain")},
    )

    assert response.status_code == 202
    assert response.json()["title"] == "meu_documento"


async def test_ingest_empty_file_returns_422(client):
    response = await client.post(
        "/documents/ingest",
        files={"file": ("vazio.txt", b"", "text/plain")},
    )

    assert response.status_code == 422


async def test_ingest_unsupported_extension_returns_415(client):
    response = await client.post(
        "/documents/ingest",
        files={"file": ("malware.exe", b"conteudo qualquer", "application/octet-stream")},
    )

    assert response.status_code == 415


async def test_ingest_corrupted_pdf_returns_422(client):
    response = await client.post(
        "/documents/ingest",
        files={"file": ("fake.pdf", b"%PDF-1.4 nao e um pdf valido", "application/pdf")},
    )

    assert response.status_code == 422
