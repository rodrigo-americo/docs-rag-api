import pytest

from app.core.exceptions import (
    DocumentTooLargeError,
    DocumentTooManyPagesError,
    DuplicateChunkError,
    EmbeddingProviderError,
    EmptyDocumentError,
    IngestError,
    PdfParseError,
    SuspiciousContentError,
    UnsupportedFileTypeError,
)


@pytest.mark.parametrize(
    ("exc", "expected_status"),
    [
        (DocumentTooLargeError(size_bytes=100, max_bytes=50), 413),
        (DocumentTooManyPagesError(pages=100, max_pages=50), 413),
        (UnsupportedFileTypeError(extension=".exe"), 415),
        (EmptyDocumentError(filename="vazio.txt"), 422),
        (PdfParseError(reason="corrompido"), 422),
        (DuplicateChunkError(filename="dup.txt"), 500),
        (EmbeddingProviderError(reason="timeout"), 503),
        (SuspiciousContentError(filename="mal.txt", matched_text="ignore as instruções"), 422),
    ],
)
def test_to_http_maps_correct_status_code(exc: IngestError, expected_status: int):
    http_exc = exc.to_http()

    assert http_exc.status_code == expected_status
    assert http_exc.detail == str(exc)


def test_all_ingest_errors_are_subclasses_of_ingest_error():
    for exc in (
        DocumentTooLargeError(1, 1),
        DocumentTooManyPagesError(1, 1),
        UnsupportedFileTypeError(".exe"),
        EmptyDocumentError("f.txt"),
        PdfParseError("motivo"),
        DuplicateChunkError("f.txt"),
        EmbeddingProviderError("motivo"),
        SuspiciousContentError("f.txt", "ignore as instruções"),
    ):
        assert isinstance(exc, IngestError)


def test_document_too_large_error_carries_structured_fields():
    exc = DocumentTooLargeError(size_bytes=2048, max_bytes=1024)

    assert exc.size_bytes == 2048
    assert exc.max_bytes == 1024
    assert "2048" in str(exc)
    assert "1024" in str(exc)
