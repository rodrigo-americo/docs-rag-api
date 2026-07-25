import uuid

import pytest

from app.core.sanitize import strip_html
from app.rag.retrieval import RetrievedChunk
from app.schemas.query import Citation


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("<script>alert(1)</script>", "alert(1)"),
        ('<img src=x onerror="alert(1)">', ""),
        ("<b>texto em negrito</b>", "texto em negrito"),
        ("texto normal sem tags", "texto normal sem tags"),
        ("O prazo é de <b>10</b> dias úteis.", "O prazo é de 10 dias úteis."),
    ],
)
def test_strip_html_removes_tags_keeps_text(raw: str, expected: str):
    assert strip_html(raw) == expected


def test_citation_from_chunk_sanitizes_snippet():
    chunk = RetrievedChunk(
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        document_title="doc.txt",
        content="Prazo: <script>alert('xss')</script>10 dias úteis.",
        similarity=0.9,
    )

    citation = Citation.from_chunk(chunk)

    assert "<script>" not in citation.snippet
    assert "10 dias úteis" in citation.snippet
