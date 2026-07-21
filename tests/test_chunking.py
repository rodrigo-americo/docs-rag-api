import tiktoken

from app.core.config import settings
from app.services.chunking import chunk_text

_encoding = tiktoken.get_encoding("cl100k_base")


def _token_count(text: str) -> int:
    return len(_encoding.encode(text))


def test_chunk_text_respects_token_size():
    # Texto repetitivo o suficiente pra gerar múltiplos chunks.
    text = ("Este é um parágrafo de teste sobre RAG e embeddings. " * 40 + "\n\n") * 5

    chunks = chunk_text(text)

    assert len(chunks) > 1
    for chunk in chunks:
        assert _token_count(chunk.content) <= settings.chunk_size


def test_chunk_text_preserves_order_via_index():
    text = ("Parágrafo número um.\n\n" * 30) + ("Parágrafo número dois.\n\n" * 30)

    chunks = chunk_text(text)

    indices = [c.index for c in chunks]
    assert indices == list(range(len(chunks)))


def test_chunk_text_empty_string_returns_empty_list():
    assert chunk_text("") == []


def test_chunk_text_short_text_returns_single_chunk():
    text = "Um texto curto que cabe num único chunk."

    chunks = chunk_text(text)

    assert len(chunks) == 1
    assert chunks[0].content == text
    assert chunks[0].index == 0
