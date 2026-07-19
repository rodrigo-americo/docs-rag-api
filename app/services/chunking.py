from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import settings


@dataclass(frozen=True)
class TextChunk:
    """Chunk de texto pronto pra embedding.

    `index` é a posição original — preservada pra reconstruir ordem
    do documento se algum dia for útil (citation com page hint, etc).
    """
    index: int
    content: str


def chunk_text(text: str) -> list[TextChunk]:
    """Divide texto em chunks conforme config.

    chunk_size e chunk_overlap vêm do Settings e são medidos em tokens
    (encoding cl100k_base, usado por text-embedding-3-small) — não em
    caracteres. Em entrevista, defender:
    'chunk_size=700 tokens é sweet spot pra text-embedding-3-small:
    pequeno o suficiente pra retrieval ter granularidade, grande o
    suficiente pra carregar contexto completo de uma ideia.'
    """
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        # Tenta cada separador em ordem, do mais estrutural ao mais genérico:
        # só desce de parágrafo pra linha/frase/palavra se o nível acima
        # ainda exceder chunk_size.
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    pieces = splitter.split_text(text)
    return [TextChunk(index=i, content=piece) for i, piece in enumerate(pieces)]