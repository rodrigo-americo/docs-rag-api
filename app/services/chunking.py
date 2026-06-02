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

    chunk_size e chunk_overlap vêm do Settings. Em entrevista, defender:
    'chunk_size=700 tokens é sweet spot pra text-embedding-3-small:
    pequeno o suficiente pra retrieval ter granularidade, grande o
    suficiente pra carregar contexto completo de uma ideia.'
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        # Ordem dos separadores importa — primeiro o mais "grosso".
        # Quebra em parágrafo se possível; só desce pra frase/palavra
        # se um parágrafo único exceder chunk_size.
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    
    pieces = splitter.split_text(text)
    return [TextChunk(index=i, content=piece) for i, piece in enumerate(pieces)]