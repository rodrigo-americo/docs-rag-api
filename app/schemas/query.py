import uuid

from pydantic import BaseModel, Field

from app.core.config import settings
from app.core.sanitize import strip_html
from app.rag.retrieval import RetrievedChunk


class QueryRequest(BaseModel):
    question: str = Field(..., description="Pergunta feita pelo usuário")
    top_k: int = Field(
        settings.retrieval_top_k, description="Número de chunks buscados como contexto"
    )


class Citation(BaseModel):
    document_id: uuid.UUID = Field(..., description="ID do documento de origem")
    chunk_id: uuid.UUID = Field(..., description="ID do chunk de origem")
    snippet: str = Field(..., description="Trecho do chunk que embasou a resposta")
    score: float = Field(..., description="Proximidade da pergunta, sendo 1 o maior valor possível")

    @classmethod
    def from_chunk(cls, chunk: RetrievedChunk) -> "Citation":
        return cls(
            document_id=chunk.document_id,
            chunk_id=chunk.chunk_id,
            snippet=strip_html(chunk.content),
            score=chunk.similarity,
        )


class QueryResponse(BaseModel):
    answer: str = Field(..., description="Resposta gerada com base nos documentos indexados")
    answerable: bool | None = Field(
        None,
        description="False se o sistema recusou responder por falta de informação "
        "nos documentos — sinal estruturado, não depende de interpretar o texto de "
        "answer para saber se foi uma recusa.",
    )
    citations: list[Citation] = Field(..., description="Chunks usados como fonte da resposta")
    trace_id: str | None = Field(
        None, description="ID do trace no LangSmith — ainda não implementado"
    )
