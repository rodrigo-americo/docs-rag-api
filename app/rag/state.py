from typing import TypedDict

from app.rag.retrieval import RetrievedChunk


class RagState(TypedDict):
    question: str
    top_k: int
    rewritten_question: str | None
    retrieved_chunks: list[RetrievedChunk]
    retry_count: int
    answer: str | None
    answerable: bool | None
    citations: list[RetrievedChunk]
