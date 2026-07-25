from fastapi import APIRouter, Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.db import get_db_session
from app.core.rate_limit import limiter
from app.rag.graph import build_graph
from app.rag.state import RagState
from app.schemas.query import Citation, QueryRequest, QueryResponse
from app.services.chat import ChatProvider, get_chat_provider
from app.services.embedding import EmbeddingProvider, get_embedding_provider

router = APIRouter(prefix="/query", tags=["query"])


def get_chat() -> ChatProvider:
    """Dependency pra injetar o chat provider — facilita override em testes."""
    return get_chat_provider()


def get_embedding() -> EmbeddingProvider:
    """Dependency pra injetar o embedding provider — facilita override em testes."""
    return get_embedding_provider()


@router.post(
    "",
    response_model=QueryResponse,
    summary="Responde uma pergunta com base nos documentos indexados",
)
@limiter.limit(settings.rate_limit_query)
async def query(
    request: Request,
    body: QueryRequest,
    session: AsyncSession = Depends(get_db_session),
    embedding_provider: EmbeddingProvider = Depends(get_embedding),
    chat_provider: ChatProvider = Depends(get_chat),
) -> QueryResponse:
    initial_state: RagState = {
        "question": body.question,
        "top_k": body.top_k,
        "rewritten_question": None,
        "retrieved_chunks": [],
        "retry_count": 0,
        "answer": None,
        "citations": [],
    }
    compiled_graph = build_graph(
        session=session,
        embedding_provider=embedding_provider,
        chat_provider=chat_provider,
    )
    final_state = await compiled_graph.ainvoke(initial_state)
    return QueryResponse(
        answer=final_state["answer"],
        citations=[Citation.from_chunk(chunk) for chunk in final_state["citations"]],
    )
