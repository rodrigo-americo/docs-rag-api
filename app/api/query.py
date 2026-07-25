from fastapi import APIRouter, Depends, Request
from slowapi.util import get_remote_address
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.db import get_db_session
from app.core.logging import get_logger
from app.core.rate_limit import limiter
from app.core.sanitize import strip_html
from app.rag.graph import build_graph
from app.rag.state import RagState
from app.schemas.query import Citation, QueryRequest, QueryResponse
from app.services.chat import ChatProvider, get_chat_provider
from app.services.embedding import EmbeddingProvider, get_embedding_provider

log = get_logger(__name__)

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
        "answerable": None,
        "citations": [],
    }
    compiled_graph = build_graph(
        session=session,
        embedding_provider=embedding_provider,
        chat_provider=chat_provider,
    )
    final_state = await compiled_graph.ainvoke(initial_state)

    if final_state["retry_count"] >= settings.max_rewrite_attempts:
        # Sinal de segurança, não de erro: uma pergunta difícil bate o
        # limite de vez em quando, mas o mesmo IP fazendo isso repetidas
        # vezes pode indicar sondagem adversarial do sistema de retrieval.
        # Agregação por IP fica pra quando houver um lugar pra consultar
        # isso (ver issue de observabilidade) — por ora, só garante que o
        # dado existe no log estruturado.
        log.warning(
            "security.rewrite_limit_reached",
            client_ip=get_remote_address(request),
            retry_count=final_state["retry_count"],
            question=body.question,
        )

    if final_state["answerable"] is False:
        # Mesmo raciocínio do log acima: uma recusa isolada é normal, o
        # mesmo IP recusando repetidamente pode indicar tentativa de
        # sondar informação fora do escopo dos documentos. Usa o campo
        # estruturado direto — nada de casar frase de recusa em texto livre.
        log.warning(
            "security.answer_refused",
            client_ip=get_remote_address(request),
            question=body.question,
        )

    return QueryResponse(
        answer=strip_html(final_state["answer"]),
        answerable=final_state["answerable"],
        citations=[Citation.from_chunk(chunk) for chunk in final_state["citations"]],
    )
