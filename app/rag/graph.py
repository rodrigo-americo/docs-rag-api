from collections.abc import Callable

from langgraph.graph import END, StateGraph
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.rag.retrieval import search_similar_chunks
from app.rag.state import RagState
from app.services.chat import ChatProvider
from app.services.embedding import EmbeddingProvider


def make_retrieve_node(
    session: AsyncSession,
    embedding_provider: EmbeddingProvider,
) -> Callable:
    async def retrieve(state: RagState) -> dict:
        question = state["rewritten_question"] or state["question"]
        query_embedding = await embedding_provider.embed_query(question)
        chunks = await search_similar_chunks(
            session=session,
            query_embedding=query_embedding,
            top_k=state["top_k"],
        )
        return {"retrieved_chunks": chunks}

    return retrieve


def decide_after_retrieve(state: RagState) -> str:
    if state["retry_count"] >= settings.max_rewrite_attempts:
        return "generate_answer"
    if not state["retrieved_chunks"]:
        return "rewrite_query"
    best_similarity = max(c.similarity for c in state["retrieved_chunks"])
    if best_similarity < settings.retrieval_quality_threshold:
        return "rewrite_query"

    return "generate_answer"


def make_rewrite_node(chat_provider: ChatProvider) -> Callable:
    async def rewrite_query(state: RagState) -> dict:
        prompt = (
            f"A pergunta abaixo não encontrou resultados relevantes na busca. "
            f"Reformule-a de forma mais direta ou com sinônimos, mantendo a mesma intenção.\n\n"
            f"Pergunta original: {state['question']}"
        )

        rewritten = await chat_provider.complete(prompt)
        return {
            "rewritten_question": rewritten,
            "retry_count": state["retry_count"] + 1,
        }

    return rewrite_query


def make_generate_node(chat_provider: ChatProvider) -> Callable:
    async def generate_answer(state: RagState) -> dict:
        context = "\n\n".join(chunk.content for chunk in state["retrieved_chunks"])
        question = state["rewritten_question"] or state["question"]

        prompt = (
            f"Responda a pergunta usando APENAS o contexto abaixo. "
            f"Se o contexto não contiver a informação necessária, diga que não sabe. "
            f"Não calcule nem combine números de trechos diferentes — cite os valores "
            f"e regras exatamente como aparecem no contexto, sem fazer contas.\n\n"
            f"Contexto:\n{context}\n\n"
            f"Pergunta: {question}"
        )

        answer = await chat_provider.complete(prompt)

        return {
            "answer": answer,
            "citations": state["retrieved_chunks"],
        }

    return generate_answer


def build_graph(
    session: AsyncSession,
    embedding_provider: EmbeddingProvider,
    chat_provider: ChatProvider,
):
    graph = StateGraph(RagState)

    graph.add_node("retrieve", make_retrieve_node(session, embedding_provider))
    graph.add_node("rewrite_query", make_rewrite_node(chat_provider))
    graph.add_node("generate_answer", make_generate_node(chat_provider))
    graph.set_entry_point("retrieve")
    graph.add_conditional_edges(
        "retrieve",
        decide_after_retrieve,
        {
            "rewrite_query": "rewrite_query",
            "generate_answer": "generate_answer",
        },
    )
    graph.add_edge("rewrite_query", "retrieve")
    graph.add_edge("generate_answer", END)

    return graph.compile()
