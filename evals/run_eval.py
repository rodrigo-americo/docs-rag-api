"""Avaliação do pipeline de RAG: Recall@5, Faithfulness, Latência e Custo.

Pré-requisito: rodar `uv run python -m evals.setup_dataset` uma vez antes
(ingere os documentos de evals/documents/ e gera os expected_chunk_id que
evals/dataset.json referencia).

Exige EMBEDDING_PROVIDER=openai e CHAT_PROVIDER=openai no .env, com uma
OPENAI_API_KEY real — Faithfulness (LLM-juiz) e Custo/query só fazem
sentido medidos contra o provider real, não o fake.

Uso: uv run python -m evals.run_eval
"""

import asyncio
import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import tiktoken

from app.core.config import settings
from app.core.db import AsyncSessionLocal
from app.rag.graph import build_graph
from app.services.chat import get_chat_provider
from app.services.embedding import get_embedding_provider

DATASET_PATH = Path(__file__).parent / "dataset.json"

# Preços por 1M tokens (gpt-4o-mini, text-embedding-3-small) — ajustar se
# o modelo configurado em Settings for outro.
PRICE_PER_1M_INPUT_TOKENS = 0.150
PRICE_PER_1M_OUTPUT_TOKENS = 0.600
PRICE_PER_1M_EMBEDDING_TOKENS = 0.020

_encoding = tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str) -> int:
    return len(_encoding.encode(text))


@dataclass
class QueryTrace:
    question: str
    # list quando o conteúdo esperado aparece em mais de um chunk do
    # documento-fonte (repetição real do texto) — qualquer um dos ids
    # conta como acerto, não só o primeiro que o dataset registrou.
    expected_chunk_id: str | list[str] | None
    out_of_scope: bool
    retrieved_chunk_ids: list[str]
    answer: str
    answerable: bool | None
    context: str
    retry_count: int
    latency_seconds: float
    input_tokens: int
    output_tokens: int
    embedding_tokens: int


FAITHFULNESS_JUDGE_PROMPT = """\
Você é um avaliador rigoroso de sistemas de RAG. Dado um CONTEXTO e uma \
RESPOSTA gerada a partir dele, julgue se a RESPOSTA é inteiramente \
suportada pelo CONTEXTO (sem inventar informação que não está lá).

Responda apenas com um número entre 0.0 e 1.0, onde:
- 1.0 = toda a resposta é diretamente suportada pelo contexto
- 0.5 = parcialmente suportada, com alguma informação não verificável
- 0.0 = a resposta contradiz o contexto ou inventa informação não presente

Responda APENAS o número, sem explicação.

CONTEXTO:
{context}

RESPOSTA:
{answer}

NOTA:"""


async def _run_single_query(
    question: str, expected_chunk_id: str | list[str] | None, out_of_scope: bool
) -> QueryTrace:
    embedding_provider = get_embedding_provider()
    chat_provider = get_chat_provider()

    async with AsyncSessionLocal() as session:
        compiled_graph = build_graph(
            session=session,
            embedding_provider=embedding_provider,
            chat_provider=chat_provider,
        )

        initial_state = {
            "question": question,
            "top_k": settings.retrieval_top_k,
            "rewritten_question": None,
            "retrieved_chunks": [],
            "retry_count": 0,
            "answer": None,
            "answerable": None,
            "citations": [],
        }

        start = time.perf_counter()
        final_state = await compiled_graph.ainvoke(initial_state)
        latency = time.perf_counter() - start

    context = "\n\n".join(c.content for c in final_state["citations"])
    answer = final_state["answer"] or ""

    return QueryTrace(
        question=question,
        expected_chunk_id=expected_chunk_id,
        out_of_scope=out_of_scope,
        retrieved_chunk_ids=[str(c.chunk_id) for c in final_state["citations"]],
        answer=answer,
        answerable=final_state["answerable"],
        context=context,
        retry_count=final_state["retry_count"],
        latency_seconds=latency,
        input_tokens=_count_tokens(context) + _count_tokens(question),
        output_tokens=_count_tokens(answer),
        embedding_tokens=_count_tokens(question),
    )


async def _judge_faithfulness(trace: QueryTrace) -> float:
    if not trace.context.strip():
        return 0.0

    chat_provider = get_chat_provider()
    prompt = FAITHFULNESS_JUDGE_PROMPT.format(context=trace.context, answer=trace.answer)
    verdict = await chat_provider.complete(prompt)

    try:
        return max(0.0, min(1.0, float(verdict.strip())))
    except ValueError:
        return 0.0


def _recall_at_k(trace: QueryTrace) -> bool:
    expected = trace.expected_chunk_id
    if isinstance(expected, list):
        return any(chunk_id in trace.retrieved_chunk_ids for chunk_id in expected)
    return expected in trace.retrieved_chunk_ids


def _correctly_refused(trace: QueryTrace) -> bool:
    """Lê o campo estruturado `answerable` (app/rag/graph.py:GeneratedAnswer)
    em vez de casar frases de recusa em texto livre — não depende da
    fraseologia exata que o LLM usou pra dizer "não sei"."""
    return trace.answerable is False


async def main() -> None:
    if settings.embedding_provider != "openai" or settings.chat_provider != "openai":
        raise SystemExit(
            "Esta avaliação exige EMBEDDING_PROVIDER=openai e CHAT_PROVIDER=openai "
            "no .env, com uma OPENAI_API_KEY real. Faithfulness e Custo/query não "
            "fazem sentido medidos contra os providers fake."
        )

    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    print(f"Avaliando {len(dataset)} perguntas...\n")

    traces: list[QueryTrace] = []
    for i, item in enumerate(dataset, start=1):
        out_of_scope = item.get("out_of_scope", False)
        trace = await _run_single_query(
            item["question"], item.get("expected_chunk_id"), out_of_scope
        )
        traces.append(trace)

        if out_of_scope:
            hit = "OK" if _correctly_refused(trace) else "MISS"
        else:
            hit = "OK" if _recall_at_k(trace) else "MISS"
        rewrite_tag = f" [rewrite x{trace.retry_count}]" if trace.retry_count > 0 else ""
        print(
            f"[{i}/{len(dataset)}] {hit} ({trace.latency_seconds:.2f}s){rewrite_tag} "
            f"{item['question']}"
        )

    answerable_traces = [t for t in traces if not t.out_of_scope]
    out_of_scope_traces = [t for t in traces if t.out_of_scope]

    recall_hits = sum(_recall_at_k(t) for t in answerable_traces)
    recall_at_5 = recall_hits / len(answerable_traces)

    rewrite_triggers = sum(1 for t in traces if t.retry_count > 0)

    refusal_hits = sum(_correctly_refused(t) for t in out_of_scope_traces)
    refusal_rate = refusal_hits / len(out_of_scope_traces) if out_of_scope_traces else None

    print("\nJulgando faithfulness com LLM-juiz...")
    # Faithfulness só faz sentido pra respostas que tentaram afirmar algo com
    # base no contexto — uma recusa correta ("não sei") não é "infiel", é o
    # comportamento certo, e o LLM-juiz penalizaria injustamente com nota 0.
    scored_traces = [t for t in traces if not _correctly_refused(t)]
    faithfulness_scores = await asyncio.gather(*[_judge_faithfulness(t) for t in scored_traces])
    avg_faithfulness = statistics.mean(faithfulness_scores) if faithfulness_scores else None

    latencies = sorted(t.latency_seconds for t in traces)
    p50 = statistics.median(latencies)
    p95_index = min(len(latencies) - 1, int(len(latencies) * 0.95))
    p95 = latencies[p95_index]

    total_input_tokens = sum(t.input_tokens for t in traces)
    total_output_tokens = sum(t.output_tokens for t in traces)
    total_embedding_tokens = sum(t.embedding_tokens for t in traces)
    total_cost = (
        total_input_tokens * PRICE_PER_1M_INPUT_TOKENS / 1_000_000
        + total_output_tokens * PRICE_PER_1M_OUTPUT_TOKENS / 1_000_000
        + total_embedding_tokens * PRICE_PER_1M_EMBEDDING_TOKENS / 1_000_000
    )
    avg_cost_per_query = total_cost / len(traces)

    print("\n--- Resultado ---")
    print(f"Recall@5:              {recall_at_5:.0%} ({recall_hits}/{len(answerable_traces)})")
    print(f"Rewrite disparado:     {rewrite_triggers}/{len(traces)} perguntas")
    if refusal_rate is not None:
        print(
            f"Recusa correta (OOS):  {refusal_rate:.0%} ({refusal_hits}/{len(out_of_scope_traces)})"
        )
    if avg_faithfulness is not None:
        print(
            f"Faithfulness:          {avg_faithfulness:.2f} "
            f"(sobre {len(scored_traces)} respostas não-recusa)"
        )
    print(f"Latência p50:          {p50:.2f}s")
    print(f"Latência p95:          {p95:.2f}s")
    print(f"Custo médio/query:     US$ {avg_cost_per_query:.5f}")


if __name__ == "__main__":
    asyncio.run(main())
