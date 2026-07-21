# docs-rag-api

API REST em FastAPI que ingere documentos (PDF/Markdown), indexa em PostgreSQL com pgvector, e responde perguntas em linguagem natural com **citação da fonte** — usando LangChain + LangGraph.

<!-- TODO: adicionar GIF de demo aqui -->

---

## Como rodar

**Pré-requisitos:** Docker e uma chave da OpenAI.

```bash
git clone https://github.com/rodrigo-americo/docs-rag-api
cd docs-rag-api
cp .env.example .env          # preencher OPENAI_API_KEY
docker compose up --build
```

A API estará disponível em `http://localhost:8000`.

---

## Exemplo de uso

**Ingerir um documento:**
```bash
curl -X POST http://localhost:8000/documents/ingest \
  -F "file=@contrato.pdf" \
  -F "title=Contrato de Serviço"
```

```json
{
  "document_id": "3f1a...",
  "title": "Contrato de Serviço",
  "chunks_created": 42,
  "status": "indexed"
}
```

**Fazer uma pergunta:**
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Qual o prazo de entrega?", "top_k": 5}'
```

```json
{
  "answer": "O prazo de entrega é de 10 dias úteis...",
  "citations": [
    {
      "document_id": "3f1a...",
      "chunk_id": "a9c2...",
      "snippet": "O prazo de entrega acordado é de 10 dias úteis a partir...",
      "score": 0.91
    }
  ],
  "trace_id": "langsmith-..."
}
```

---

## Endpoints

| Método | Rota | Descrição |
|--------|------|-----------|
| `POST` | `/documents/ingest` | Ingere PDF ou Markdown (máx. 10 MB / 50 páginas) |
| `POST` | `/query` | Pergunta em linguagem natural com citação |
| `GET` | `/documents` | Lista documentos indexados |
| `DELETE` | `/documents/{id}` | Remove documento e seus chunks |
| `GET` | `/health` | Liveness — processo vivo |
| `GET` | `/ready` | Readiness — conexão com Postgres |

---

## Stack

- **API:** FastAPI + Pydantic v2
- **Banco:** PostgreSQL 16 + pgvector
- **ORM:** SQLAlchemy 2.0 (async) + Alembic
- **LLM/Embeddings:** OpenAI (`gpt-4o-mini` + `text-embedding-3-small`)
- **Orquestração RAG:** LangChain + LangGraph
- **Observabilidade:** LangSmith (tracing) + logging estruturado em JSON
- **Testes:** pytest + pytest-asyncio + httpx + pytest-recording (VCR)
- **CI:** GitHub Actions (lint + testes em cada PR)

---

## Arquitetura

```
Client → FastAPI → Ingest Service → PostgreSQL + pgvector
                 ↘ RAG Graph (LangGraph) ↗         ↕
                                             OpenAI API
```

### Fluxo do RAG Graph

```
[query] → [retrieve top-k] → [check quality]
                                    ↓
                   ┌────────────────┴────────────────┐
                   ▼ scores baixos                   ▼ ok
             [rewrite query]                  [build context]
             [retrieve again] ──────────────▶ [generate answer]
                                                     ↓
                                            [resposta + citações]
```

O branch de rewrite dispara quando nenhum chunk retornado ultrapassa o threshold de similaridade — em vez de gerar uma resposta com retrieval ruim, o grafo reformula a pergunta e tenta novamente.

---

## Decisões técnicas

**pgvector em vez de Chroma/Pinecone**
Mantém tudo no Postgres, sem serviço externo adicional. É o que empresas usam em produção quando o volume não justifica um banco vetorial dedicado.

**HNSW em vez de IVFFlat**
HNSW não precisa de treino prévio — performa bem desde o primeiro vetor. IVFFlat exige milhares de vetores existentes para dar resultado decente.

**LangGraph em vez de chain linear**
Permite branching com estado (retry de query com rewrite). Demonstra um padrão real cobrado em vagas de AI em 2026, não só uma sequência simples de chamadas.

**VCR (pytest-recording) em vez de mocks manuais**
Grava as chamadas reais à OpenAI uma vez e as replica nos testes. Mais robusto que mocks frágeis e garante que o CI rode sem custo de API.

**Recursive splitter em vez de semantic chunking**
Ganho marginal do semantic chunking não justifica a complexidade no MVP. O baseline é honesto e suficiente para o volume esperado.

---

## Avaliação

Dataset: 26 perguntas sobre 3 documentos sintéticos (contrato de serviço,
política de reembolso, manual de onboarding) com `expected_chunk_id`.
22 perguntas têm resposta nos documentos (algumas com vocabulário
propositalmente distante do texto original, para forçar o branch de
rewrite do grafo); 4 são propositalmente fora de escopo — informação que
não existe em nenhum documento — para medir se o sistema reconhece que
não sabe em vez de inventar uma resposta.

| Métrica | Resultado |
|---------|-----------|
| Recall@5 (perguntas respondíveis) | 100% (22/22) |
| Rewrite disparado | 17/26 perguntas |
| Recusa correta (fora de escopo) | 100% (4/4) |
| Faithfulness | 0.95 (21 respostas não-recusa) |
| Latência p50 | 3.87s |
| Latência p95 | 6.87s |
| Custo médio / query | US$ 0.00010 |

Medido com `gpt-4o-mini` + `text-embedding-3-small`, `chunk_size=150` tokens
(reduzido só para a avaliação — os documentos sintéticos são curtos demais
para gerar múltiplos chunks com o `chunk_size=700` padrão). Faithfulness
avaliada por um segundo LLM-juiz, dado o contexto recuperado e a resposta
gerada, e calculada apenas sobre respostas que tentaram afirmar algo com
base no contexto — uma recusa correta ("não sei") não é falta de fidelidade,
é o comportamento esperado, e incluí-la penalizaria a métrica injustamente.
O valor oscila cerca de ±0.02 entre execuções: o juiz é o próprio
`gpt-4o-mini`, e LLM-as-judge não é determinístico — a mesma resposta
correta ocasionalmente recebe 0.5 em vez de 1.0 numa pergunta de fronteira.
Isso é ruído de medição, não um bug do sistema avaliado.

O prompt de geração também instrui o modelo a não combinar números de
trechos diferentes (ex: taxa de multa de um chunk + valor total de outro) —
LLMs erram aritmética com frequência maior do que aparentam, e esse tipo de
inferência silenciosa é mais difícil de auditar do que citar os valores
como aparecem no texto.

Para rodar a avaliação (requer `EMBEDDING_PROVIDER=openai` e
`CHAT_PROVIDER=openai` no `.env`, com uma `OPENAI_API_KEY` real):

```bash
CHUNK_SIZE=150 uv run python -m evals.setup_dataset   # ingere os documentos uma vez
CHUNK_SIZE=150 uv run python -m evals.run_eval
```
