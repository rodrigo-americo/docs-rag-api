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

> **Segurança:** configure um spending limit na sua chave da OpenAI antes de
> usar — não há autenticação nesta API (ver [docs/seguranca.md](docs/seguranca.md)).

A API estará disponível em `http://localhost:8000`.

---

## Exemplo de uso

**Ingerir um documento:**
```bash
curl -X POST http://localhost:8000/documents/ingest \
  -F "file=@contrato.pdf" \
  -F "title=Contrato de Serviço"
```

A resposta chega imediatamente, antes do processamento pesado terminar —
`chunks_created` é sempre `0` neste momento, e `status` é `pending`:

```json
{
  "document_id": "3f1a...",
  "title": "Contrato de Serviço",
  "chunks_created": 0,
  "status": "pending"
}
```

**Consultar se o processamento terminou:**
```bash
curl http://localhost:8000/documents/3f1a...
```

```json
{
  "id": "3f1a...",
  "title": "Contrato de Serviço",
  "source_filename": "contrato.pdf",
  "status": "indexed",
  "created_at": "2026-07-27T03:00:00Z"
}
```

`status` evolui `pending` → `indexed` (sucesso) ou `pending` → `failed`
(erro de parse, tipo de arquivo não suportado, conteúdo suspeito ou falha
da OpenAI durante o embedding — detalhes em
[docs/decisoes-tecnicas.md](docs/decisoes-tecnicas.md)).

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
| `POST` | `/documents/ingest` | Ingere PDF ou Markdown (máx. 10 MB / 50 páginas). Responde imediatamente com `status=pending`; processamento pesado roda em background |
| `GET` | `/documents/{id}` | Consulta um documento e seu status de processamento (`pending`/`indexed`/`failed`) |
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
- **Rate limiting:** slowapi, por IP, em `/query` e `/documents/ingest`

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

## Mais detalhes

Este README cobre o essencial. Para o raciocínio completo por trás das
decisões do projeto:

- **[docs/decisoes-tecnicas.md](docs/decisoes-tecnicas.md)** — por que
  pgvector, HNSW, LangGraph, ingest assíncrono, prompt injection,
  rate limiting, e outras escolhas de arquitetura.
- **[docs/seguranca.md](docs/seguranca.md)** — o que a API protege
  (e o que não protege, por decisão de escopo).
- **[docs/avaliacao.md](docs/avaliacao.md)** — dataset, métricas
  (Recall@5, Faithfulness, latência, custo), viés do juiz de
  Faithfulness, e a calibração do threshold de retrieval.
