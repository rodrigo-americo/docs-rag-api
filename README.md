# docs-rag-api — Escopo do Projeto

> Projeto de portfólio para vagas de Backend Python com diferencial AI/LLM.
> Prazo: 4 semanas. Última revisão: maio/2026.

---

## 1. Pitch em uma frase

API REST em FastAPI que ingere documentos (PDF/Markdown), indexa em PostgreSQL com pgvector, e responde perguntas em linguagem natural com **citação da fonte** — usando LangChain + LangGraph, com testes e avaliação automatizada.

---

## 2. Por que esse projeto

Cobre simultaneamente cinco lacunas marcadas como prioridade alta na estratégia de carreira:

| Lacuna | Como é coberta |
|---|---|
| LangChain / RAG | Núcleo do projeto |
| Testes unitários (pytest) | Cobertura do pipeline de ingest e retrieval |
| Docker / CI/CD | `docker-compose` + GitHub Actions rodando os testes |
| Observabilidade | Logs estruturados + LangSmith tracing |
| Mensageria (opcional, fase 2) | Fila para ingest assíncrono de documentos grandes |

Encaixa direto no perfil-alvo: backend Python que sabe integrar LLM. Não exige virar engenheiro de ML.

---

## 3. Stack

- **API:** FastAPI + Pydantic v2
- **Banco:** PostgreSQL 16 + extensão `pgvector`
- **ORM:** SQLAlchemy 2.0 (async) + Alembic para migrações
- **LLM/Embeddings:** OpenAI (`gpt-4o-mini` e `text-embedding-3-small`) — barato e suficiente
- **Orquestração:** LangChain + LangGraph
- **Observabilidade:** LangSmith (tracing) + logging estruturado em JSON
- **Testes:** pytest + pytest-asyncio + httpx + `pytest-recording` (VCR) para chamadas OpenAI
- **Container:** Docker + docker-compose
- **CI:** GitHub Actions rodando lint (ruff) + testes em PR
- **Parsing de PDF:** `pypdf` (suficiente para o escopo)

Justificativa de escolhas não óbvias:
- **pgvector e não Chroma/Pinecone:** reforça perfil backend, mostra Postgres a fundo, é o que empresas usam em produção quando não querem mais um serviço externo.
- **OpenAI e não modelo local:** custo de inferência local em CPU é proibitivo e tira o foco. Embedding pequeno custa centavos.
- **LangGraph e não chain linear:** permite mostrar fluxo com estado, retry e branching — é o que está sendo cobrado em vagas sênior de AI em 2026.
- **HNSW e não IVFFlat:** HNSW não precisa de treino prévio e performa bem desde o primeiro vetor. IVFFlat exige ~milhares de vetores existentes pra dar resultado decente.
- **VCR (pytest-recording) e não mock manual:** grava chamadas OpenAI reais uma vez e replica nos testes. Mais robusto que mocks frágeis, e mostra cuidado de engenharia.

---

## 4. Arquitetura

```
┌─────────────┐      ┌──────────────────────────────────┐
│   Client    │──────│         FastAPI                   │
└─────────────┘      │  ┌────────────────────────────┐  │
                     │  │  /ingest /query /health /ready│ │
                     │  └─────┬──────────┬───────────┘  │
                     │        │          │              │
                     │   ┌────▼────┐ ┌───▼─────────┐    │
                     │   │ Ingest  │ │  RAG Graph  │    │
                     │   │ Service │ │ (LangGraph) │    │
                     │   └────┬────┘ └───┬─────────┘    │
                     └────────┼──────────┼──────────────┘
                              │          │
                              ▼          ▼
                     ┌────────────────────────────┐
                     │  PostgreSQL + pgvector     │
                     │  - documents               │
                     │  - chunks (com embedding)  │
                     └────────────────────────────┘
                              │
                              ▼
                     ┌────────────────┐
                     │  OpenAI API    │
                     │  (embed + LLM) │
                     └────────────────┘
```

### Fluxo do RAG Graph (LangGraph)

```
[query] → [retrieve top-k chunks] → [check retrieval quality]
                                          ↓
                          ┌───────────────┴────────────────┐
                          ▼ (todos scores < threshold)     ▼ (ok)
                    [rewrite query] ──┐            [build context]
                    [retrieve again]  │                    ↓
                          ↓           │            [generate answer]
                          └───────────┘                    ↓
                                                    [response + citations]
```

A escolha do branching é deliberada: **query rewriting quando o retrieval inicial é fraco** é um padrão real, justifica usar grafo com estado em vez de chain linear, e é fácil de demonstrar em entrevista. Vale mais que um "retry com k maior" que quase nunca dispara.

---

## 5. Endpoints

### `POST /documents/ingest`
Recebe arquivo PDF ou markdown, faz chunking, gera embeddings, persiste.

**Request:** `multipart/form-data` com `file` e `title` (opcional).

**Limites:**
- Tamanho máximo: 10 MB
- PDF: até 50 páginas
- Acima disso: retorna `413 Payload Too Large` com mensagem orientando o caminho da fila assíncrona (fase 2).

**Response 202:**
```json
{
  "document_id": "uuid",
  "title": "string",
  "chunks_created": 42,
  "status": "indexed"
}
```

### `POST /query`
Pergunta em linguagem natural; devolve resposta com citações.

**Request:**
```json
{
  "question": "Qual a política de devolução?",
  "top_k": 5,
  "document_ids": ["uuid1", "uuid2"]
}
```

**Response 200:**
```json
{
  "answer": "A política permite devolução em até 30 dias...",
  "citations": [
    {
      "document_id": "uuid1",
      "chunk_id": "uuid",
      "snippet": "trecho exato do documento",
      "score": 0.87
    }
  ],
  "trace_id": "langsmith-id"
}
```

### `GET /documents`
Lista documentos indexados com paginação.

### `DELETE /documents/{id}`
Remove documento e seus chunks.

### `GET /health` (liveness)
Sempre retorna 200 se o processo está vivo. Sem dependências externas.

### `GET /ready` (readiness)
Verifica conexão com Postgres. OpenAI **não** entra no readiness — é validada no startup e logada, mas não no path crítico (evita derrubar o serviço quando a OpenAI tem hiccup).

---

## 6. Modelo de dados

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title TEXT NOT NULL,
    source_filename TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INT NOT NULL,
    content TEXT NOT NULL,
    embedding VECTOR(1536),  -- dimensão do text-embedding-3-small
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Índice HNSW: não precisa de treino prévio, performa bem desde o primeiro vetor.
-- Parâmetros padrão são adequados para o volume esperado do projeto.
CREATE INDEX chunks_embedding_idx ON chunks
    USING hnsw (embedding vector_cosine_ops);

CREATE INDEX chunks_document_id_idx ON chunks(document_id);
```

**Estratégia de chunking** (definida antes de começar para evitar refatoração):
- Recursive character splitter (LangChain)
- `chunk_size = 700` tokens
- `chunk_overlap = 100` tokens
- Separadores em ordem: `\n\n`, `\n`, `. `, ` `

Semantic chunking foi considerado e descartado para o MVP — ganho marginal não justifica complexidade. Fica como bala na manga pro README ("experimentei X, optei por Y porque...").

---

## 7. Estrutura de pastas

```
docs-rag-api/
├── app/
│   ├── api/
│   │   ├── __init__.py
│   │   ├── deps.py              # Dependências FastAPI (DB session, settings)
│   │   ├── documents.py         # Endpoints de documentos
│   │   ├── query.py             # Endpoint de query
│   │   └── health.py            # /health e /ready
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py            # Settings via Pydantic
│   │   ├── logging.py           # JSON logger estruturado
│   │   └── db.py                # SQLAlchemy engine e session
│   ├── models/
│   │   ├── __init__.py
│   │   └── document.py          # Models SQLAlchemy
│   ├── schemas/
│   │   ├── __init__.py
│   │   ├── document.py
│   │   └── query.py             # Pydantic schemas
│   ├── services/
│   │   ├── __init__.py
│   │   ├── ingest.py            # Lógica de ingest (parse, chunk, embed)
│   │   ├── chunking.py          # Estratégia de chunking
│   │   └── embedding.py         # Wrapper OpenAI embeddings
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── graph.py             # LangGraph definition
│   │   ├── nodes.py             # Nós: retrieve, check_quality, rewrite, generate
│   │   ├── prompts.py           # Templates de prompt
│   │   └── retriever.py         # Retriever em cima do pgvector
│   └── main.py                  # FastAPI app
├── tests/
│   ├── conftest.py
│   ├── test_health.py
│   ├── test_ingest.py
│   ├── test_query.py
│   ├── test_chunking.py
│   ├── test_retriever.py
│   ├── cassettes/               # Gravações VCR das chamadas OpenAI
│   └── fixtures/
│       └── sample.pdf
├── evals/
│   ├── dataset.jsonl            # 20 Q&A com expected_chunk_id
│   ├── run_eval.py              # Script de avaliação
│   └── README.md                # Resultados de acurácia
├── migrations/
│   └── alembic/
├── docker-compose.yml
├── Dockerfile
├── .github/
│   └── workflows/
│       └── ci.yml
├── pyproject.toml
├── ruff.toml
├── .env.example
└── README.md
```

---

## 8. Plano de execução (4 semanas)

### Semana 1 — Scaffold + Ingest

**Objetivo:** subir o esqueleto e conseguir indexar um PDF.

- [ ] Repositório criado, `pyproject.toml`, `.gitignore`, `ruff` configurado
- [ ] `docker-compose.yml` com FastAPI + Postgres+pgvector
- [ ] Migrações Alembic criando `documents` e `chunks` (com índice HNSW)
- [ ] `/health` (liveness) e `/ready` (readiness com Postgres)
- [ ] Endpoint `/documents/ingest` funcionando: recebe PDF, valida tamanho, faz chunking, gera embeddings, persiste
- [ ] Limite de 10 MB / 50 páginas com retorno `413` claro
- [ ] Logger estruturado em JSON

**Critério de aceite:** rodar `docker compose up`, fazer `curl -F file=@doc.pdf .../ingest` e ver as linhas no `chunks` com vetores preenchidos. Subir PDF de 200 páginas retorna 413 com mensagem útil.

### Semana 2 — Query com LangGraph

**Objetivo:** responder perguntas com citação, usando branching real no grafo.

- [ ] Retriever em cima do pgvector (similarity search com cosine)
- [ ] LangGraph com nós: `retrieve → check_quality → (rewrite → retrieve) | generate`
- [ ] Nó `check_quality`: se max(scores) < threshold (ex: 0.7), dispara rewrite
- [ ] Nó `rewrite`: usa LLM para reformular a pergunta original com base nos chunks fracos
- [ ] Endpoint `/query` retornando answer + citations
- [ ] LangSmith configurado, `trace_id` no response

**Critério de aceite:** indexar um PDF de 20 páginas, perguntar 5 coisas sobre ele, todas as respostas vêm com citação correta. Pelo menos 1 pergunta vaga deve disparar o branch de rewrite (visível no trace do LangSmith).

### Semana 3 — Testes + Avaliação

**Objetivo:** mostrar que sabe testar e medir.

- [ ] `tests/` com pytest-asyncio cobrindo: ingest, query, chunking, retriever
- [ ] **Estratégia de mock:**
  - VCR (`pytest-recording`) para testes end-to-end de ingest/query
  - Fake embedding determinístico (hash do texto → vetor 1536) para testes unitários de retriever
- [ ] **Qualidade dos testes** (mais importante que % de coverage):
  - Cada endpoint tem ao menos 1 happy path + 1 caso de erro
  - Pipeline de ingest tem teste end-to-end com PDF de fixture
  - `check_quality` do grafo tem teste isolado cobrindo branch de rewrite
- [ ] **Avaliação:**
  - `evals/dataset.jsonl` com 20 Q&A sobre 2-3 documentos, **incluindo `expected_chunk_id`** em cada entrada
  - `evals/run_eval.py` calcula:
    - **Recall@5**: o chunk certo está no top-5 retornado? (objetivo, automatizável)
    - **Faithfulness**: LLM-as-judge avalia se resposta está suportada pelos chunks citados (prompt manual + uso opcional de `ragas`)
    - Latência média (p50, p95)
    - Custo médio em tokens por query
  - Orçamento do eval: limitado a $2 por run (proteção contra loop esquecido)
- [ ] GitHub Actions: lint + testes em cada PR (sem chamadas reais à OpenAI — só VCR cassettes)

**Critério de aceite:** `pytest` verde, CI verde, `evals/README.md` com tabela mostrando recall@5 ≥ 0.8 e faithfulness ≥ 0.9 no dataset.

### Semana 4 — Polimento e narrativa

**Objetivo:** transformar em artefato de portfólio.

- [ ] README principal com:
  - GIF curto mostrando ingest + query
  - Diagrama da arquitetura (ASCII deste doc, ou desenhar no Excalidraw)
  - Seção "Decisões técnicas e trade-offs" — pgvector vs alternativas, HNSW vs IVFFlat, LangGraph vs chain, VCR vs mock, chunking strategy, etc.
  - Resultados da avaliação (recall@5, faithfulness, latência, custo)
  - Como rodar localmente em 3 comandos
- [ ] `make demo` ou script que ingere um documento de exemplo e roda uma query (zero atrito para quem clonar)
- [ ] Post no LinkedIn: "construí um RAG em FastAPI com pgvector — 3 coisas que aprendi"
- [ ] Pinned no GitHub
- [ ] Atualizar LinkedIn na seção Projetos

**Critério de aceite:** mandar o link para 2 amigos devs que não conhecem o projeto, eles entendem o que faz em menos de 1 minuto lendo o README.

---

## 9. Fora de escopo (não tentar fazer)

Lista deliberada do que **NÃO** entra no MVP, para não inflar o projeto:

- ❌ Frontend / GUI — é uma API, ponto
- ❌ Autenticação (JWT, OAuth) — irrelevante para o ponto que se quer provar
- ❌ Multi-tenancy — adiciona complexidade sem ganho narrativo
- ❌ Streaming de resposta (SSE) — bonito, mas não essencial
- ❌ Re-ranking com cross-encoder — complica, ganho marginal
- ❌ Suporte a outros formatos além de PDF/MD (docx, html, etc.)
- ❌ Modelo local (Llama, etc.) — caro em tempo, irrelevante para a vaga-alvo
- ❌ Semantic chunking — recursive splitter é baseline honesto e suficiente

Se sobrar tempo no final, **fase 2** opcional:
- Ingest assíncrono via fila (Celery + Redis ou SQS) para arquivos > 10 MB — cobriria a lacuna de mensageria e justifica o limite atual
- Cache de queries idênticas

---

## 10. Métricas de sucesso

Como saber se o projeto cumpriu o objetivo de portfólio:

1. **Tempo de compreensão:** dev lendo o README entende o projeto em < 1 min
2. **Avaliação documentada:** README tem tabela com recall@5, faithfulness, latência e custo
3. **Reprodutibilidade:** clone + `docker compose up` + `make demo` funciona em qualquer máquina
4. **Conversa de entrevista:** consegue defender por 20 min — decisões técnicas, trade-offs, o que aprendeu, o que faria diferente
5. **Engajamento no LinkedIn:** post sobre o projeto tem ao menos 1 mensagem de recrutador ou dev sênior

---

## 11. O que esse projeto NÃO é

Para evitar o erro do InsightChain:

- Não é um "agregador de coisas que usa LLM"
- Não é um "tool de análise de empresas"
- Não é uma "interface gráfica para chat"

É **uma API de RAG**. O nome do repo, o título do README, o post no LinkedIn e a descrição em entrevista devem dizer exatamente isso. Coerência narrativa é metade do valor do projeto de portfólio.

---

## 12. Primeira ação concreta (faz hoje)

```bash
mkdir docs-rag-api && cd docs-rag-api
git init
echo "# docs-rag-api" > README.md
git add . && git commit -m "init"
gh repo create docs-rag-api --public --source=. --remote=origin --push
```

A partir daí, semana 1 do plano acima.