# Testes

[← Voltar ao README](../README.md)

## Isolamento de custo real de OpenAI

Por padrão, toda a suite roda sem tocar a API real da OpenAI nem gastar
custo: `tests/conftest.py` sobrescreve as dependencies de chat e embedding
(`app.dependency_overrides`) forçando `FakeChatProvider`/
`FakeEmbeddingProvider` em todo teste. `FakeEmbeddingProvider` é
determinístico (hash do texto → seed → vetor normalizado), então mesma
entrada sempre gera mesmo embedding, sem exigir rede.

Isso cobre a lógica do pipeline (retrieval, rewrite, grafo, endpoints),
mas nunca exercita o formato de resposta real da OpenAI — schema exato,
erros reais do SDK. Para isso existe a camada VCR, abaixo.

## VCR (pytest-recording) para o contrato real da API

`pytest-recording`/`vcrpy` gravam a chamada HTTP real uma única vez e
reproduzem esse cassete nas rodadas seguintes, sem rede nem custo. Dois
testes usam isso hoje, marcados com `@pytest.mark.vcr`:

- `test_ingest_txt_succeeds_with_real_openai_embedding`
  (`tests/test_ingest_endpoint.py`) — embedding real via
  `real_embedding_provider`.
- `test_query_returns_answer_with_real_openai`
  (`tests/test_query_endpoint.py`) — embedding **e** chat reais via
  `real_embedding_provider`/`real_chat_provider`. O ingest desse teste usa
  embedding fake de propósito (só o `/query` em si precisa do contrato
  real), mantendo o cassete pequeno.

Os cassetes ficam em `tests/cassettes/` (ignorado no `.gitignore` —
não versionado). `vcr_config` (fixture `scope="module"` em
`conftest.py`) redige os headers `authorization` e `openai-organization`
antes de gravar, então a chave real nunca fica no cassete mesmo que ele
seja versionado por engano. `record_mode="once"`: grava só se o cassete
não existir; se existir, sempre reproduz — nunca faz uma chamada real de
novo sem apagar o arquivo manualmente.

**Para gravar (ou re-gravar) os cassetes**, apague o(s) arquivo(s) em
`tests/cassettes/` e rode a suite normalmente com uma
`OPENAI_API_KEY` válida no `.env`:

```bash
rm tests/cassettes/test_query_endpoint/test_query_returns_answer_with_real_openai.yaml
pytest tests/test_query_endpoint.py::test_query_returns_answer_with_real_openai
```

Confirmado que o replay não depende de rede nem de chave válida: a suite
passa normalmente mesmo com `OPENAI_API_KEY` inválida, desde que o
cassete já exista.

## Banco de teste, separado do banco de dev

Os testes rodam contra Postgres real, não mockado — `_clean_tables`
(`conftest.py`, autouse) faz `TRUNCATE TABLE chunks, documents CASCADE`
antes de **cada** teste, pra isolar um teste do outro. Rodar isso contra o
banco de dev/produção apagaria dados reais sem aviso.

Por isso `tests/conftest.py` monta o engine de teste a partir de
`TEST_DATABASE_URL` (não `DATABASE_URL`), com duas checagens que falham
alto na importação do conftest — antes de qualquer teste rodar ou tocar o
banco:

- `TEST_DATABASE_URL` vazia → `RuntimeError` (não faz fallback silencioso
  para `DATABASE_URL`).
- `TEST_DATABASE_URL == DATABASE_URL` → `RuntimeError` (protege contra
  copiar/colar errado no `.env`).

`TEST_DATABASE_URL` aponta para um segundo banco (`docsrag_test`) no
**mesmo** container/instância Postgres do `docker-compose.yml` — não um
serviço separado. Esse banco é criado automaticamente por
`docker/postgres-init/01-create-test-db.sh`, que o Postgres executa uma
única vez, na primeira inicialização do container (volume vazio). Se o
container já existe de antes desse script ter sido adicionado (volume
`postgres_data` não está mais vazio), crie o banco manualmente uma vez:

```bash
docker exec docs-rag-postgres psql -U docsrag -d docsrag -c "CREATE DATABASE docsrag_test"
```

`docsrag_test` precisa do mesmo schema do banco de dev — migrations não
rodam automaticamente nele (`alembic/env.py` sempre lê
`settings.database_url_sync`, ou seja, `DATABASE_URL`, não
`TEST_DATABASE_URL`). Aplique sobrepondo a env var só nesse comando:

```bash
DATABASE_URL=$TEST_DATABASE_URL uv run alembic upgrade head
```

Precisa repetir isso a cada nova migration adicionada ao projeto — o CI
(`.github/workflows/ci.yml`) já faz isso automaticamente, como um step
separado ("Run migrations (banco de teste)"), contra um segundo container
Postgres efêmero (`postgres_test`, porta 5433) só para esse propósito.

## Cobertura de código: 100%, combinada entre dois modos

`pytest-cov` (`[tool.coverage.*]` em `pyproject.toml`) mede a suite com
`fail_under = 100` e `branch = true`. A suite roda em dois modos que juntos
cobrem 100% do código — nenhum dos dois sozinho chega lá, porque parte do
código só é alcançável num modo específico:

- **`queue_backend=inline`** (padrão): endpoints, grafo RAG, parsers,
  providers fake, branches de erro do ingest.
- **`queue_backend=redis`** (`test_worker_integration.py`, `pytest.mark.skipif`
  se `queue_backend != "redis"`): o worker real (`app/worker.py`)
  consumindo de um Redis de verdade, incluindo o branch de produção
  `session_factory=None` (que cria seu próprio engine a partir de
  `settings.database_url`) e o branch de fila vazia (`dequeue()` retornando
  `None` após o timeout de 5s).

Localmente, os dois comandos:

```bash
pytest --ignore=tests/test_worker_integration.py --cov=app --cov-fail-under=0
QUEUE_BACKEND=redis pytest tests/test_worker_integration.py --cov=app --cov-append --cov-report=term-missing
```

`--cov-fail-under=0` no primeiro comando desliga o gate ali — cobertura é
cumulativa (`--cov-append` no segundo), e boa parte do código só é
alcançável no modo Redis, então falhar antes de somar os dois rejeitaria
até uma suite que cobre 100% do sistema inteiro. O CI (`.github/workflows/ci.yml`)
já roda exatamente essa sequência, como dois steps separados. O jeito mais
simples de rodar os dois de uma vez, com a mesma infraestrutura do CI
(Postgres + Redis), é via Docker — ver seção abaixo.

### Exclusões documentadas (`exclude_lines`)

Três categorias, cada uma justificada linha a linha, não um jeito de
esconder lógica não testada:

- **Entrypoint de processo** (`if __name__ == "__main__":`) — nunca
  executa sob pytest, só quando o arquivo roda como script.
- **Corpo de método de `typing.Protocol`** (`...` isolado, ou
  `def foo(...): ...` numa linha só) — assinatura de interface pra
  structural typing, nunca instanciado nem chamado em runtime.
- **`# pragma: no cover` pontual, com comentário explicando o motivo** —
  ver "Bug conhecido do coverage.py" abaixo. Cada ocorrência linka de
  volta pra esta seção.

### Bug conhecido do coverage.py: funções que executam mas não são marcadas

Alguns handlers/métodos específicos nunca são marcados como cobertos pelo
`coverage.py`, mesmo comprovadamente executando com sucesso — investigado
a fundo, não é falta de teste real. Confirmado com `print()` de debug
dentro do corpo da função (a mensagem aparece na saída do teste,
executado com `-s`) enquanto o relatório de cobertura simultaneamente
marca a mesma linha como não executada. Reproduzido de forma consistente
tanto em Windows quanto em Linux (dentro do container `test`, ver
abaixo) — não é peculiaridade de plataforma.

Afeta especificamente:

- `list_documents`, `get_document`, `delete_document`, `ingest_document`
  em `app/api/documents.py` — mesmo padrão estrutural (`async def` +
  `Depends` + `raise HTTPException`) funciona corretamente em
  `app/api/health.py` e `app/api/query.py`, então não é sobre FastAPI/
  Starlette/async em geral.
- `OpenAIChatProvider.complete_structured` (`app/services/chat.py`) —
  `OpenAIChatProvider.complete`, no mesmo arquivo, é medido normalmente.
- `OpenAIEmbeddingProvider.embed_texts`/`embed_query`
  (`app/services/embedding.py`).
- O `return text` final de `PdfParser.parse` (`app/services/parsers/pdf.py`).

Hipóteses descartadas depois de testar cada uma: bytecode obsoleto
(`__pycache__` limpo manualmente), branch vs. line coverage (mesmo
resultado nos dois modos), `sys.monitoring`/`core=sysmon` (não suporta
branch coverage nesta versão do coverage.py), identidade do objeto de
código do handler (`__code__.co_filename`/`co_firstlineno` batem com o
arquivo fonte real), decorators (`@limiter.limit` está presente tanto em
`documents.py` quanto em `query.py`, que funciona), middleware
(`SlowAPIMiddleware` é global, afeta os dois arquivos igualmente). A causa
exata dentro do coverage.py não foi identificada — as linhas afetadas
têm `# pragma: no cover` com um comentário linkando pra este parágrafo,
cada teste que exercita esse código de verdade continua existindo e
passando normalmente (não foram removidos).

## Rodando a suite completa via Docker (recomendado)

```bash
docker compose --profile test build test
docker compose --profile test run --rm test
```

`--profile test`: o serviço `test` não sobe com `docker compose up`
normal (não é um servidor de vida longa) — só quando explicitamente
pedido. Builda a partir do estágio `test` do `Dockerfile` (camada extra
sobre a imagem de produção, só com o grupo dev instalado — `pytest`,
`ruff` etc., que a imagem de produção deliberadamente não tem), conecta
no mesmo `postgres`/`redis` do compose (rede interna), e roda os dois
comandos de cobertura em sequência (migrations nos dois bancos, depois os
dois modos de teste com `--cov-append`).

Por que rodar assim, além de bater com o ambiente do CI: eliminou um
flake real que existia tanto em Windows quanto em Linux — sockets
assíncronos (Postgres via `asyncpg`, HTTP via `httpx`) que não fecham a
tempo do processo `pytest` terminar geram `ResourceWarning`, e
`filterwarnings = ["error"]` (`pyproject.toml`) promovia isso a erro
fatal intermitente, derrubando testes aleatórios sem relação com o
warning em si. Duas causas raiz identificadas e corrigidas:

1. `langsmith.trace()` real (não mockado) abre uma conexão HTTP em
   background mesmo sem `LANGSMITH_API_KEY` configurada — o teste que
   exercita `LANGSMITH_TRACING=true` agora mocka `app.api.query.trace`
   em vez de deixá-lo rodar de verdade (`test_query_includes_trace_id_when_langsmith_tracing_enabled`).
2. `_test_engine` (`conftest.py`) nunca era descartado ao fim da suite —
   `_dispose_test_engine` (fixture `scope="session"`, autouse) chama
   `await _test_engine.dispose()` no fim de toda a suite.

Depois dessas duas correções, sobrou flakiness residual e menor, rotativo
entre testes diferentes a cada rodada — isolado como acúmulo de conexões
`asyncpg` não fechadas a tempo (`NullPool` trocando de event loop a cada
teste, uma característica conhecida dessa combinação de stack, não um
teste específico com bug). `filterwarnings` ganhou uma exceção pontual
para `ResourceWarning`, com o registro completo da investigação como
comentário — outros warnings (deprecation etc.) continuam sendo
promovidos a erro normalmente.

## Outras infraestruturas relevantes

- **`ASGITransport`** — o `client` (fixture) roda a app via
  `httpx.AsyncClient` + `ASGITransport`, sem subir um servidor de verdade.
  Isso também faz `BackgroundTasks` rodar de forma síncrona, antes do
  `client.post()` retornar — diferente de produção, onde roda depois. Por
  isso testes de ingest já conseguem afirmar o resultado final
  (`status=indexed`/`failed`) sem precisar de polling.
- **Rate limiting desligado em teste** (`app.state.limiter.enabled =
  False`) — `ASGITransport` faz todas as requisições de teste
  compartilharem o mesmo IP; com o limiter ligado, a suite se auto-limitaria.
- **`test_worker_integration.py`** — os dois testes desse arquivo
  exercitam um worker real consumindo de Redis de verdade e são
  `SKIPPED` por padrão fora do ambiente de CI/Docker (exigem Redis
  rodando). `wait_until` (`conftest.py`) faz polling do estado do
  documento para esses casos, já que o processamento é genuinamente
  assíncrono (diferente do caminho `ASGITransport` acima).
