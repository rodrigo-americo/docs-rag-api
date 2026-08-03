# Seed do banco de avaliação

`docsrag_eval_seed.dump` é um dump (`pg_dump --format=custom`) do Postgres
já com os 3 documentos de `evals/documents/` ingeridos — mesmo estado que
gerou os `expected_chunk_id` em `evals/dataset.json`.

## Por que isso existe

`Chunk.id` é `uuid.uuid4()` — cada reingestão (`evals/setup_dataset.py`)
gera IDs novos, diferentes dos que `evals/dataset.json` espera. Sem este
dump, qualquer ambiente novo (clone, container recriado, ou até
`tests/conftest.py`, que faz `TRUNCATE chunks, documents CASCADE` antes de
cada teste porque roda contra o mesmo Postgres do `.env`) quebra o eval:
Recall@5 cai pra 0% não porque o retrieval piorou, mas porque nenhum
`expected_chunk_id` bate mais com o banco.

## Restaurar

```
docker exec -i docs-rag-postgres pg_restore -U docsrag -d docsrag --clean --if-exists < evals/db_seed/docsrag_eval_seed.dump
```

Depois de restaurar, **não rode `uv run pytest` antes do eval** — o
`TRUNCATE` do `conftest.py` apaga os dados restaurados (ver
`docs/roadmap.md` para o contexto completo desse problema). Se isso
acontecer, restaure de novo.

## Atualizar o dump

Só necessário se os PDFs em `evals/documents/` mudarem:

```
uv run python -m evals.setup_dataset
docker exec docs-rag-postgres sh -c 'pg_dump -U docsrag -d docsrag --format=custom --file=/tmp/docsrag_eval_seed.dump'
docker cp docs-rag-postgres:/tmp/docsrag_eval_seed.dump evals/db_seed/docsrag_eval_seed.dump
```

`evals/dataset.json` precisa ser remapeado para os novos `chunk_id` nesse
caso (o `chunk_index` de cada chunk é estável entre ingestões do mesmo
PDF, mas o `chunk_id` não é).
