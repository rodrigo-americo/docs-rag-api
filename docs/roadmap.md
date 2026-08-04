# Próximos passos

## Espinha dorsal: ablation study do pipeline de retrieval

Em vez de tratar cada melhoria de retrieval como uma entrega isolada,
organizar o trabalho em torno de uma tabela de ablation — cada etapa
isola exatamente uma variável nova em cima da anterior, medida contra o
mesmo harness de avaliação (Recall@5, Faithfulness). Isso troca "eu
implementei X" por "eu medi se X realmente ajuda", que é uma resposta
muito mais forte a qualquer pergunta de banca sobre justificar a
complexidade adicionada — inclusive o resultado "não ajudou o suficiente
pra valer a complexidade" é defensável, se for medido.

| Pipeline | Recall@5 | Faithfulness | custo | tempo
|---|---|---|---|---|
| Dense (baseline inicial) | 96% (22/23) | 1.00 | US$ 0.00046/query | p50 3.36s / p95 10.92s |
| Dense + rewrite | 96% (22/23) | 1.00 | US$ 0.00045/query | p50 2.35s / p95 7.36s |
| **Hybrid (BM25 + dense) — padrão atual** | 100% (23/23) | 1.00 | US$ 0.00075/query | p50 1.77s / p95 3.13s |
| Hybrid + RAPTOR | — | — | | |

Cada linha só é adicionada depois que a anterior está medida e registrada
— nada de implementar a próxima camada antes de ter o número da atual.
Análise detalhada de cada linha já medida (Dense, Dense + rewrite,
Hybrid — incluindo o caso CRI/CRA, calibração do threshold RRF e o fix da
tsquery para perguntas longas) vive em [avaliacao.md](avaliacao.md), não
duplicada aqui.

### Próximo passo

**Linha "Hybrid + RAPTOR"** — plano detalhado completo em
[planos/raptor-chunking.md](planos/raptor-chunking.md). Só entra na
tabela depois que "Hybrid" sozinho já está medido — o ponto inteiro é
ver se a árvore soma valor em cima do hybrid search, não assumir que
soma. Maior escopo e risco da lista.

As colunas acima são preenchidas pela metodologia própria já existente e
documentada (Recall@5 exato por `chunk_id`, Faithfulness com viés de juiz
medido empiricamente, threshold calibrado por sweep — ver
[avaliacao.md](avaliacao.md)). RAGAS foi avaliado como fonte adicional de
métrica (Context Precision, Response Relevancy) mas está bloqueado por
incompatibilidade de dependências — toda versão publicada exige uma
stack LangChain incompatível com a deste projeto (detalhes e caminho
viável não implementado em [avaliacao.md](avaliacao.md#ragas--bloqueado-por-incompatibilidade-de-dependências)).

## Trilha paralela: tipos de arquivo (não bloqueia a tabela acima)

- **Suporte a mais tipos de arquivo (.docx etc.)** — mecânico, o parser
  registry (`app/services/parsers/`) foi desenhado exatamente pra isso
  ser barato: um parser novo + registro em `_PARSERS`, sem tocar em
  `ingest.py`.
- **Chunk size configurável por tipo de parser + sweep empírico** — hoje
  `chunk_size`/`chunk_overlap` são globais (`app/core/config.py`),
  aplicados igual a PDF e texto puro. Duas partes: (a) permitir que cada
  `DocumentParser` declare seu próprio chunk_size/overlap; (b) sweep real
  contra o harness de avaliação (mesmo padrão usado para calibrar
  `retrieval_quality_threshold`) pra descobrir empiricamente qual valor
  funciona melhor por tipo de documento. Depende de ter documentos de
  tipos variados primeiro (item anterior).

## Fora da sequência ativa

- **Chunking incremental com checkpoint** (salvar progresso por chunk e
  retomar de onde parou em caso de falha no embedding) — ideia levantada
  durante o trabalho do worker e adiada. Sem evidência de que reprocessar
  do zero seja caro o suficiente pra justificar a complexidade —
  reavaliar só se isso virar um problema real observado.
