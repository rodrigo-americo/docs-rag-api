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

A única falha de recall do baseline "Dense" (1/23) é a pergunta sobre CRI
e CRA (siglas exatas) no glossário do BCB: mesmo após 2 rewrites, o
retrieval denso nunca recupera o chunk certo — os 5 chunks retornados são
todos sobre outras modalidades de crédito, vizinhos temáticos que erram o
termo exato. É evidência concreta, não hipotética, do tipo de gap que a
linha "Hybrid (BM25 + dense)" existe para fechar (busca por termo exato
resolveria isso de cara).

O rewrite dispara em 11/34 perguntas (`MAX_REWRITE_ATTEMPTS=2`,
`retrieval_quality_threshold=0.6`) e não muda Recall@5 nem Faithfulness
em relação ao Dense puro — inclusive falha no mesmo caso CRI/CRA, dois
rewrites incluídos. Custo por query ficou estatisticamente igual
(US$ 0.00045 vs 0.00046). A latência medida ficou menor com rewrite (p50
3.36s → 2.35s, p95 10.92s → 7.36s), mas isso é contraintuitivo — rewrite
adiciona 1-2 chamadas de LLM extras por pergunta afetada, então deveria
somar tempo, não reduzir. O grafo não tem nenhum outro loop de retry que
o rewrite estaria evitando (`decide_after_retrieve` só decide entre
`rewrite_query` e `generate_answer`), então não há mecanismo no código
que explique a queda. Cada rodada foi uma única execução do dataset em
momentos diferentes — a diferença é mais provável de ser variância de
latência da API da OpenAI entre chamadas do que um efeito real do
rewrite. Não tratar essa queda de latência como conclusão sem repetir a
medição (múltiplas rodadas, mesma janela de tempo) antes de citá-la.

A linha "Hybrid" fecha o gap: busca por termo exato (BM25, full-text
search do Postgres com `tsvector`/`ts_rank_cd`) rodando em paralelo com o
retrieval denso existente, fundidos por Reciprocal Rank Fusion (RRF,
k=60, sem normalização de score — só posição em cada lista). O caso
CRI/CRA que o Dense sempre errava agora acerta: BM25 encontra o chunk por
match de termo exato mesmo quando o embedding não. Recall@5 vai de 22/23
para 23/23.

Dois ajustes de calibração foram necessários e não são óbvios de antemão:

- **`websearch_to_tsquery` não serve para perguntas longas.** Ele une
  todos os termos com AND — numa pergunta de 11 palavras, nenhum chunk
  contém todas simultaneamente e a busca BM25 retornava 0 resultados
  (incluindo, ironicamente, para a própria pergunta de CRI/CRA). A
  correção foi reescrever a tsquery trocando `&` por `|` (OR entre
  termos): um único termo específico batendo já produz match, e
  `ts_rank_cd` naturalmente ranqueia mais alto quem bate mais termos —
  ver `search_bm25_chunks` em `app/rag/retrieval.py`.
- **O threshold de rewrite precisa de uma escala própria pro RRF.**
  `retrieval_quality_threshold=0.6` foi calibrado para cosine similarity
  (0-1, boa separação entre pergunta respondível e fora-de-escopo). RRF
  produz scores numa faixa muito mais estreita (~0.016-0.033 no sweep das
  34 perguntas do dataset) e, nessa escala, o score não separa
  respondível de fora-de-escopo — o mesmo valor (1/61≈0.01639) aparece
  nos dois grupos. Não existe corte que funcione bem aqui: em vez de
  calibrar um ponto de corte que não existe, `retrieval_quality_threshold_rrf`
  ficou abaixo do mínimo observado (0.005), desligando o gate de score na
  prática — a recusa por falta de contexto passa a ser decidida
  inteiramente pelo `answerable=False` do LLM (`GeneratedAnswer`), que já
  é quem decide isso mesmo quando o retrieval "passa" no threshold mas o
  conteúdo não responde à pergunta.

O efeito colateral da calibração acima: rewrite nunca dispara no modo
Hybrid (0/34, contra 11/34 no Dense), o que também explica a queda de
latência (p50 3.36s → 1.77s) — sem chamada extra de LLM pra reformular. O
custo por query subiu (US$ 0.00046 → 0.00075) na direção oposta: com 0
rewrites, toda pergunta chega em `generate_answer` de primeira, gerando
uma resposta completa (mais tokens de output) em vez de, em alguns casos
do modo Dense, gastar esse orçamento em uma chamada de reformulação mais
barata. Essa é a explicação mais plausível, mas não foi isolada por
medição controlada (mesmo padrão de cautela já registrado acima pra
variação de latência do rewrite) — não tratar como conclusão fechada sem
comparar token-a-token entre os dois modos.

### Ordem de trabalho

1. ~~**Linha "Hybrid (BM25 + dense)"**~~ — feito, ver acima.

2. **Linha "Hybrid + RAPTOR"** — plano detalhado completo em
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

## Trilha paralela: infraestrutura de teste (não bloqueia a tabela acima)

- **VCR (pytest-recording) nos testes de `/query` e `/documents/ingest`** —
  `pytest-recording`/`vcrpy` já estão no `pyproject.toml` e
  `docs/decisoes-tecnicas.md` já documenta a decisão de usar VCR em vez de
  mocks manuais, mas nenhum teste usa `@pytest.mark.vcr` ainda. Hoje o
  isolamento de custo real de OpenAI nos testes depende só de
  `app.dependency_overrides` forçando `Fake*Provider` no `conftest.py` —
  funciona, mas nunca exercita o formato de resposta real da API (schema,
  erros reais do SDK). VCR resolveria isso: grava a chamada real uma vez,
  reproduz nas rodadas seguintes sem custo nem rede.

## Fora da sequência ativa

- **Chunking incremental com checkpoint** (salvar progresso por chunk e
  retomar de onde parou em caso de falha no embedding) — ideia levantada
  durante o trabalho do worker e adiada. Sem evidência de que reprocessar
  do zero seja caro o suficiente pra justificar a complexidade —
  reavaliar só se isso virar um problema real observado.
