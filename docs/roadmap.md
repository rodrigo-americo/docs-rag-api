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
| Dense (baseline atual) | 96% (22/23) | 1.00 | US$ 0.00046/query | p50 3.36s / p95 10.92s |
| Dense + rewrite | — | — | | |
| Hybrid (BM25 + dense) | — | — | | |
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

### Ordem de trabalho

1. **Linha "Dense"** — já é o comportamento atual (retrieval só por
   similaridade vetorial, sem rewrite). Rodar o eval como está hoje e
   registrar o número na tabela — isso já existe, só falta capturar
   formalmente como baseline antes de mexer em mais nada.

2. **Linha "Dense + rewrite"** — quantificar o ganho real do branch de
   rewrite do LangGraph (quantas perguntas só acertam graças à
   reformulação, latência/custo adicional que ele introduz). Maior parte
   da infraestrutura já existe (`retry_count` no estado do grafo, dataset
   de eval já força o branch em 11/34 perguntas hoje).

3. **Linha "Hybrid (BM25 + dense)"** — combina busca por similaridade
   semântica (o que já existe) com busca por termo exato (BM25) —
   resolve o caso onde embedding sozinho erra por termos muito
   específicos (nomes próprios, códigos, jargão exato). Padrão real de
   indústria, maior valor de defesa entre as mudanças de arquitetura
   antes do RAPTOR.

4. **Linha "Hybrid + RAPTOR"** — plano detalhado completo em
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
