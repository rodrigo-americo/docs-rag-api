# Plano: chunking hierárquico estilo RAPTOR

## Contexto

Hoje o retrieval é uma busca plana: cada chunk é um pedaço de texto
independente, sem relação estrutural entre si além de pertencer ao mesmo
documento. Isso significa que um tema que aparece espalhado em partes
não-contíguas do texto (ex: seção 2 e seção 9 de um documento longo) não
tem nenhum jeito de "se encontrar" no retrieval — cada chunk só compete
individualmente pela similaridade com a pergunta.

RAPTOR (Recursive Abstractive Processing for Tree-Organized Retrieval)
resolve isso construindo uma árvore de baixo pra cima: agrupa chunks
semanticamente parecidos (não por proximidade no texto, por significado),
gera um resumo por LLM de cada grupo, e repete recursivamente sobre os
resumos até sobrar pouca coisa. No retrieval, todos os níveis da árvore
(chunks originais e resumos de todos os níveis) competem juntos num único
ranking por similaridade — a estratégia "collapsed tree", que teve melhor
desempenho empírico no paper original e é a mais barata de implementar
aqui, já que reaproveita a busca por similaridade que já existe quase sem
mudança.

Decisões já confirmadas com o usuário:
- **UMAP + GMM completo**, fiel ao paper (não um k-means simplificado) —
  novas dependências `umap-learn` e `scikit-learn`.
- ~~Adicionar um documento maior e realista ao dataset de avaliação~~ —
  **feito**: os 3 documentos sintéticos foram substituídos por documentos
  reais (Glossário de Crédito do BCB, Guia CVM de FII, Cartilha do
  Consumidor do MJ — ver [avaliação](../avaliacao.md)). A Cartilha sozinha
  gera 78 chunks com `chunk_size=700`, volume e dispersão de temas
  suficientes para RAPTOR formar clusters com sinal real.
- **Retrieval "collapsed tree"** — todos os níveis competem juntos num
  único ranking, sem navegação nível-a-nível.
- **Clustering many-to-many de verdade**: GMM faz soft clustering (um
  chunk pode pertencer a mais de um grupo/pai) — isso é persistido via
  uma tabela de associação (`chunk_parents`), não uma FK simples de pai
  único. Mais fiel ao paper, mais complexo de propagar por
  retrieval/citações — mapeado nos passos abaixo.
- **`chunk_index` nullable**, com unique constraint parcial só para
  leaves (`WHERE level = 0`) — nós de resumo não têm "posição no texto
  original", então a coluna não deveria fingir que têm.

Este é um trabalho guiado: cada passo é pequeno, testável isoladamente, e
revisado antes do próximo começar — o usuário escreve o código, eu reviso
e explico.

## Impacto do many-to-many na forma da árvore

Com pai único, "ancestralidade" seria uma linha reta fácil de percorrer.
Com múltiplos pais por nó, um chunk pode estar em mais de um "ramo" da
árvore ao mesmo tempo — isso afeta três lugares que precisam de decisão
explícita durante os passos:

- **Citações**: um nó de resumo retornado como citação pode ter vindo de
  múltiplos grupos de origem. `Citation`/`RetrievedChunk` precisam expor
  o `level`, mas não necessariamente toda a lista de pais — decidir no
  passo 7 se vale expor `parent_ids: list[UUID]` na citação ou deixar
  isso como detalhe interno.
- **Recall ancestor-aware no eval** (passo 8): "este chunk recuperado é
  ancestral do chunk esperado" deixa de ser uma caminhada linear
  (`parent_chunk_id` único) e vira uma busca em grafo (um nó pode ter
  múltiplos caminhos até a raiz). O helper de ancestralidade precisa
  fazer uma busca em largura/profundidade sobre `chunk_parents`, não um
  simples loop `while parent_id is not None`.
- **Custo de escrita**: cada nó de resumo pode gerar múltiplas linhas em
  `chunk_parents` (uma por pai), não uma atualização de coluna — pequena
  diferença na hora de persistir, mas real.

## O que precisa ser criado

**Schema (`app/models/document.py`)**:
- `Chunk.level: int` — `0` para leaves (chunk_text), `1..N` para nós de
  resumo. Default `0` (linhas existentes e o caminho sem RAPTOR não
  precisam de backfill).
- `Chunk.chunk_index` passa a ser `nullable=True`. `UniqueConstraint`
  vira um índice único parcial: `postgresql_where=(level == 0)` (ou
  equivalente via `Index` com `postgresql_where`) — só leaves precisam
  de posição única no documento.
- Nova tabela `chunk_parents`: `chunk_id (FK chunks.id)`,
  `parent_chunk_id (FK chunks.id)`, chave primária composta
  `(chunk_id, parent_chunk_id)`. `ondelete="CASCADE"` em ambas as FKs —
  se um nó é apagado (documento inteiro cascateando), as relações somem
  junto, sem órfãos.
- Índice em `chunk_parents.parent_chunk_id` (consultas "quais são os
  filhos deste resumo") e em `chunk_parents.chunk_id` (a PK composta já
  cobre isso na ordem `chunk_id` primeiro, mas vale confirmar durante o
  passo 1 se a ordem da PK atende as duas direções de busca ou se precisa
  de um índice adicional).

**Novo módulo `app/services/tree_builder.py`**:
- Função pura de clustering: `cluster_embeddings(embeddings) -> list[list[int]]`
  (UMAP + GMM + seleção de número de clusters via BIC), sem I/O, testável
  com embeddings sintéticos.
- Função de sumarização: recebe os textos de um cluster, chama
  `ChatProvider.complete` com um novo prompt de sumarização, devolve uma
  string. Testável com `FakeChatProvider`, sem custo real.
- `build_tree(...)` — compõe as duas acima recursivamente, devolvendo uma
  estrutura intermediária (nós por nível + suas relações em
  `chunk_parents`), até um nível máximo ou até o clustering colapsar num
  único grupo.

**Config (`app/core/config.py`)**, seguindo o estilo `Literal` +
`model_validator` já usado:
- `raptor_enabled: bool = False`
- `raptor_min_chunks_for_tree: int = 8` (documentos abaixo disso ficam
  só com leaves, exatamente como hoje — não há construção de árvore)
- `raptor_max_levels: int = 3` (teto de profundidade, também teto de
  custo de LLM)
- `raptor_min_cluster_size: int = 2`
- `retrieval_strategy: Literal["flat", "collapsed_tree"] = "collapsed_tree"`
  — permite comparar flat vs árvore no mesmo corpus indexado, útil pro
  eval A/B do passo 8.

**Retrieval (`app/rag/retrieval.py`)**:
- `RetrievedChunk` ganha `level: int`.
- Introduzir um `Protocol RetrievalStrategy` (mesmo padrão de
  `EmbeddingProvider`/`ChatProvider`) com duas implementações: uma que
  filtra `level == 0` (flat, comparável ao comportamento de hoje) e outra
  sem filtro (collapsed tree — o comportamento novo). `make_retrieve_node`
  em `app/rag/graph.py` passa a receber a estratégia injetada, em vez de
  importar `search_similar_chunks` direto por nome.

**Schemas (`app/schemas/query.py`)**:
- `Citation` ganha `level: int` — uma citação vinda de um resumo LLM tem
  proveniência diferente de uma vinda de texto verbatim, e isso deveria
  ser visível pra quem consome a API.

**Ingest (`app/services/ingest.py`)**:
- `process_document`, depois de `_persist_chunks` (leaves, level=0,
  inalterado), chama `build_tree` só se
  `raptor_enabled and len(chunks_data) >= raptor_min_chunks_for_tree`.
  Persiste os nós de resumo + as relações em `chunk_parents` na mesma
  transação, antes de `status = INDEXED`. Sem gate, documentos pequenos
  seguem exatamente como hoje.

**Eval (`evals/`)**:
- Novo documento maior e realista em `evals/documents/`, com temas que se
  repetem em seções não-contíguas — o caso que a árvore existe pra
  resolver.
- `evals/dataset.json` (34 perguntas atuais) continua intacto — essas
  perguntas usam os documentos curtos, que ficam abaixo do threshold e
  nunca constroem árvore, então o recall atual não pode regredir.
- Novo `evals/dataset_synthesis.json` — perguntas que só um resumo (ou
  combinação de múltiplos chunks distantes) consegue responder bem;
  avaliadas por `answerable`/Faithfulness, não por recall de um único
  `chunk_id` (métrica errada pra esse tipo de pergunta).
- `_recall_at_k` ganha uma checagem "ancestor-aware": o chunk recuperado
  é o esperado, ou é ancestral dele na árvore? Com múltiplos pais, isso é
  busca em grafo sobre `chunk_parents` (BFS/DFS), não um loop linear.

## Passos sequenciados

1. **Schema (migration + model, sem mudança de comportamento).** `level`,
   `chunk_index` nullable + índice único parcial, tabela `chunk_parents`.
   `_persist_chunks` passa a gravar `level=0` explicitamente. Toda a
   suite continua passando sem nenhuma mudança de comportamento visível.
2. **Dependências e settings, sem código novo as usando ainda.**
   `umap-learn`, `scikit-learn` no `pyproject.toml`; novos campos em
   `Settings`, todos com defaults conservadores (`raptor_enabled=False`).
   Só confirma que instalar as libs não quebra nada.
3. **Clustering isolado, testável sem LLM/banco.** `cluster_embeddings`
   em `tree_builder.py` — UMAP + GMM + BIC. Testado com embeddings
   sintéticos (dois blobs bem separados, por exemplo). **Este é o passo
   mais rico em aprendizado do plano inteiro** — GMM, BIC e UMAP são
   matemática nova, e vale entender cada peça isoladamente antes de
   qualquer outra coisa se misturar.
4. **Sumarização isolada, com `FakeChatProvider`.** A função que recebe
   textos de um cluster e devolve um resumo via `ChatProvider.complete`.
   Testável sem custo real.
5. **`build_tree` completo, ainda não chamado do ingest.** Compõe os
   passos 3+4 recursivamente, incluindo a montagem das relações
   many-to-many (quais nós têm quais pais). Testado ponta a ponta com
   fakes, confirmando que a estrutura resultante é bem formada (todo nó
   não-raiz tem ao menos um pai, número de níveis respeita o teto).
6. **Conectar ao `IngestService.process_document`, atrás do gate.**
   Chama `build_tree` só quando `raptor_enabled` e chunks suficientes;
   persiste nós de resumo + `chunk_parents` na mesma transação. Testa
   tanto o caminho "documento grande, constrói árvore" quanto "documento
   pequeno, continua só com leaves" (regressão explícita).
7. **Retrieval: `level` exposto, `RetrievalStrategy` (flat vs collapsed
   tree).** Protocol + duas implementações + factory + settings switch;
   `make_retrieve_node` recebe a estratégia injetada. Testes unitários
   com uma árvore pequena montada à mão (algumas leaves + um resumo).
   Suite existente (`test_query_endpoint.py`) continua passando sem
   mudança, já que o comportamento default sobre um corpus só-leaves é
   idêntico ao de hoje.
8. **Reconciliação do eval: recall ancestor-aware (busca em grafo) + novo
   documento grande + novo dataset de síntese.** Roda o eval de verdade
   (`raptor_enabled=True`, providers reais) — primeira medição real de
   recall flat vs árvore, e de faithfulness nas perguntas de síntese.
   Custo real de OpenAI aceito aqui, mesmo padrão já usado pro eval hoje.
9. **Documentação.** Registrar as decisões (hard vs soft assignment nos
   nós — aqui, many-to-many de verdade —, por que UMAP+GMM, por que
   collapsed tree, aritmética do teto de custo, números flat vs árvore)
   em `docs/decisoes-tecnicas.md`, seguindo a voz já estabelecida nas
   outras entradas.

## Verificação

- Passos 1-2: `uv run pytest` e `uv run ruff check` verdes, sem mudança
  de comportamento observável.
- Passos 3-5: testes unitários isolados, sem depender de Postgres/Redis/
  API real — puro numpy/fakes.
- Passo 6: teste de regressão explícito garantindo que documentos abaixo
  do threshold continuam idênticos ao comportamento pré-RAPTOR.
- Passo 7: suite existente (`test_query_endpoint.py`) inalterada; novos
  testes unitários das duas estratégias de retrieval.
- Passo 8: rodar `evals/run_eval.py` de verdade, comparando recall/
  faithfulness com `retrieval_strategy=flat` vs `collapsed_tree` sobre o
  mesmo corpus indexado com árvore — essa comparação é a evidência de
  que a feature vale o custo, não só uma alegação.
