# Avaliação

[← Voltar ao README](../README.md)

Dataset: 34 perguntas sobre 3 documentos reais e vigentes, de finanças e
direito (Glossário de Estatísticas Monetárias e de Crédito do Banco
Central, Guia CVM do Investidor sobre Fundos de Investimento Imobiliário,
e Cartilha do Consumidor do Ministério da Justiça) com `expected_chunk_id`.
23 perguntas têm resposta nos documentos; 11 são propositalmente fora de
escopo — tópicos plausíveis no mesmo domínio, mas nunca mencionados nos
três documentos. Uma pergunta (`venda casada`) tem `expected_chunk_id`
como lista — o CDC repete a mesma definição em duas seções do texto
("Práticas abusivas" e "Venda casada"), e qualquer um dos dois chunks
conta como acerto de recall.

> **Nota:** as seções abaixo (calibração de `retrieval_quality_threshold`,
> teste de viés do juiz de Faithfulness, e o padrão de confusão do LLM em
> perguntas de resposta negativa) foram medidas contra os 3 documentos
> sintéticos anteriores (contrato de serviço, política de reembolso,
> manual de onboarding), substituídos pelos 3 documentos reais acima. As
> conclusões qualitativas provavelmente continuam válidas, mas os números
> específicos (tabelas de sweep, scores de Faithfulness por caso) não
> foram re-medidos contra o dataset novo.

Resultado com o pipeline atual, retrieval híbrido
(`HYBRID_SEARCH_ENABLED=true`, o padrão — busca densa + BM25 fundidos por
Reciprocal Rank Fusion, ver [Decisões técnicas](decisoes-tecnicas.md)):

| Métrica | Resultado |
|---------|-----------|
| Recall@5 (perguntas respondíveis) | 100% (23/23) |
| Rewrite disparado | 0/34 perguntas |
| Recusa correta (fora de escopo) | 100% (11/11) |
| Faithfulness | 1.00 (22 respostas não-recusa) |
| Latência p50 | 1.77s |
| Latência p95 | 3.13s |
| Custo médio / query | US$ 0.00075 |

Com `HYBRID_SEARCH_ENABLED=false` (retrieval só-denso, baseline "Dense" do
[roadmap](roadmap.md)):

| Métrica | Resultado |
|---------|-----------|
| Recall@5 (perguntas respondíveis) | 96% (22/23) |
| Rewrite disparado | 11/34 perguntas |
| Recusa correta (fora de escopo) | 100% (11/11) |
| Faithfulness | 1.00 (21 respostas não-recusa) |
| Latência p50 | 3.36s |
| Latência p95 | 10.92s |
| Custo médio / query | US$ 0.00046 |

A única falha de recall do modo só-denso (1/23) é a pergunta sobre
CRI/CRA (siglas exatas) no glossário do BCB: mesmo após 2 rewrites, o
retrieval denso nunca traz o chunk certo — os 5 chunks recuperados são
todos sobre outras modalidades de crédito, vizinhos temáticos que erram o
termo exato. A resposta do sistema ("não encontrei") é correta dado o
contexto que recebeu — a falha é de retrieval, não de geração. É esse
caso, concreto e medido (não hipotético), que o retrieval híbrido
resolve: BM25 encontra o chunk certo por match de termo exato, RRF o
promove ao top-5 fundido. Comparação completa e detalhes de calibração
(threshold de RRF, fix da tsquery para perguntas longas) em
[roadmap](roadmap.md).

Custo por query do modo híbrido é maior (US$ 0.00075 vs 0.00046) porque
`rewrite` nunca dispara (0/34 vs 11/34) — toda pergunta chega em
`generate_answer` na primeira tentativa, gerando uma resposta completa em
vez de, em parte dos casos do modo denso, gastar parte do orçamento numa
chamada de reformulação mais barata. Latência é menor pelo mesmo motivo
(sem chamada extra de LLM para reformular).

Medido com `gpt-4o-mini` + `text-embedding-3-small`, `chunk_size=700`
(padrão — diferente da rodada anterior, que usava `chunk_size=150` porque
os documentos sintéticos eram curtos demais para gerar múltiplos chunks
no tamanho padrão; os documentos reais não têm essa limitação). Faithfulness
avaliada por um segundo LLM-juiz, dado o contexto recuperado e a resposta
gerada, e calculada apenas sobre respostas que tentaram afirmar algo com
base no contexto — uma recusa correta ("não sei") não é falta de fidelidade,
é o comportamento esperado, e incluí-la penalizaria a métrica injustamente.
"Recusa correta" e a exclusão de recusas do cálculo de Faithfulness usam
o campo estruturado `answerable` (ver [Decisões técnicas](decisoes-tecnicas.md)),
não mais regex sobre frases de recusa.

Faithfulness é medida sobre 21 (não 22) respostas não-recusa: a pergunta
"Produtos com lacre de segurança rompido podem ser devolvidos por
arrependimento?" (cuja resposta correta é "não", com uma exceção) foi
classificada com `answerable=false` em múltiplas rodadas de avaliação
independentes — não foi um evento isolado. O LLM interpreta uma resposta
negativa como "não encontrei a informação" em vez de "encontrei, e a
resposta é não". É um padrão real e reproduzível, não ruído aleatório:
perguntas cuja resposta correta é uma negação parecem sistematicamente
mais propensas a essa confusão do que perguntas com resposta afirmativa.
Vale ter isso em mente ao desenhar novas perguntas respondíveis para o
dataset — e é um caso interessante para investigar se vale reforçar o
prompt de geração no futuro (fora do escopo desta avaliação).

## Viés do juiz de Faithfulness

O juiz de Faithfulness é o mesmo modelo que gera as respostas
(`gpt-4o-mini`), o que é uma limitação metodológica conhecida de
LLM-as-judge: um modelo tende a dar nota mais generosa à sua própria
"forma de escrever" do que um avaliador independente daria
(self-preference bias). Testado de duas formas, com um segundo juiz
diferente do gerador (`gpt-4.1-mini`):

**Contra o dataset real de avaliação**: as 21 respostas não-recusa desta
rodada resultaram em nota 1.00 idêntica para os dois juízes — diferença
zero. Isso não significa que o viés não existe; significa que, com
`retrieval_quality_threshold=0.6`, o pipeline acerta a maioria das
respostas com folga suficiente para não deixar margem de discordância
entre juízes — a amostra não teve "zona cinzenta".

**Contra 5 casos fabricados deliberadamente ambíguos** (respostas com
extrapolação, generalização além do texto ou inferência não dita
explicitamente pelo contexto, construídos à parte do dataset oficial
só para este teste): os juízes discordaram em 2 dos 5 casos, com
diferença média de 0.20 na amostra — `gpt-4.1-mini` (juiz diferente)
julgou consistentemente mais rigoroso que `gpt-4o-mini` (mesmo modelo
do gerador) nos casos de extrapolação. Isso é evidência concreta de que
o viés existe e é mensurável, mesmo não tendo aparecido nas 21
respostas reais desta rodada.

Conclusão: o número de Faithfulness reportado acima (0.98) provavelmente
não está inflado *nesta rodada específica*, porque as respostas
avaliadas eram, em sua maioria, diretas o bastante para não dar margem
de discordância — mas o mecanismo de viés está presente e se manifestaria
caso o gerador cometesse mais extrapolações sutis. Não trocamos o juiz de
produção por isso (adicionaria custo/latência a cada avaliação para um
ganho que só se manifesta em casos de fronteira raros neste dataset),
mas fica documentado como limitação conhecida, com evidência empírica
em vez de suposição.

## Calibração do `retrieval_quality_threshold`

`app/core/config.py` define `retrieval_quality_threshold=0.6` — o corte de
similaridade de cosseno que decide, em `decide_after_retrieve`
(`app/rag/graph.py`), se o retrieval foi bom o suficiente pra gerar a
resposta ou se vale reformular a pergunta e tentar de novo. Esse valor foi
calibrado via sweep sobre as 34 perguntas do dataset, não escolhido a
priori.

Um sweep retrieval-only (sem LLM, só `search_similar_chunks` pra cada
pergunta) mediu `best_similarity` da primeira tentativa e separou
explicitamente perguntas respondíveis de fora-de-escopo — são duas
populações diferentes e não deveriam ser lidas num histograma único:

| threshold | respondíveis: passam direto | respondíveis: disparam rewrite | OOS: passam direto | OOS: disparam rewrite |
|-----------|------|------|------|------|
| 0.55 | 21/22 | 1/22 | 8/12 | 4/12 |
| 0.60 | 16/22 | 6/22 | 7/12 | 5/12 |
| 0.65 | 14/22 | 8/22 | 3/12 | 9/12 |
| 0.70 (anterior) | 8/22 | 14/22 | 1/12 | 11/12 |
| 0.75 | 5/22 | 17/22 | 0/12 | 12/12 |
| 0.80 | 2/22 | 20/22 | 0/12 | 12/12 |

A distribuição de similaridade das duas populações se sobrepõe bastante —
não existe um corte que separe perfeitamente "tem resposta" de "não tem
resposta" usando só similaridade de cosseno. Isso é esperado: quem decide
a recusa final é o LLM (`answerable`, ver [Decisões técnicas](decisoes-tecnicas.md)),
não este threshold — ele só controla **quantas vezes tentar de novo** antes de
desistir, não **se** o sistema vai acertar.

Isso levanta a pergunta certa: já que a similaridade sozinha não separa os
dois grupos, o threshold deveria só minimizar rewrite desnecessário sem
piorar as métricas que importam (Recall@5 e recusa correta). Rodando o
eval completo (com LLM) comparando `0.70` (valor anterior) contra `0.60`:

| Métrica | 0.70 | 0.60 |
|---------|------|------|
| Recall@5 | 100% (22/22) | 100% (22/22) |
| Recusa correta (OOS) | 100% (12/12) | 100% (12/12) |
| Rewrite disparado | 25/34 | 11/34 |
| Latência p50 | 3.51s | 1.65s |
| Latência p95 | 5.93s | 4.51s |

`0.60` corta o número de rewrites quase pela metade e quase dobra a
velocidade de resposta (latência mediana), sem custar nada nas duas
métricas de correção — por isso é o valor adotado. Não foi testado abaixo
de 0.60 porque, na tabela de distribuição acima, `0.55` já deixa passar
8/12 casos out-of-scope sem nenhuma tentativa de rewrite, dependendo cada
vez mais só do julgamento final do LLM — o ganho de latência marginal não
parecia valer reduzir ainda mais o uso do mecanismo de rewrite que o
projeto existe para demonstrar.

Latência p95 desta rodada inclui uma execução com retry/timeout de rede
transitório do lado da OpenAI (uma única chamada levou ~34 minutos,
bem fora da faixa normal de segundos) — não influenciou a métrica
reportada, já que o outlier ficou acima do percentil 95 de 34 amostras,
mas fica registrado que latência de cauda longa aqui reflete
instabilidade de rede externa, não o código do pipeline.

O prompt de geração também instrui o modelo a não combinar números de
trechos diferentes (ex: taxa de multa de um chunk + valor total de outro) —
LLMs erram aritmética com frequência maior do que aparentam, e esse tipo de
inferência silenciosa é mais difícil de auditar do que citar os valores
como aparecem no texto.

## Calibração do `retrieval_quality_threshold_rrf`

O retrieval híbrido funde busca densa e BM25 por posição (Reciprocal Rank
Fusion, k=60, ver [Decisões técnicas](decisoes-tecnicas.md)) — o score
resultante não é cosine similarity e não pode reusar o threshold `0.6`
calibrado para essa escala. Um sweep retrieval-only (sem LLM) sobre as 34
perguntas mediu `best_similarity` do resultado fundido:

- Range observado inteiro: `[0.01639, 0.03279]` — uma faixa muito mais
  estreita que a de cosine similarity (que vai de ~0 a ~1).
- A maioria dos scores (perguntas respondíveis **e** fora de escopo) cai
  exatamente em `0.01639` (= 1/61, o score de um chunk que aparece na
  posição 1 de uma única lista, dense ou BM25, mas não das duas). Esse
  valor aparece nos dois grupos — diferente do sweep de cosine acima, não
  existe corte que separe as duas populações nessa escala.

Como o score não é discriminativo o suficiente para decidir rewrite,
`retrieval_quality_threshold_rrf=0.005` fica abaixo do mínimo observado —
o gate de rewrite por score fica efetivamente desligado no modo híbrido,
e a recusa por falta de contexto passa a depender inteiramente de
`answerable=False` do LLM (mesmo mecanismo que já decide isso no modo
denso quando o retrieval "passa" no threshold mas o conteúdo não responde
a pergunta). Na prática, isso zera o rewrite no eval (0/34, contra 11/34
no modo denso) sem custar Recall@5 nem recusa correta — ver tabela
comparativa no topo deste documento.

## Como rodar a avaliação

Requer `EMBEDDING_PROVIDER=openai` e `CHAT_PROVIDER=openai` no `.env`,
com uma `OPENAI_API_KEY` real:

```bash
uv run python -m evals.setup_dataset   # ingere os documentos uma vez
uv run python -m evals.run_eval
```

## RAGAS — bloqueado por incompatibilidade de dependências

Avaliado adicionar [RAGAS](https://docs.ragas.io/) para métricas
padronizadas e comparáveis com outros projetos (Faithfulness, Context
Recall, Context Precision, Response Relevancy). `evals/dataset.json` já
tem `reference_answer` em texto para as 23 perguntas respondíveis —
pré-requisito para Context Recall/Precision, que julgam contra uma
resposta de referência, não só o `expected_chunk_id` exato usado no
Recall@5 atual.

A integração em si está bloqueada: toda versão de `ragas` publicada até
agora (testado `0.4.3` e `0.2.15`) importa `ChatVertexAI` de
`langchain_community.chat_models.vertexai` incondicionalmente
(`ragas/llms/base.py`), um caminho removido em `langchain-community>=0.4`
— a versão puxada por `langchain>=1.3`, que este projeto usa. Testado
manualmente (não só verificado em changelog):

- **Mesmo venv, downgrade da família langchain para 0.3.x**: resolve o
  import do `ragas`, mas força `langchain`/`langchain-core`/
  `langchain-openai` de volta para 0.3.x — incompatível com o resto do
  código do projeto, que depende de `langchain>=1.3,<2.0`.
- **Venv isolado, mas importando `app.core.config` no mesmo processo**:
  ainda colide, porque `uv sync`/`uv pip install -e .` traz
  `[project.dependencies]` (LangChain 1.x) para dentro da mesma
  resolução, mesmo em grupo `[dependency-groups]` separado — `uv` trata
  grupos como aditivos ao pacote raiz, não como ambientes independentes.
- **Venv isolado, instalando dependências do app uma a uma com
  `--no-deps`** (contornando o pacote raiz): funciona até o próximo
  import cruzado — `langchain-core` mais recente sem pin explícito quebra
  o `ragas`, e alinhar isso arrasta `pydantic`/`pydantic-core` para uma
  combinação que quebra de novo. Cadeia de reação, não resolvível só com
  flags de instalação.

**Caminho viável, não implementado ainda**: separar em dois processos
com um arquivo intermediário — 1) `run_eval.py` (venv normal do projeto)
exporta `question`/`answer`/`retrieved_contexts` para JSON; 2) um script
`ragas_eval.py` roda num venv **sem o pacote `app` instalado** (só
`ragas` + `langchain-openai<0.4`, sem `langchain>=1.3`), lê esse JSON e
chama `ragas.evaluate()`. Não dá para o script RAGAS importar
`app.core.config` ou `evals.run_eval` diretamente — os dois processos não
podem compartilhar interpretador. Não implementado por decisão explícita
(mais fricção operacional do que o valor imediato justificava); fica
registrado aqui para retomar se `ragas` corrigir o import incondicional
upstream ou se o projeto migrar para uma versão do RAGAS compatível com
LangChain 1.x.
