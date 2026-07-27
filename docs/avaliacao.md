# Avaliação

[← Voltar ao README](../README.md)

Dataset: 34 perguntas sobre 3 documentos sintéticos (contrato de serviço,
política de reembolso, manual de onboarding) com `expected_chunk_id`.
22 perguntas têm resposta nos documentos (algumas com vocabulário
propositalmente distante do texto original, para forçar o branch de
rewrite do grafo); 12 são propositalmente fora de escopo — cobrindo três
variações: tópicos plausíveis nunca mencionados, perguntas que usam
vocabulário/entidades reais dos documentos mas pedem um dado que não
existe (mais difíceis de recusar corretamente que um tópico totalmente
alheio), e casos de fronteira que exploram os limites exatos de uma
definição do texto (ex: garantia vale só para "produtos duráveis" —
pergunta sobre produtos não-duráveis). Amostra ampliada de 4 para 12
casos negativos porque 100% sobre 4 exemplos tem intervalo de confiança
grande demais pra ser uma alegação séria — 12 casos, incluindo os mais
difíceis de cada variação, é uma prova mais forte do mecanismo de recusa.

| Métrica | Resultado |
|---------|-----------|
| Recall@5 (perguntas respondíveis) | 100% (22/22) |
| Rewrite disparado | 11/34 perguntas |
| Recusa correta (fora de escopo) | 100% (12/12) |
| Faithfulness | 0.98 (21 respostas não-recusa) |
| Latência p50 | 1.65s |
| Latência p95 | 4.51s |
| Custo médio / query | US$ 0.00010 |

Medido com `gpt-4o-mini` + `text-embedding-3-small`, `chunk_size=150` tokens
(reduzido só para a avaliação — os documentos sintéticos são curtos demais
para gerar múltiplos chunks com o `chunk_size=700` padrão). Faithfulness
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

## Como rodar a avaliação

Requer `EMBEDDING_PROVIDER=openai` e `CHAT_PROVIDER=openai` no `.env`,
com uma `OPENAI_API_KEY` real:

```bash
CHUNK_SIZE=150 uv run python -m evals.setup_dataset   # ingere os documentos uma vez
CHUNK_SIZE=150 uv run python -m evals.run_eval
```
