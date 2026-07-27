# Segurança

[← Voltar ao README](../README.md)

Este é um projeto de portfólio para demonstrar a arquitetura RAG, não um
sistema multi-tenant em produção — por isso não há login/autenticação:
adicionar isso exigiria gerenciamento de usuários e credenciais sem servir
ao objetivo do projeto. As proteções existentes hoje cobrem os riscos
relevantes para esse escopo:

- **Rate limiting por IP** em `/query` e `/documents/ingest`.
- **Bloqueio de conteúdo suspeito** no ingest (padrões de prompt injection).
- **Sanitização de HTML** em `answer` e `citations[].snippet` antes de sair
  da API.
- **Limite de tamanho de upload** (10 MB / 50 páginas).

Configure um spending limit na sua chave da OpenAI antes de rodar
publicamente — é a rede de segurança real contra uso indevido, dado que
não há autenticação.

**Contrato dos campos de resposta:** `answer` e `citations[].snippet` são
texto sanitizado (sem HTML), mas continuam sendo **conteúdo gerado por LLM
e/ou derivado de documentos enviados por terceiros** — nunca confie neles
como se fossem gerados pelo seu próprio backend. Um cliente que consumir
esta API não deveria, por exemplo, usar `answer` para tomar decisões
automatizadas sensíveis (ex: como entrada de outro sistema que executa
ações) sem validação adicional — sanitização de HTML remove um vetor
(XSS), não todos os riscos de tratar saída de LLM como dado confiável.

Ver também [Decisões técnicas](decisoes-tecnicas.md) para o raciocínio
detalhado por trás de cada uma dessas proteções.
