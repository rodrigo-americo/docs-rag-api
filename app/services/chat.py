from typing import Protocol

from app.core.config import settings


class ChatProvider(Protocol):
    """Protocol pra qualquer provider de chat/completions.

    Mesmo raciocínio de EmbeddingProvider: Protocol (PEP 544) em vez de
    ABC, structural typing.
    """

    async def complete(self, prompt: str, *, system: str | None = None) -> str:
        """Gera uma resposta a partir de um prompt.

        `system` separa instrução (comportamento fixo, confiável) de
        `prompt` (conteúdo variável, que pode incluir texto de documentos
        ingeridos por terceiros — não confiável). Ver OpenAIChatProvider.
        """
        ...


class FakeChatProvider:
    """Provider para teste sem gasto real de token.

    Atualmente só fatia o texto, pode ser melhorado.
    """

    async def complete(self, text: str, *, system: str | None = None) -> str:
        return f"[fake-completion] {text[:50]}"


class OpenAIChatProvider:
    """Provider real usando OpenAI via langchain-openai.

    Não instancia a OpenAI no __init__ — adia pra primeira chamada.
    Assim o app sobe mesmo sem chave configurada (útil em dev fake).
    """

    def __init__(self, api_key: str, model: str) -> None:
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY vazia mas CHAT_PROVIDER=openai. "
                "Configura a chave no .env ou volta pra fake."
            )
        from langchain_openai import ChatOpenAI

        self._client = ChatOpenAI(api_key=api_key, model=model)

    async def complete(self, text: str, *, system: str | None = None) -> str:
        if system is None:
            response = await self._client.ainvoke(text)
            return response.content

        from langchain_core.messages import HumanMessage, SystemMessage

        messages = [SystemMessage(content=system), HumanMessage(content=text)]
        response = await self._client.ainvoke(messages)
        return response.content


def get_chat_provider() -> ChatProvider:
    """Factory: escolhe provider conforme Settings.

    Chamada a cada uso é OK — providers são leves de instanciar.
    Se virar gargalo, cacheia com @lru_cache.
    """
    if settings.chat_provider == "openai":
        return OpenAIChatProvider(
            api_key=settings.openai_api_key,
            model=settings.openai_chat_model,
        )
    return FakeChatProvider()
