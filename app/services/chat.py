from typing import Protocol, TypeVar

from pydantic import BaseModel

from app.core.config import settings

T = TypeVar("T", bound=BaseModel)


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

    async def complete_structured(self, prompt: str, *, system: str, schema: type[T]) -> T:
        """Gera uma resposta validada contra `schema` (Pydantic).

        Usa function calling nativo do provider (não parsing de texto/JSON
        manual) — garante o formato pelo mecanismo da própria API, não por
        "confiar" que o LLM escreveu JSON válido em texto livre.
        """
        ...


class FakeChatProvider:
    """Provider para teste sem gasto real de token.

    Atualmente só fatia o texto, pode ser melhorado.
    """

    async def complete(self, text: str, *, system: str | None = None) -> str:
        return f"[fake-completion] {text[:50]}"

    async def complete_structured(self, prompt: str, *, system: str, schema: type[T]) -> T:
        # Preenche cada campo com um valor fake plausível pelo tipo declarado
        # — evita hardcodar conhecimento de schemas específicos aqui.
        values = {}
        for field_name, field in schema.model_fields.items():
            if field.annotation is bool:
                values[field_name] = True
            elif field.annotation is str:
                values[field_name] = f"[fake-completion] {prompt[:50]}"
            else:
                values[field_name] = field.get_default(call_default_factory=True)
        return schema(**values)


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

    async def complete_structured(self, prompt: str, *, system: str, schema: type[T]) -> T:
        from langchain_core.messages import HumanMessage, SystemMessage

        messages = [SystemMessage(content=system), HumanMessage(content=prompt)]
        structured_client = self._client.with_structured_output(schema)
        return await structured_client.ainvoke(messages)


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
