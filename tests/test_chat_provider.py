from unittest.mock import AsyncMock

import pytest
from pydantic import BaseModel

from app.core.config import settings
from app.services.chat import FakeChatProvider, OpenAIChatProvider, get_chat_provider


class _FieldTypes(BaseModel):
    # Cobre o branch "else" de FakeChatProvider.complete_structured (campo
    # cujo tipo não é bool nem str) — GeneratedAnswer (o schema real do
    # pipeline) só tem esses dois tipos, então esse branch nunca dispara
    # organicamente via /query.
    count: int = 0


class _Answer(BaseModel):
    answerable: bool
    answer: str


async def test_fake_chat_provider_complete_structured_matches_schema_types():
    provider = FakeChatProvider()

    result = await provider.complete_structured(
        "pergunta qualquer", system="instrução qualquer", schema=_Answer
    )

    assert isinstance(result, _Answer)
    assert result.answerable is True
    assert "pergunta qualquer" in result.answer


async def test_fake_chat_provider_complete_structured_fills_non_bool_non_str_field_with_default():
    provider = FakeChatProvider()

    result = await provider.complete_structured(
        "pergunta qualquer", system="instrução qualquer", schema=_FieldTypes
    )

    assert result.count == 0


def test_openai_chat_provider_raises_without_api_key():
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        OpenAIChatProvider(api_key="", model="gpt-4o-mini")


async def test_openai_chat_provider_complete_without_system_calls_ainvoke_with_plain_text():
    """system=None (branch não usado pelo pipeline real, que sempre passa
    system_prompt — ver app/rag/graph.py) — troca o client langchain
    interno por um mock simples pra exercitar sem rede (ChatOpenAI é um
    BaseModel pydantic congelado, patch.object em um único método dele
    quebra no cleanup — substituir o objeto inteiro evita isso)."""
    provider = OpenAIChatProvider(api_key="sk-fake", model="gpt-4o-mini")
    fake_response = type("Response", (), {"content": "resposta"})()
    mock_client = AsyncMock()
    mock_client.ainvoke = AsyncMock(return_value=fake_response)
    provider._client = mock_client

    result = await provider.complete("pergunta sem system")

    assert result == "resposta"
    mock_client.ainvoke.assert_awaited_once_with("pergunta sem system")


async def test_openai_chat_provider_complete_with_system_wraps_in_messages():
    from langchain_core.messages import HumanMessage, SystemMessage

    provider = OpenAIChatProvider(api_key="sk-fake", model="gpt-4o-mini")
    fake_response = type("Response", (), {"content": "resposta com system"})()
    mock_client = AsyncMock()
    mock_client.ainvoke = AsyncMock(return_value=fake_response)
    provider._client = mock_client

    result = await provider.complete("pergunta", system="instrução")

    assert result == "resposta com system"
    (call_args,), _ = mock_client.ainvoke.call_args
    assert isinstance(call_args[0], SystemMessage)
    assert isinstance(call_args[1], HumanMessage)


def test_get_chat_provider_returns_fake_when_provider_is_fake(monkeypatch):
    monkeypatch.setattr(settings, "chat_provider", "fake")

    provider = get_chat_provider()

    assert isinstance(provider, FakeChatProvider)


def test_get_chat_provider_returns_openai_when_provider_is_openai(monkeypatch):
    # api_key fake mas não-vazia: OpenAIChatProvider só valida que a chave
    # não é vazia no __init__ (ver test acima) — não faz nenhuma chamada de
    # rede até complete()/complete_structured() serem chamados.
    monkeypatch.setattr(settings, "chat_provider", "openai")
    monkeypatch.setattr(settings, "openai_api_key", "sk-fake-key-for-unit-test")

    provider = get_chat_provider()

    assert isinstance(provider, OpenAIChatProvider)
