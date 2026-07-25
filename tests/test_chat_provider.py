from pydantic import BaseModel

from app.services.chat import FakeChatProvider


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
