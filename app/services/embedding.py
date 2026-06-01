import hashlib
from typing import Protocol

import numpy as np

from app.core.config import settings


class EmbeddingProvider(Protocol):
    """Protocol pra qualquer provider de embeddings.

    Protocol (PEP 544) em vez de ABC: structural typing. Qualquer classe
    com esses métodos satisfaz o tipo, sem precisar herdar nada.
    """

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Gera embeddings pra lista de textos. Retorna na mesma ordem."""
        ...

    async def embed_query(self, text: str) -> list[float]:
        """Gera embedding pra uma query única."""
        ...


class FakeEmbeddingProvider:
    """Provider determinístico pra dev/teste.

    Hash do texto -> seed -> vetor aleatório normalizado.
    Mesma entrada sempre gera mesma saída. Vetores diferentes pra textos
    diferentes. Magnitude 1 (normalizado), igual à OpenAI — assim coseno
    e L2 produzem ranking equivalente, e nada quebra quando trocar pelo
    provider real.
    """

    def __init__(self, dim: int = 1536) -> None:
        self.dim = dim

    def _embed_one(self, text: str) -> list[float]:
        # Hash determinístico -> seed do numpy
        seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        vec = rng.standard_normal(self.dim).astype(np.float32)
        # Normaliza pra magnitude 1 (mimetiza OpenAI)
        vec = vec / np.linalg.norm(vec)
        return vec.tolist()

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]

    async def embed_query(self, text: str) -> list[float]:
        return self._embed_one(text)


class OpenAIEmbeddingProvider:
    """Provider real usando OpenAI via langchain-openai.

    Não instancia a OpenAI no __init__ — adia pra primeira chamada.
    Assim o app sobe mesmo sem chave configurada (útil em dev fake).
    """

    def __init__(self, api_key: str, model: str) -> None:
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY vazia mas EMBEDDING_PROVIDER=openai. "
                "Configura a chave no .env ou volta pra fake."
            )
        # Import lazy: evita importar langchain_openai se nunca usar.
        from langchain_openai import OpenAIEmbeddings

        self._client = OpenAIEmbeddings(api_key=api_key, model=model)

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return await self._client.aembed_documents(texts)

    async def embed_query(self, text: str) -> list[float]:
        return await self._client.aembed_query(text)


def get_embedding_provider() -> EmbeddingProvider:
    """Factory: escolhe provider conforme Settings.

    Chamada a cada uso é OK — providers são leves de instanciar.
    Se virar gargalo, cacheia com @lru_cache.
    """
    if settings.embedding_provider == "openai":
        return OpenAIEmbeddingProvider(
            api_key=settings.openai_api_key,
            model=settings.openai_embedding_model,
        )
    return FakeEmbeddingProvider(dim=settings.openai_embedding_dim)