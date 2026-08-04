from functools import lru_cache
from typing import Literal

from pydantic import Field, computed_field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    app_version: str = "0.1.0"
    database_url: str = Field(
        ..., description="URL do Postgres com driver async (postgresql+asyncpg)"
    )
    # Só lida por tests/conftest.py — banco separado do de dev/produção
    # acima, pra suite de teste (que faz TRUNCATE a cada teste) nunca
    # apagar dados reais por engano. Vazia por padrão porque só é
    # obrigatória ao rodar pytest, não ao subir a aplicação normalmente.
    test_database_url: str = Field(
        "", description="URL do Postgres de teste — deve ser um banco separado do de dev"
    )

    # --- OpenAI ---
    # Vazia por padrão: só é obrigatória quando embedding_provider="openai"
    # (validado abaixo). Rodar 100% em modo fake não deve exigir uma chave
    # que nunca será usada.
    openai_api_key: str = Field("", description="Chave da API OpenAI")
    openai_chat_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"

    # --- Provider switch ---
    # "fake" pra dev/teste sem custo. "openai" quando tiver chave.
    # Independentes: permite testar o /query (rewrite + generate) em modo
    # fake sem precisar reingerir documentos, mesmo com embeddings reais
    # já no banco — e vice-versa.
    embedding_provider: Literal["openai", "fake"] = "fake"
    chat_provider: Literal["openai", "fake"] = "fake"
    # --- Fila ---
    # "inline" pra dev/teste sem Redis (chama process_document direto, no
    # mesmo processo — equivalente ao que BackgroundTasks já fazia).
    # "redis" enfileira de verdade, consumido por um worker separado.
    queue_backend: Literal["inline", "redis"] = "inline"
    redis_url: str = ""
    ingest_queue_name: str = "ingest_queue"
    ingest_max_retries: int = 3

    @model_validator(mode="after")
    def _require_redis_url_when_selected(self) -> "Settings":
        if self.queue_backend == "redis" and not self.redis_url:
            raise ValueError("REDIS_URL é obrigatória quando QUEUE_BACKEND=redis.")
        return self

    @model_validator(mode="after")
    def _require_openai_key_when_selected(self) -> "Settings":
        if self.embedding_provider == "openai" and not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY é obrigatória quando EMBEDDING_PROVIDER=openai.")
        if self.chat_provider == "openai" and not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY é obrigatória quando CHAT_PROVIDER=openai.")
        return self

    langsmith_tracing: bool = False
    langsmith_api_key: str = ""
    langsmith_project: str = "docs-rag-api"

    uploads_dir: str = "/data/uploads"
    max_upload_size_mb: int = 10
    # Não é mais sobre travar o request (o worker separado processa fora
    # do ciclo HTTP) — é sobre custo: um PDF de texto puro pode ter muitas
    # páginas/chunks em poucos MB, e max_upload_size_mb sozinho não limita
    # isso. Cada chunk gera uma chamada de embedding; este limite evita um
    # único documento gerar um custo de OpenAI grande e silencioso.
    max_pdf_pages: int = 50
    chunk_size: int = 700
    chunk_overlap: int = 100

    retrieval_top_k: int = 5
    # Calibrado via sweep sobre o dataset de avaliação (34 perguntas, ver
    # README "Calibração do retrieval_quality_threshold"): 0.6 corta o
    # número de rewrites quase pela metade em relação a 0.7 (25->11 de 34)
    # sem perder Recall@5 nem recusa correta (ambos seguem 100%) — reduz
    # latência mediana de ~3.5s pra ~1.7s. Não existe corte que separe
    # perfeitamente perguntas respondíveis de fora-de-escopo por
    # similaridade pura; quem decide a recusa final é o LLM (`answerable`),
    # não este threshold — ele só controla quantas vezes tentar de novo
    # antes de desistir.
    retrieval_quality_threshold: float = 0.6
    # Se a 1ª reformulação não melhorou o retrieval o suficiente, é mais
    # provável que a informação não esteja nos documentos do que uma 2ª
    # reformulação achar algo que a 1ª não achou — melhor responder "não
    # encontrei" do que fazer o usuário esperar mais rodadas de LLM.
    max_rewrite_attempts: int = 2

    # --- Hybrid search (BM25 + dense, fundidos por RRF) ---
    # Quando ligado, retrieve roda search_similar_chunks e search_bm25_chunks
    # em paralelo e funde os dois rankings via reciprocal_rank_fusion — ver
    # docs/roadmap.md, linha "Hybrid (BM25 + dense)" da tabela de ablation.
    # Default True: é o pipeline medido (Recall@5 100%, ver README/avaliacao.md),
    # estritamente melhor que Dense sozinho no dataset de avaliação — Dense
    # continua acessível via HYBRID_SEARCH_ENABLED=false, preservado como
    # baseline de comparação no ablation, não removido.
    hybrid_search_enabled: bool = True
    # Sweep sobre as 34 perguntas do dataset de avaliação (best_similarity
    # do RRF, k=60): quase todos os scores colam em 1/61≈0.01639 (chunk que
    # aparece na posição 1 de uma única lista, dense OU bm25) e essa faixa
    # NÃO separa pergunta respondível de fora-de-escopo — o range observado
    # inteiro foi [0.01639, 0.03279], e 0.01639 aparece tanto em HIT quanto
    # em OOS. Diferente de cosine similarity, RRF nesta escala não é um
    # sinal discriminativo o suficiente pra decidir rewrite. Threshold
    # abaixo do mínimo observado desliga o gate na prática — a recusa por
    # falta de contexto fica inteiramente a cargo do answerable=False do
    # LLM (GeneratedAnswer), que já é quem decide isso hoje mesmo no modo
    # Dense quando o retrieval tecnicamente "passa" no threshold mas o
    # conteúdo não responde a pergunta.
    retrieval_quality_threshold_rrf: float = 0.005

    # --- Logging ---
    log_level: str = "INFO"
    log_format: Literal["json", "console"] = "console"

    # --- Rate limiting ---
    # Sem autenticação (projeto de portfólio, não multi-tenant — ver README),
    # o limite é por IP. Cobre os dois endpoints que custam dinheiro (OpenAI).
    # Desligável em teste (rate_limit_enabled=False) porque o client de teste
    # usa ASGITransport, onde todas as requisições compartilham o mesmo IP.
    rate_limit_enabled: bool = True
    rate_limit_query: str = "20/minute"
    rate_limit_ingest: str = "10/minute"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def database_url_sync(self) -> str:
        return self.database_url.replace("postgresql+asyncpg", "postgresql+psycopg")


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
