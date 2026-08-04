# syntax=docker/dockerfile:1
FROM python:3.13-slim AS base

# Copia o binário uv da imagem oficial em vez de instalar via pip —
# mais rápido e evita puxar pip/setuptools só pra isso.
COPY --from=ghcr.io/astral-sh/uv:0.5 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Camada de dependências isolada do código da app: só reinstala quando
# pyproject.toml/uv.lock mudam, não a cada alteração em app/.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

COPY . .
RUN uv sync --frozen --no-dev

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

# --- estágio de teste ---
# Só usado pelo serviço `test` do docker-compose.yml — roda a suite dentro
# de Linux (o mesmo SO do CI), não Windows, onde o proactor do asyncio tem
# comportamento diferente de coleta de warnings de socket (ver
# docs/testes.md). Camada extra em cima de `base`: reusa tudo que já foi
# instalado ali, só adiciona o grupo dev (pytest, ruff etc.) que a imagem
# de produção deliberadamente não tem.
FROM base AS test
RUN uv sync --frozen

CMD ["uv", "run", "pytest"]
