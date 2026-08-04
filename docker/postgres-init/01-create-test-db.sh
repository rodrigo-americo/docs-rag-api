#!/usr/bin/env bash
# Roda uma única vez, na primeira inicialização do container (Postgres só
# executa scripts em docker-entrypoint-initdb.d/ quando o volume de dados
# está vazio — não roda de novo em restarts subsequentes).
#
# Cria um segundo banco, "${POSTGRES_DB}_test", no mesmo container/instância
# do banco de dev — isolamento suficiente pra suite de teste nunca truncar
# dados de dev por engano (ver tests/conftest.py e docs/testes.md), sem
# subir um segundo serviço Postgres inteiro.
set -euo pipefail

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    CREATE DATABASE "${POSTGRES_DB}_test";
EOSQL
