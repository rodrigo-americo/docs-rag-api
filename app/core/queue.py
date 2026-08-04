import asyncio
import json
import uuid
from typing import Protocol

import redis.asyncio as redis

from app.core.config import settings

_PROCESSING_SUFFIX = ":processing"


def get_redis_client() -> redis.Redis:
    """Cria um client Redis a partir da redis_url configurada.

    from_url faz o parse do host/porta/db a partir da URL
    (redis://host:porta/db) — não precisamos montar isso na mão.
    decode_responses=True faz o client devolver str em vez de bytes,
    já que só trafegamos JSON de texto.
    """
    return redis.from_url(settings.redis_url, decode_responses=True)


async def enqueue(client: redis.Redis, document_id: uuid.UUID, attempt: int = 0) -> None:
    """Publica uma mensagem de ingest na fila principal.

    RPUSH insere no final da lista — combinado com RPOPLPUSH (que tira do
    final), a fila funciona em ordem FIFO: primeiro a entrar, primeiro a
    sair.
    """
    payload = json.dumps({"document_id": str(document_id), "attempt": attempt})
    await client.rpush(settings.ingest_queue_name, payload)


async def dequeue(client: redis.Redis, timeout: int = 5) -> tuple[dict, str] | None:
    """Consome uma mensagem da fila principal com segurança contra perda.

    BRPOPLPUSH faz, atomicamente: tira o item mais antigo da fila
    principal e coloca na fila "<nome>:processing" — nunca existe um
    instante em que a mensagem não esteja em NENHUMA lista. Se o processo
    que consumiu morrer antes de terminar o trabalho, a mensagem continua
    visível em "<nome>:processing" (ninguém a perdeu, só ninguém a
    removeu de lá ainda).

    `timeout` em segundos: quanto tempo esperar por uma mensagem antes de
    desistir e devolver None. Sem timeout, um BLPOP/BRPOPLPUSH bloqueia
    pra sempre — não dá pra testar isso numa suite automatizada sem travar.
    """
    processing_queue = settings.ingest_queue_name + _PROCESSING_SUFFIX
    raw_payload = await client.brpoplpush(
        settings.ingest_queue_name, processing_queue, timeout=timeout
    )
    if raw_payload is None:
        return None
    return json.loads(raw_payload), raw_payload


async def queue_depths(client: redis.Redis) -> dict[str, int]:
    """Tamanho atual das duas listas Redis que compõem a fila: "pending"
    (aguardando um worker) e "processing" (já tirada por um worker via
    BRPOPLPUSH, ainda sem ack — ver dequeue()). Usado pelo endpoint /metrics
    para popular o Gauge ingest_queue_depth no momento do scrape.
    """
    pending, processing = await asyncio.gather(
        client.llen(settings.ingest_queue_name),
        client.llen(settings.ingest_queue_name + _PROCESSING_SUFFIX),
    )
    return {"pending": pending, "processing": processing}


async def ack(client: redis.Redis, raw_payload: str) -> None:
    """Confirma que uma mensagem foi processada com sucesso, removendo-a
    da fila "<nome>:processing".

    Precisa do payload exato (a string original, não o dict já decodificado)
    porque LREM compara por igualdade de valor na lista Redis.
    """
    processing_queue = settings.ingest_queue_name + _PROCESSING_SUFFIX
    await client.lrem(processing_queue, count=1, value=raw_payload)


class QueueClient(Protocol):
    """Abstração do lado do produtor: 'como o ingest_document manda um
    documento pra ser processado'. RedisQueueClient enfileira de verdade;
    InlineQueueClient processa na hora, sem fila — usado em dev/teste
    (queue_backend="inline"), equivalente ao BackgroundTasks de antes.
    """

    async def enqueue(self, document_id: uuid.UUID) -> None: ...


class RedisQueueClient:
    def __init__(self, client: redis.Redis) -> None:
        self._client = client

    async def enqueue(self, document_id: uuid.UUID) -> None:
        await enqueue(self._client, document_id, attempt=0)


class InlineQueueClient:
    """Processa o documento na hora, no mesmo processo — sem Redis.
    Usado quando settings.queue_backend="inline" (default em dev/teste).
    """

    def __init__(self, ingest_service) -> None:
        self._service = ingest_service

    async def enqueue(self, document_id: uuid.UUID) -> None:
        await self._service.process_document(document_id)
