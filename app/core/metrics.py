from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

# Registry próprio (não o REGISTRY global do prometheus_client): evita estado
# compartilhado entre testes (cada import do módulo em processos diferentes —
# api, worker, test — parte de um registry limpo, sem risco de "Duplicated
# timeseries" ao recarregar o módulo em suites que reimportam app.*).
registry = CollectorRegistry()

ingest_processed_total = Counter(
    "ingest_processed_total",
    "Documentos processados pelo pipeline de ingest, por status final",
    labelnames=["status"],
    registry=registry,
)

ingest_processing_seconds = Histogram(
    "ingest_processing_seconds",
    "Tempo de process_document (parse + embedding + persist), em segundos",
    registry=registry,
)

# Gauge com valor "puxado" (set() no momento do scrape via /metrics), não
# incrementado por evento — profundidade de fila é um estado atual do Redis,
# não algo que a aplicação observa acontecer. Ver app.api.health:metrics().
ingest_queue_depth = Gauge(
    "ingest_queue_depth",
    "Mensagens na fila de ingest, por lista (pending ou processing)",
    labelnames=["queue"],
    registry=registry,
)
