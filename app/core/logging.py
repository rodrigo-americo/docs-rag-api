import logging
import sys

import structlog
from structlog.types import Processor

from app.core.config import settings


def configure_logging() -> None:
    """Configura structlog + stdlib logging em modo unificado.

    Chamada uma vez no startup do app (no lifespan).
    """
    # Pipeline de processors compartilhado entre structlog e stdlib.
    # Cada processor recebe o event dict e devolve modificado.
    shared_processors: list[Processor] = [
        # Adiciona timestamp ISO 8601 com UTC.
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        # Adiciona level (info/warning/etc) no event dict.
        structlog.stdlib.add_log_level,
        # Adiciona o logger name (módulo de origem).
        structlog.stdlib.add_logger_name,
        # Resolve exc_info pra texto formatado.
        structlog.processors.format_exc_info,
        # Adiciona stack info se requisitado.
        structlog.processors.StackInfoRenderer(),
    ]

    # Renderer final muda conforme formato escolhido.
    if settings.log_format == "json":
        renderer: Processor = structlog.processors.JSONRenderer()
    else:
        # ConsoleRenderer: colorido, indentado, mais legível pra dev.
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    # Configura o structlog "core".
    structlog.configure(
        processors=[
            *shared_processors,
            # Prepara pro renderer final.
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        # BoundLogger: contexto fica grudado entre chamadas.
        wrapper_class=structlog.stdlib.BoundLogger,
        # Usa logging.Logger por baixo (integração stdlib).
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Configura o stdlib pra usar o mesmo formatter.
    # Resultado: logs do uvicorn/SQLAlchemy passam pelo mesmo pipeline.
    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            # Remove o _record (interno do stdlib) antes do render final.
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    # Handler único: stdout. Containers/Kubernetes leem daí.
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    # Remove handlers default (uvicorn adiciona uns sem ser pedido).
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(settings.log_level.upper())

    # Silencia chatice excessiva de libs específicas.
    # SQLAlchemy.engine no INFO loga TODA query (útil pra debug, ruim em prod).
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
    # uvicorn.access loga cada request — útil em dev, redundante em prod
    # se você tiver access logs de proxy reverso. Por ora, deixa em INFO.


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Retorna um logger structlog.

    Uso típico: `log = get_logger(__name__)` no topo de cada módulo.
    """
    return structlog.get_logger(name)
