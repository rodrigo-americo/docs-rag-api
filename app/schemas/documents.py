import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class IngestResponse(BaseModel):
    document_id: uuid.UUID = Field(..., description="ID gerado para o documento ingerido")
    title: str = Field(
        ..., description="Título do documento (informado ou derivado do nome do arquivo)"
    )
    chunks_created: int = Field(..., description="Número de chunks indexados")
    status: Literal["indexed"] = Field("indexed", description="Estado do documento após o ingest")


class DocumentSummary(BaseModel):
    id: uuid.UUID = Field(..., description="ID do documento")
    title: str = Field(..., description="Título do documento")
    source_filename: str | None = Field(..., description="Nome do arquivo original")
    created_at: datetime = Field(..., description="Data de ingestão do arquivo")
