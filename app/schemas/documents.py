import uuid
from datetime import datetime

from pydantic import BaseModel, Field

from app.models.document import DocumentStatus


class IngestResponse(BaseModel):
    document_id: uuid.UUID = Field(..., description="ID gerado para o documento ingerido")
    title: str = Field(
        ..., description="Título do documento (informado ou derivado do nome do arquivo)"
    )
    chunks_created: int = Field(..., description="Número de chunks indexados até o momento")
    status: DocumentStatus = Field(
        ..., description="Estado do processamento — consulte GET /documents/{id} para acompanhar"
    )


class DocumentSummary(BaseModel):
    id: uuid.UUID = Field(..., description="ID do documento")
    title: str = Field(..., description="Título do documento")
    source_filename: str | None = Field(..., description="Nome do arquivo original")
    status: DocumentStatus = Field(..., description="Estado do processamento do documento")
    created_at: datetime = Field(..., description="Data de ingestão do arquivo")
