import uuid
from typing import Literal

from pydantic import BaseModel, Field


class IngestResponse(BaseModel):
    document_id: uuid.UUID
    title: str
    chunks_created: int = Field(..., description="Número de chunks indexados")
    status: Literal["indexed"] = "indexed"