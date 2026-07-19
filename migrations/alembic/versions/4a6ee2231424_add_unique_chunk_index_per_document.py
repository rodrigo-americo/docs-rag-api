"""add unique constraint on (document_id, chunk_index)

Revision ID: 4a6ee2231424
Revises: e30d2d2cd779
Create Date: 2026-07-18 00:00:00.000000
"""
from typing import Sequence, Union

from alembic import op

revision: str = "4a6ee2231424"
down_revision: Union[str, None] = "e30d2d2cd779"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_unique_constraint(
        "uq_chunks_document_id_chunk_index",
        "chunks",
        ["document_id", "chunk_index"],
    )


def downgrade() -> None:
    op.drop_constraint(
        "uq_chunks_document_id_chunk_index",
        "chunks",
        type_="unique",
    )
