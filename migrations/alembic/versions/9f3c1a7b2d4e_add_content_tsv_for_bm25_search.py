"""add content_tsv generated column and gin index for bm25 search

Revision ID: 9f3c1a7b2d4e
Revises: 652a8bda6c86
Create Date: 2026-08-03 00:00:00.000000
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import TSVECTOR

revision: str = "9f3c1a7b2d4e"
down_revision: Union[str, Sequence[str], None] = "652a8bda6c86"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Coluna gerada: o próprio Postgres mantém content_tsv sincronizada com
    # content, sem precisar reindexar manualmente durante o ingest.
    op.execute(
        """
        ALTER TABLE chunks
        ADD COLUMN content_tsv tsvector
        GENERATED ALWAYS AS (to_tsvector('portuguese', content)) STORED
        """
    )

    op.create_index(
        "chunks_content_tsv_gin_idx",
        "chunks",
        ["content_tsv"],
        postgresql_using="gin",
    )


def downgrade() -> None:
    op.drop_index("chunks_content_tsv_gin_idx", table_name="chunks")
    op.drop_column("chunks", "content_tsv")
