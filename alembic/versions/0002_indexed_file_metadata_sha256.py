"""indexed_file.metadata_sha256: отпечаток метаданных карточки для инкрементального инжеста (FR-3, §4)

Revision ID: 0002
Revises: 0001
Create Date: 2026-09-15

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0002"
down_revision: str | Sequence[str] | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("indexed_file", sa.Column("metadata_sha256", sa.String(length=64), nullable=True))


def downgrade() -> None:
    op.drop_column("indexed_file", "metadata_sha256")
