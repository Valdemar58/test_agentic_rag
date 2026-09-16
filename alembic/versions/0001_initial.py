"""initial: пользователи, диалоги, сообщения, элементы, фидбэк, реестр инжеста (FR-9)

Revision ID: 0001
Revises:
Create Date: 2026-09-15

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0001"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Схема с нуля: сгенерирована autogenerate по src/db/models.py и проверена вручную."""
    op.create_table(
        "app_user",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("identifier", sa.String(length=200), nullable=False),
        sa.Column("metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("app_user_pkey")),
        sa.UniqueConstraint("identifier", name=op.f("uq_app_user_identifier")),
    )
    op.create_table(
        "ingest_run",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("corpus_path", sa.String(length=1000), nullable=False),
        sa.Column("synthetic", sa.Boolean(), nullable=False),
        sa.Column("outcome", sa.String(length=20), nullable=False),
        sa.Column("files_total", sa.Integer(), nullable=False),
        sa.Column("files_indexed", sa.Integer(), nullable=False),
        sa.Column("files_skipped", sa.Integer(), nullable=False),
        sa.Column("files_failed", sa.Integer(), nullable=False),
        sa.Column("files_unchanged", sa.Integer(), nullable=False),
        sa.Column("chunks_total", sa.Integer(), nullable=False),
        sa.Column("error_text", sa.Text(), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id", name=op.f("ingest_run_pkey")),
    )
    op.create_index(op.f("ix_ingest_run_outcome"), "ingest_run", ["outcome"], unique=False)
    op.create_table(
        "conversation",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("user_id", sa.Uuid(), nullable=True),
        sa.Column("title", sa.String(length=500), nullable=True),
        sa.Column("tags", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(
            ["user_id"], ["app_user.id"], name=op.f("fk_conversation_user_id_app_user"), ondelete="SET NULL"
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("conversation_pkey")),
    )
    op.create_index(op.f("ix_conversation_created_at"), "conversation", ["created_at"], unique=False)
    op.create_index(op.f("ix_conversation_user_id"), "conversation", ["user_id"], unique=False)
    op.create_table(
        "indexed_file",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("card_id", sa.Uuid(), nullable=False),
        sa.Column("file_row_id", sa.Uuid(), nullable=False),
        sa.Column("relative_path", sa.String(length=1000), nullable=False),
        sa.Column("sha256", sa.String(length=64), nullable=False),
        sa.Column("size_bytes", sa.Integer(), nullable=False),
        sa.Column("extension", sa.String(length=20), nullable=False),
        sa.Column("file_role", sa.String(length=30), nullable=True),
        sa.Column("parse_route", sa.String(length=20), nullable=True),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("reason", sa.Text(), nullable=True),
        sa.Column("chunk_count", sa.Integer(), nullable=False),
        sa.Column("doc_status", sa.String(length=20), nullable=True),
        sa.Column("last_run_id", sa.Integer(), nullable=True),
        sa.Column("indexed_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(
            ["last_run_id"],
            ["ingest_run.id"],
            name=op.f("fk_indexed_file_last_run_id_ingest_run"),
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("indexed_file_pkey")),
        sa.UniqueConstraint("card_id", "file_row_id", name="uq_indexed_file_card_file"),
    )
    op.create_index(op.f("ix_indexed_file_card_id"), "indexed_file", ["card_id"], unique=False)
    op.create_index(op.f("ix_indexed_file_last_run_id"), "indexed_file", ["last_run_id"], unique=False)
    op.create_index(op.f("ix_indexed_file_sha256"), "indexed_file", ["sha256"], unique=False)
    op.create_index(op.f("ix_indexed_file_status"), "indexed_file", ["status"], unique=False)
    op.create_table(
        "message",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("conversation_id", sa.Uuid(), nullable=False),
        sa.Column("parent_id", sa.Uuid(), nullable=True),
        sa.Column("kind", sa.String(length=30), nullable=False),
        sa.Column("name", sa.String(length=200), nullable=False),
        sa.Column("input_text", sa.Text(), nullable=True),
        sa.Column("output_text", sa.Text(), nullable=True),
        sa.Column("is_error", sa.Boolean(), nullable=False),
        sa.Column("show_input", sa.String(length=20), nullable=True),
        sa.Column("language", sa.String(length=50), nullable=True),
        sa.Column("trace_id", sa.String(length=100), nullable=True),
        sa.Column("metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(
            ["conversation_id"],
            ["conversation.id"],
            name=op.f("fk_message_conversation_id_conversation"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["parent_id"], ["message.id"], name=op.f("fk_message_parent_id_message"), ondelete="SET NULL"
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("message_pkey")),
    )
    op.create_index(op.f("ix_message_conversation_id"), "message", ["conversation_id"], unique=False)
    op.create_index(op.f("ix_message_created_at"), "message", ["created_at"], unique=False)
    op.create_index(op.f("ix_message_kind"), "message", ["kind"], unique=False)
    op.create_index(op.f("ix_message_trace_id"), "message", ["trace_id"], unique=False)
    op.create_table(
        "element",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("conversation_id", sa.Uuid(), nullable=False),
        sa.Column("message_id", sa.Uuid(), nullable=True),
        sa.Column("name", sa.String(length=500), nullable=False),
        sa.Column("kind", sa.String(length=30), nullable=False),
        sa.Column("mime", sa.String(length=100), nullable=True),
        sa.Column("display", sa.String(length=20), nullable=True),
        sa.Column("content", sa.Text(), nullable=True),
        sa.Column("url", sa.String(length=2000), nullable=True),
        sa.Column("object_key", sa.String(length=500), nullable=True),
        sa.Column("size", sa.String(length=20), nullable=True),
        sa.Column("language", sa.String(length=50), nullable=True),
        sa.Column("page", sa.Integer(), nullable=True),
        sa.Column("chainlit_key", sa.String(length=200), nullable=True),
        sa.Column("props", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(
            ["conversation_id"],
            ["conversation.id"],
            name=op.f("fk_element_conversation_id_conversation"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["message_id"], ["message.id"], name=op.f("fk_element_message_id_message"), ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("element_pkey")),
    )
    op.create_index(op.f("ix_element_conversation_id"), "element", ["conversation_id"], unique=False)
    op.create_index(op.f("ix_element_message_id"), "element", ["message_id"], unique=False)
    op.create_table(
        "feedback",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("message_id", sa.Uuid(), nullable=False),
        sa.Column("conversation_id", sa.Uuid(), nullable=False),
        sa.Column("value", sa.Integer(), nullable=False),
        sa.Column("comment", sa.Text(), nullable=True),
        sa.Column("trace_id", sa.String(length=100), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(
            ["conversation_id"],
            ["conversation.id"],
            name=op.f("fk_feedback_conversation_id_conversation"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["message_id"], ["message.id"], name=op.f("fk_feedback_message_id_message"), ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("feedback_pkey")),
        sa.UniqueConstraint("message_id", name=op.f("uq_feedback_message_id")),
    )
    op.create_index(op.f("ix_feedback_conversation_id"), "feedback", ["conversation_id"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_feedback_conversation_id"), table_name="feedback")
    op.drop_table("feedback")
    op.drop_index(op.f("ix_element_message_id"), table_name="element")
    op.drop_index(op.f("ix_element_conversation_id"), table_name="element")
    op.drop_table("element")
    op.drop_index(op.f("ix_message_trace_id"), table_name="message")
    op.drop_index(op.f("ix_message_kind"), table_name="message")
    op.drop_index(op.f("ix_message_created_at"), table_name="message")
    op.drop_index(op.f("ix_message_conversation_id"), table_name="message")
    op.drop_table("message")
    op.drop_index(op.f("ix_indexed_file_status"), table_name="indexed_file")
    op.drop_index(op.f("ix_indexed_file_sha256"), table_name="indexed_file")
    op.drop_index(op.f("ix_indexed_file_last_run_id"), table_name="indexed_file")
    op.drop_index(op.f("ix_indexed_file_card_id"), table_name="indexed_file")
    op.drop_table("indexed_file")
    op.drop_index(op.f("ix_conversation_user_id"), table_name="conversation")
    op.drop_index(op.f("ix_conversation_created_at"), table_name="conversation")
    op.drop_table("conversation")
    op.drop_index(op.f("ix_ingest_run_outcome"), table_name="ingest_run")
    op.drop_table("ingest_run")
    op.drop_table("app_user")
