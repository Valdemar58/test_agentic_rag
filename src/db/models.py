"""Модели прикладной БД (FR-9): диалоги, сообщения, элементы, фидбэк, реестр инжеста.

Структура диалогов повторяет сущности data layer Chainlit (пользователь → диалог → шаг →
элемент, фидбэк к шагу), чтобы UI на этапе 7 хранил историю в этих таблицах без своей БД.
Реестр `indexed_file` — основа инкрементального инжеста (FR-3: повторный запуск обрабатывает
только новые и изменённые файлы по хэшу). Векторы здесь не хранятся (Qdrant).
"""

from __future__ import annotations

import datetime as dt
import uuid
from typing import Any

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    Uuid,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from db.base import Base

# Значения строковых «перечислений» — в коде, а не в типе ENUM PostgreSQL: проще мигрировать.
MESSAGE_KINDS = ("user_message", "assistant_message", "system_message", "run", "tool", "llm", "retrieval")
FILE_STATUSES = ("indexed", "skipped", "error")
PARSE_ROUTES = ("native", "vlm")
RUN_OUTCOMES = ("running", "success", "partial", "failed")


def _utcnow() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


class AppUser(Base):
    """Пользователь UI. В MVP один пользователь; таблица нужна data layer Chainlit."""

    __tablename__ = "app_user"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    identifier: Mapped[str] = mapped_column(String(200), nullable=False, unique=True)
    metadata_json: Mapped[dict[str, Any]] = mapped_column("metadata", JSONB, nullable=False, default=dict)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow, server_default=func.now()
    )

    conversations: Mapped[list[Conversation]] = relationship(back_populates="user")


class Conversation(Base):
    """Диалог (thread в терминах Chainlit)."""

    __tablename__ = "conversation"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("app_user.id", ondelete="SET NULL"), nullable=True, index=True
    )
    title: Mapped[str | None] = mapped_column(String(500), nullable=True)
    tags: Mapped[list[str]] = mapped_column(JSONB, nullable=False, default=list)
    metadata_json: Mapped[dict[str, Any]] = mapped_column("metadata", JSONB, nullable=False, default=dict)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow, server_default=func.now(), index=True
    )
    updated_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=_utcnow,
        server_default=func.now(),
        onupdate=_utcnow,
    )

    user: Mapped[AppUser | None] = relationship(back_populates="conversations")
    messages: Mapped[list[Message]] = relationship(
        back_populates="conversation", cascade="all, delete-orphan", order_by="Message.created_at"
    )


class Message(Base):
    """Сообщение или шаг агента (step в терминах Chainlit): вопрос, ответ, вызов инструмента."""

    __tablename__ = "message"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("conversation.id", ondelete="CASCADE"), nullable=False, index=True
    )
    parent_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("message.id", ondelete="SET NULL"), nullable=True
    )
    kind: Mapped[str] = mapped_column(String(30), nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False, default="")
    input_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    output_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    is_error: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    show_input: Mapped[str | None] = mapped_column(String(20), nullable=True)
    language: Mapped[str | None] = mapped_column(String(50), nullable=True)
    trace_id: Mapped[str | None] = mapped_column(String(100), nullable=True, index=True)
    metadata_json: Mapped[dict[str, Any]] = mapped_column("metadata", JSONB, nullable=False, default=dict)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow, server_default=func.now(), index=True
    )
    started_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    conversation: Mapped[Conversation] = relationship(back_populates="messages")
    elements: Mapped[list[Element]] = relationship(back_populates="message", cascade="all, delete-orphan")
    feedback: Mapped[Feedback | None] = relationship(
        back_populates="message", cascade="all, delete-orphan", uselist=False
    )


class Element(Base):
    """Элемент, показанный в диалоге: текст процитированного фрагмента, карточка, файл (FR-4)."""

    __tablename__ = "element"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("conversation.id", ondelete="CASCADE"), nullable=False, index=True
    )
    message_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("message.id", ondelete="CASCADE"), nullable=True, index=True
    )
    name: Mapped[str] = mapped_column(String(500), nullable=False)
    kind: Mapped[str] = mapped_column(String(30), nullable=False)
    mime: Mapped[str | None] = mapped_column(String(100), nullable=True)
    display: Mapped[str | None] = mapped_column(String(20), nullable=True)
    content: Mapped[str | None] = mapped_column(Text, nullable=True)
    url: Mapped[str | None] = mapped_column(String(2000), nullable=True)
    object_key: Mapped[str | None] = mapped_column(String(500), nullable=True)
    size: Mapped[str | None] = mapped_column(String(20), nullable=True)
    language: Mapped[str | None] = mapped_column(String(50), nullable=True)
    page: Mapped[int | None] = mapped_column(Integer, nullable=True)
    chainlit_key: Mapped[str | None] = mapped_column(String(200), nullable=True)
    props: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow, server_default=func.now()
    )

    message: Mapped[Message | None] = relationship(back_populates="elements")


class Feedback(Base):
    """Оценка ответа 👍/👎 (FR-7); связана с трейсом Langfuse. Логики поверх фидбэка в MVP нет (§7)."""

    __tablename__ = "feedback"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    message_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("message.id", ondelete="CASCADE"), nullable=False, unique=True
    )
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("conversation.id", ondelete="CASCADE"), nullable=False, index=True
    )
    value: Mapped[int] = mapped_column(Integer, nullable=False)
    comment: Mapped[str | None] = mapped_column(Text, nullable=True)
    trace_id: Mapped[str | None] = mapped_column(String(100), nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow, server_default=func.now()
    )
    updated_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=_utcnow,
        server_default=func.now(),
        onupdate=_utcnow,
    )

    message: Mapped[Message] = relationship(back_populates="feedback")


class IngestRun(Base):
    """Прогон инжеста: итоги по файлам и чанкам, пометка синтетических данных (§9)."""

    __tablename__ = "ingest_run"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    corpus_path: Mapped[str] = mapped_column(String(1000), nullable=False)
    synthetic: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    outcome: Mapped[str] = mapped_column(String(20), nullable=False, default="running", index=True)
    files_total: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    files_indexed: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    files_skipped: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    files_failed: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    files_unchanged: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    chunks_total: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    error_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    started_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow, server_default=func.now()
    )
    finished_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    files: Mapped[list[IndexedFile]] = relationship(back_populates="last_run")


class IndexedFile(Base):
    """Реестр проиндексированных файлов с хэшами (FR-3, FR-9): один файл карточки — одна запись."""

    __tablename__ = "indexed_file"
    __table_args__ = (UniqueConstraint("card_id", "file_row_id", name="uq_indexed_file_card_file"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    card_id: Mapped[uuid.UUID] = mapped_column(Uuid, nullable=False, index=True)
    file_row_id: Mapped[uuid.UUID] = mapped_column(Uuid, nullable=False)
    relative_path: Mapped[str] = mapped_column(String(1000), nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    extension: Mapped[str] = mapped_column(String(20), nullable=False)
    file_role: Mapped[str | None] = mapped_column(String(30), nullable=True)
    parse_route: Mapped[str | None] = mapped_column(String(20), nullable=True)
    status: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    chunk_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    doc_status: Mapped[str | None] = mapped_column(String(20), nullable=True)
    # Отпечаток метаданных карточки и роли файла: при том же sha256 файла, но изменившейся карточке
    # (например, приказ отменён) обновляется только payload в Qdrant, без повторного разбора
    metadata_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    last_run_id: Mapped[int | None] = mapped_column(
        ForeignKey("ingest_run.id", ondelete="SET NULL"), nullable=True, index=True
    )
    indexed_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow, server_default=func.now()
    )

    last_run: Mapped[IngestRun | None] = relationship(back_populates="files")
