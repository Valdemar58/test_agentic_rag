"""Хранилище диалогов UI (FR-7, FR-9): пользователи, диалоги, сообщения-шаги, элементы, фидбэк.

Только ORM SQLAlchemy поверх моделей `db.models`, сырого SQL нет (AC-9.2). Записи шагов и элементов —
upsert конструкцией PostgreSQL `INSERT … ON CONFLICT`: Chainlit пишет их фоновыми задачами, и создание
с обновлением одного шага могут выполняться параллельно. Модуль не зависит от Chainlit: его типы
переводятся в эти вызовы в `ui.data_layer`.
"""

from __future__ import annotations

import datetime as dt
import uuid
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import delete, exists, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession
from sqlalchemy.orm import selectinload

from common.settings import Settings
from db.models import AppUser, Conversation, Element, Feedback, Message
from db.session import build_engine, build_sessionmaker, session_scope

NAME_LIMIT = 200


def utcnow() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


@dataclass(frozen=True)
class MessageValues:
    """Поля сообщения или шага агента (step в терминах Chainlit) для записи."""

    id: uuid.UUID
    conversation_id: uuid.UUID
    kind: str
    name: str = ""
    parent_id: uuid.UUID | None = None
    input_text: str | None = None
    output_text: str | None = None
    is_error: bool = False
    show_input: str | None = None
    language: str | None = None
    trace_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: dt.datetime | None = None
    started_at: dt.datetime | None = None
    finished_at: dt.datetime | None = None


@dataclass(frozen=True)
class ElementValues:
    """Поля элемента диалога (текст процитированного фрагмента, карточка) для записи."""

    id: uuid.UUID
    conversation_id: uuid.UUID
    name: str
    kind: str
    message_id: uuid.UUID | None = None
    mime: str | None = None
    display: str | None = None
    content: str | None = None
    url: str | None = None
    object_key: str | None = None
    size: str | None = None
    language: str | None = None
    page: int | None = None
    chainlit_key: str | None = None
    props: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LoadedConversation:
    conversation: Conversation
    user_identifier: str | None
    messages: list[Message]
    elements: list[Element]


@dataclass(frozen=True)
class ConversationPage:
    items: list[LoadedConversation]
    has_next: bool


class ConversationStore:
    """Операции над таблицами диалогов; каждая — своя транзакция."""

    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine
        self._factory = build_sessionmaker(engine)

    @classmethod
    def from_settings(cls, settings: Settings) -> ConversationStore:
        return cls(build_engine(settings.database_url))

    async def close(self) -> None:
        await self._engine.dispose()

    # ---------- пользователи ----------

    async def get_user(self, identifier: str) -> AppUser | None:
        async with self._factory() as session:
            result = await session.execute(select(AppUser).where(AppUser.identifier == identifier))
            return result.scalar_one_or_none()

    async def upsert_user(self, identifier: str, metadata: dict[str, Any]) -> AppUser:
        async with session_scope(self._factory) as session:
            statement = (
                insert(AppUser)
                .values(
                    {
                        AppUser.id: uuid.uuid4(),
                        AppUser.identifier: identifier,
                        AppUser.metadata_json: metadata,
                    }
                )
                .on_conflict_do_update(index_elements=["identifier"], set_={AppUser.metadata_json: metadata})
            )
            await session.execute(statement)
            result = await session.execute(select(AppUser).where(AppUser.identifier == identifier))
            return result.scalar_one()

    async def user_identifier(self, user_id: uuid.UUID) -> str | None:
        async with self._factory() as session:
            result = await session.execute(select(AppUser.identifier).where(AppUser.id == user_id))
            return result.scalar_one_or_none()

    # ---------- диалоги ----------

    async def _ensure_conversation(self, session: AsyncSession, conversation_id: uuid.UUID) -> None:
        statement = (
            insert(Conversation)
            .values({Conversation.id: conversation_id, Conversation.tags: [], Conversation.metadata_json: {}})
            .on_conflict_do_nothing(index_elements=["id"])
        )
        await session.execute(statement)

    async def upsert_conversation(
        self,
        conversation_id: uuid.UUID,
        *,
        title: str | None = None,
        user_id: uuid.UUID | None = None,
        metadata: dict[str, Any] | None = None,
        tags: list[str] | None = None,
    ) -> Conversation:
        """Создаёт диалог или обновляет переданные поля; в metadata значение None удаляет ключ."""
        async with session_scope(self._factory) as session:
            await self._ensure_conversation(session, conversation_id)
            conversation = await session.get(Conversation, conversation_id)
            assert conversation is not None
            if title is not None:
                conversation.title = title[:500]
            if user_id is not None:
                conversation.user_id = user_id
            if tags is not None:
                conversation.tags = list(tags)
            if metadata:
                merged = {key: value for key, value in conversation.metadata_json.items()}
                for key, value in metadata.items():
                    if value is None:
                        merged.pop(key, None)
                    else:
                        merged[key] = value
                conversation.metadata_json = merged
            conversation.updated_at = utcnow()
            await session.flush()
            return conversation

    async def _load(self, session: AsyncSession, conversation: Conversation) -> LoadedConversation:
        messages = await session.execute(
            select(Message)
            .where(Message.conversation_id == conversation.id)
            .options(selectinload(Message.feedback))
            .order_by(Message.created_at, Message.id)
        )
        elements = await session.execute(
            select(Element).where(Element.conversation_id == conversation.id).order_by(Element.created_at)
        )
        return LoadedConversation(
            conversation=conversation,
            user_identifier=conversation.user.identifier if conversation.user is not None else None,
            messages=list(messages.scalars()),
            elements=list(elements.scalars()),
        )

    async def get_conversation(self, conversation_id: uuid.UUID) -> LoadedConversation | None:
        async with self._factory() as session:
            result = await session.execute(
                select(Conversation)
                .where(Conversation.id == conversation_id)
                .options(selectinload(Conversation.user))
            )
            conversation = result.scalar_one_or_none()
            if conversation is None:
                return None
            return await self._load(session, conversation)

    async def conversation_author(self, conversation_id: uuid.UUID) -> str | None:
        async with self._factory() as session:
            result = await session.execute(
                select(AppUser.identifier)
                .join(Conversation, Conversation.user_id == AppUser.id)
                .where(Conversation.id == conversation_id)
            )
            return result.scalar_one_or_none()

    async def list_conversations(
        self,
        user_id: uuid.UUID,
        *,
        limit: int,
        cursor: uuid.UUID | None = None,
        search: str | None = None,
        feedback: int | None = None,
    ) -> ConversationPage:
        """Диалоги пользователя от свежих к старым; страница начинается после диалога `cursor`."""
        async with self._factory() as session:
            statement = (
                select(Conversation)
                .where(Conversation.user_id == user_id)
                .options(selectinload(Conversation.user))
                .order_by(Conversation.updated_at.desc(), Conversation.id)
            )
            if search:
                statement = statement.where(
                    exists().where(
                        Message.conversation_id == Conversation.id, Message.output_text.ilike(f"%{search}%")
                    )
                )
            if feedback is not None:
                statement = statement.where(
                    exists().where(Feedback.conversation_id == Conversation.id, Feedback.value == feedback)
                )
            rows = list((await session.execute(statement)).scalars())
            start = 0
            if cursor is not None:
                for index, row in enumerate(rows):
                    if row.id == cursor:
                        start = index + 1
                        break
            page = rows[start : start + limit]
            loaded = [await self._load(session, row) for row in page]
            return ConversationPage(items=loaded, has_next=len(rows) > start + limit)

    async def delete_conversation(self, conversation_id: uuid.UUID) -> None:
        async with session_scope(self._factory) as session:
            await session.execute(delete(Conversation).where(Conversation.id == conversation_id))

    # ---------- сообщения и шаги ----------

    async def upsert_message(self, values: MessageValues) -> None:
        async with session_scope(self._factory) as session:
            await self._ensure_conversation(session, values.conversation_id)
            row: dict[Any, Any] = {
                Message.id: values.id,
                Message.conversation_id: values.conversation_id,
                Message.parent_id: values.parent_id,
                Message.kind: values.kind,
                Message.name: values.name[:NAME_LIMIT],
                Message.input_text: values.input_text,
                Message.output_text: values.output_text,
                Message.is_error: values.is_error,
                Message.show_input: values.show_input,
                Message.language: values.language,
                Message.trace_id: values.trace_id,
                Message.metadata_json: values.metadata,
                Message.started_at: values.started_at,
                Message.finished_at: values.finished_at,
            }
            if values.created_at is not None:
                row[Message.created_at] = values.created_at
            updates = {
                column: value
                for column, value in row.items()
                if column not in (Message.id, Message.conversation_id, Message.created_at)
            }
            statement = insert(Message).values(row).on_conflict_do_update(index_elements=["id"], set_=updates)
            await session.execute(statement)
            conversation = await session.get(Conversation, values.conversation_id)
            if conversation is not None:
                conversation.updated_at = utcnow()

    async def delete_message(self, message_id: uuid.UUID) -> None:
        async with session_scope(self._factory) as session:
            await session.execute(delete(Element).where(Element.message_id == message_id))
            await session.execute(delete(Message).where(Message.id == message_id))

    async def _trace_id_for(self, session: AsyncSession, message: Message) -> str | None:
        """Трейс хода: у самого сообщения или у последнего сообщения ассистента внутри него.

        Chainlit привязывает оценку к run-шагу хода, а `trace_id` лежит в metadata ответа ассистента —
        дочернего шага этого run."""
        if message.trace_id:
            return message.trace_id
        result = await session.execute(
            select(Message.trace_id)
            .where(Message.parent_id == message.id, Message.trace_id.is_not(None))
            .order_by(Message.created_at.desc())
            .limit(1)
        )
        return result.scalar_one_or_none()

    async def message_trace_id(self, message_id: uuid.UUID) -> str | None:
        async with self._factory() as session:
            message = await session.get(Message, message_id)
            return await self._trace_id_for(session, message) if message is not None else None

    # ---------- элементы ----------

    async def upsert_element(self, values: ElementValues) -> None:
        async with session_scope(self._factory) as session:
            await self._ensure_conversation(session, values.conversation_id)
            row: dict[Any, Any] = {
                Element.id: values.id,
                Element.conversation_id: values.conversation_id,
                Element.message_id: values.message_id,
                Element.name: values.name[:500],
                Element.kind: values.kind,
                Element.mime: values.mime,
                Element.display: values.display,
                Element.content: values.content,
                Element.url: values.url,
                Element.object_key: values.object_key,
                Element.size: values.size,
                Element.language: values.language,
                Element.page: values.page,
                Element.chainlit_key: values.chainlit_key,
                Element.props: values.props,
            }
            updates = {
                column: value
                for column, value in row.items()
                if column not in (Element.id, Element.conversation_id)
            }
            statement = insert(Element).values(row).on_conflict_do_update(index_elements=["id"], set_=updates)
            await session.execute(statement)

    async def get_element(self, conversation_id: uuid.UUID, element_id: uuid.UUID) -> Element | None:
        async with self._factory() as session:
            result = await session.execute(
                select(Element).where(Element.id == element_id, Element.conversation_id == conversation_id)
            )
            return result.scalar_one_or_none()

    async def delete_element(self, element_id: uuid.UUID) -> None:
        async with session_scope(self._factory) as session:
            await session.execute(delete(Element).where(Element.id == element_id))

    # ---------- фидбэк (FR-7): одна оценка на сообщение, trace_id из сообщения ----------

    async def upsert_feedback(
        self,
        *,
        message_id: uuid.UUID,
        value: int,
        comment: str | None,
        feedback_id: uuid.UUID | None = None,
    ) -> Feedback:
        async with session_scope(self._factory) as session:
            message = await session.get(Message, message_id)
            if message is None:
                raise LookupError(f"сообщение {message_id} не найдено")
            trace_id = await self._trace_id_for(session, message)
            result = await session.execute(select(Feedback).where(Feedback.message_id == message_id))
            feedback = result.scalar_one_or_none()
            if feedback is None:
                feedback = Feedback(
                    id=feedback_id or uuid.uuid4(),
                    message_id=message_id,
                    conversation_id=message.conversation_id,
                    value=value,
                    comment=comment,
                    trace_id=trace_id,
                )
                session.add(feedback)
            else:
                feedback.value = value
                feedback.comment = comment
                feedback.trace_id = trace_id
            await session.flush()
            return feedback

    async def delete_feedback(self, feedback_id: uuid.UUID) -> None:
        async with session_scope(self._factory) as session:
            await session.execute(delete(Feedback).where(Feedback.id == feedback_id))
