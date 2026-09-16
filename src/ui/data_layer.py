"""Data layer Chainlit поверх хранилища диалогов (FR-7): список диалогов, шаги, элементы, фидбэк в БД проекта.

Здесь только перевод типов Chainlit (`StepDict`, `ElementDict`, `ThreadDict`, `Feedback`) в вызовы
`ConversationStore` и обратно; таблицы и запросы — в `db.models` и `ui.persistence`. Поля шага, для
которых нет колонок, хранятся в `metadata` под ключом `_chainlit`. Текст элементов (процитированные
фрагменты) лежит в БД и отдаётся фронтенду маршрутом `ELEMENT_CONTENT_ROUTE` приложения.
"""

from __future__ import annotations

import datetime as dt
import uuid
from collections.abc import Callable
from typing import Any, cast

from chainlit.data.base import BaseDataLayer
from chainlit.data.utils import queue_until_user_message
from chainlit.element import Element as ChainlitElement
from chainlit.element import ElementDict
from chainlit.step import StepDict
from chainlit.types import Feedback as ChainlitFeedback
from chainlit.types import FeedbackDict, PageInfo, PaginatedResponse, Pagination, ThreadDict, ThreadFilter
from chainlit.user import PersistedUser, User

from db.models import AppUser, Element, Message
from ui.persistence import ConversationStore, ElementValues, LoadedConversation, MessageValues

ELEMENT_CONTENT_ROUTE = "/elements/{thread_id}/{element_id}"
CHAINLIT_EXTRAS = "_chainlit"
TRACE_ID_KEY = "trace_id"
EXTRA_STEP_FIELDS = (
    "streaming",
    "waitForAnswer",
    "tags",
    "generation",
    "defaultOpen",
    "autoCollapse",
    "command",
    "modes",
)
HIDDEN_INPUT = (None, "false")


def queued[F: Callable[..., Any]](method: F) -> F:
    """Декоратор Chainlit без аннотаций: откладывает запись до первого сообщения пользователя в сессии."""
    decorator = cast(Callable[[F], F], cast(Any, queue_until_user_message)())
    return decorator(method)


def parse_id(value: str) -> uuid.UUID:
    return uuid.UUID(str(value))


def parse_time(value: str | None) -> dt.datetime | None:
    """Метка времени Chainlit (`2026-09-16T10:00:00.123456Z`, наивный UTC) → timestamptz."""
    if not value:
        return None
    try:
        parsed = dt.datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=dt.UTC)


def format_time(value: dt.datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=dt.UTC)
    return value.astimezone(dt.UTC).replace(tzinfo=None).isoformat() + "Z"


def element_content_url(thread_id: str, element_id: str) -> str:
    return ELEMENT_CONTENT_ROUTE.format(thread_id=thread_id, element_id=element_id)


# ---------- перевод типов ----------


def message_values(step: StepDict) -> MessageValues:
    metadata = dict(step.get("metadata") or {})
    extras = {key: step.get(key) for key in EXTRA_STEP_FIELDS if step.get(key) is not None}
    if extras:
        metadata[CHAINLIT_EXTRAS] = extras
    parent = step.get("parentId")
    show_input = step.get("showInput")
    trace_id = metadata.get(TRACE_ID_KEY)
    return MessageValues(
        id=parse_id(step["id"]),
        conversation_id=parse_id(step["threadId"]),
        parent_id=parse_id(parent) if parent else None,
        kind=str(step.get("type") or "undefined"),
        name=str(step.get("name") or ""),
        input_text=step.get("input"),
        output_text=step.get("output"),
        is_error=bool(step.get("isError")),
        show_input=None if show_input is None else str(show_input).lower(),
        language=step.get("language"),
        trace_id=str(trace_id) if trace_id else None,
        metadata=metadata,
        created_at=parse_time(step.get("createdAt")),
        started_at=parse_time(step.get("start")),
        finished_at=parse_time(step.get("end")),
    )


def _show_input(value: str | None) -> bool | str | None:
    if value in ("true", "false"):
        return value == "true"
    return value


def step_dict(message: Message) -> StepDict:
    metadata = dict(message.metadata_json or {})
    extras = dict(metadata.pop(CHAINLIT_EXTRAS, None) or {})
    feedback: FeedbackDict | None = None
    if message.feedback is not None:
        feedback = FeedbackDict(
            forId=str(message.id),
            id=str(message.feedback.id),
            value=cast(Any, message.feedback.value),
            comment=message.feedback.comment,
        )
    step: StepDict = {
        "id": str(message.id),
        "threadId": str(message.conversation_id),
        "parentId": str(message.parent_id) if message.parent_id else None,
        "name": message.name,
        "type": cast(Any, message.kind),
        "input": (message.input_text or "") if message.show_input not in HIDDEN_INPUT else "",
        "output": message.output_text or "",
        "isError": message.is_error,
        "showInput": _show_input(message.show_input),
        "language": message.language,
        "metadata": metadata,
        "createdAt": format_time(message.created_at),
        "start": format_time(message.started_at),
        "end": format_time(message.finished_at),
        "streaming": False,
        "waitForAnswer": extras.get("waitForAnswer"),
        "tags": extras.get("tags"),
        "generation": extras.get("generation"),
        "defaultOpen": extras.get("defaultOpen"),
        "autoCollapse": extras.get("autoCollapse"),
        "command": extras.get("command"),
        "feedback": feedback,
    }
    return step


def element_values(element: ChainlitElement) -> ElementValues | None:
    """Элемент Chainlit → поля таблицы; элемент без сообщения не сохраняется."""
    if not element.for_id:
        return None
    content: str | None
    raw = element.content
    if isinstance(raw, bytes | bytearray):
        mime = element.mime or ""
        content = bytes(raw).decode("utf-8", errors="replace") if mime.startswith("text/") else None
    else:
        content = raw
    return ElementValues(
        id=parse_id(element.id),
        conversation_id=parse_id(element.thread_id),
        message_id=parse_id(element.for_id),
        name=element.name,
        kind=str(element.type),
        mime=element.mime,
        display=element.display,
        content=content,
        url=element.url,
        object_key=element.object_key,
        size=getattr(element, "size", None),
        language=getattr(element, "language", None),
        page=getattr(element, "page", None),
        chainlit_key=element.chainlit_key,
        props=dict(getattr(element, "props", None) or {}),
    )


def element_dict(element: Element) -> ElementDict:
    thread_id = str(element.conversation_id)
    element_id = str(element.id)
    url = element.url
    if not url and element.content is not None:
        url = element_content_url(thread_id, element_id)
    return ElementDict(
        id=element_id,
        threadId=thread_id,
        type=cast(Any, element.kind),
        chainlitKey=element.chainlit_key,
        url=url,
        objectKey=element.object_key,
        name=element.name,
        display=cast(Any, element.display or "side"),
        size=cast(Any, element.size),
        language=element.language,
        page=element.page,
        props=dict(element.props or {}),
        forId=str(element.message_id) if element.message_id else None,
        mime=element.mime,
    )


def persisted_user(user: AppUser) -> PersistedUser:
    return PersistedUser(
        id=str(user.id),
        identifier=user.identifier,
        createdAt=format_time(user.created_at) or "",
        metadata=dict(user.metadata_json or {}),
    )


def thread_dict(loaded: LoadedConversation, *, with_content: bool = True) -> ThreadDict:
    conversation = loaded.conversation
    return ThreadDict(
        id=str(conversation.id),
        createdAt=format_time(conversation.created_at) or "",
        name=conversation.title,
        userId=str(conversation.user_id) if conversation.user_id else None,
        userIdentifier=loaded.user_identifier,
        tags=list(conversation.tags or []),
        metadata=dict(conversation.metadata_json or {}),
        steps=[step_dict(message) for message in loaded.messages] if with_content else [],
        elements=[element_dict(element) for element in loaded.elements] if with_content else [],
    )


# ---------- data layer ----------


class AppDataLayer(BaseDataLayer):
    """История диалогов Chainlit в таблицах проекта (FR-7, FR-9)."""

    def __init__(self, store: ConversationStore) -> None:
        self._store = store

    @property
    def store(self) -> ConversationStore:
        return self._store

    async def get_user(self, identifier: str) -> PersistedUser | None:
        user = await self._store.get_user(identifier)
        return persisted_user(user) if user is not None else None

    async def create_user(self, user: User) -> PersistedUser | None:
        stored = await self._store.upsert_user(user.identifier, dict(user.metadata or {}))
        return persisted_user(stored)

    async def upsert_feedback(self, feedback: ChainlitFeedback) -> str:
        row = await self._store.upsert_feedback(
            message_id=parse_id(feedback.forId),
            value=int(feedback.value),
            comment=feedback.comment,
            feedback_id=parse_id(feedback.id) if feedback.id else None,
        )
        return str(row.id)

    async def delete_feedback(self, feedback_id: str) -> bool:
        await self._store.delete_feedback(parse_id(feedback_id))
        return True

    @queued
    async def create_element(self, element: ChainlitElement) -> None:
        values = element_values(element)
        if values is not None:
            await self._store.upsert_element(values)

    async def get_element(self, thread_id: str, element_id: str) -> ElementDict | None:
        element = await self._store.get_element(parse_id(thread_id), parse_id(element_id))
        return element_dict(element) if element is not None else None

    @queued
    async def delete_element(self, element_id: str, thread_id: str | None = None) -> None:
        await self._store.delete_element(parse_id(element_id))

    @queued
    async def create_step(self, step_dict: StepDict) -> None:
        await self._store.upsert_message(message_values(step_dict))

    @queued
    async def update_step(self, step_dict: StepDict) -> None:
        await self._store.upsert_message(message_values(step_dict))

    @queued
    async def delete_step(self, step_id: str) -> None:
        await self._store.delete_message(parse_id(step_id))

    async def get_thread_author(self, thread_id: str) -> str:
        return await self._store.conversation_author(parse_id(thread_id)) or ""

    async def delete_thread(self, thread_id: str) -> None:
        await self._store.delete_conversation(parse_id(thread_id))

    async def list_threads(
        self, pagination: Pagination, filters: ThreadFilter
    ) -> PaginatedResponse[ThreadDict]:
        if not filters.userId:
            raise ValueError("список диалогов запрашивается для конкретного пользователя")
        page = await self._store.list_conversations(
            parse_id(filters.userId),
            limit=pagination.first,
            cursor=parse_id(pagination.cursor) if pagination.cursor else None,
            search=filters.search,
            feedback=int(filters.feedback) if filters.feedback is not None else None,
        )
        data = [thread_dict(item, with_content=False) for item in page.items]
        return PaginatedResponse(
            pageInfo=PageInfo(
                hasNextPage=page.has_next,
                startCursor=data[0]["id"] if data else None,
                endCursor=data[-1]["id"] if data else None,
            ),
            data=data,
        )

    async def get_thread(self, thread_id: str) -> ThreadDict | None:
        loaded = await self._store.get_conversation(parse_id(thread_id))
        return thread_dict(loaded) if loaded is not None else None

    async def update_thread(
        self,
        thread_id: str,
        name: str | None = None,
        user_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        tags: list[str] | None = None,
    ) -> None:
        await self._store.upsert_conversation(
            parse_id(thread_id),
            title=name,
            user_id=parse_id(user_id) if user_id else None,
            metadata=metadata,
            tags=tags,
        )

    async def build_debug_url(self) -> str:
        return ""

    async def close(self) -> None:
        await self._store.close()

    async def get_favorite_steps(self, user_id: str) -> list[StepDict]:
        return []
