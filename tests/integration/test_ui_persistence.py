"""Хранилище диалогов и data layer Chainlit (7.1, 7.3, FR-7, FR-9) на свежем PostgreSQL в testcontainers.

Без Docker тесты скипаются (фикстура `migrated_database_url`).
"""

from __future__ import annotations

import datetime as dt
import uuid
from collections.abc import AsyncIterator
from typing import Any

import pytest
from chainlit.context import init_http_context
from chainlit.element import Text
from chainlit.step import StepDict
from chainlit.types import Feedback, Pagination, ThreadFilter
from chainlit.user import User

from db.session import build_engine
from ui.data_layer import AppDataLayer, element_content_url, format_time, parse_time
from ui.persistence import ConversationStore, ElementValues, MessageValues

pytestmark = pytest.mark.integration


@pytest.fixture
async def store(migrated_database_url: str) -> AsyncIterator[ConversationStore]:
    engine = build_engine(migrated_database_url)
    yield ConversationStore(engine)
    await engine.dispose()


def _message(
    conversation_id: uuid.UUID,
    kind: str,
    text: str,
    *,
    created_at: dt.datetime | None = None,
    trace_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> MessageValues:
    return MessageValues(
        id=uuid.uuid4(),
        conversation_id=conversation_id,
        kind=kind,
        output_text=text,
        created_at=created_at,
        trace_id=trace_id,
        metadata=metadata or {},
    )


async def test_store_round_trip_users_threads_messages_elements_feedback(store: ConversationStore) -> None:
    user = await store.upsert_user("tester", {"provider": "credentials"})
    same = await store.upsert_user("tester", {"provider": "oidc"})
    assert same.id == user.id and same.metadata_json == {"provider": "oidc"}
    assert await store.user_identifier(user.id) == "tester"

    thread_id = uuid.uuid4()
    conversation = await store.upsert_conversation(thread_id, title="Первый вопрос", user_id=user.id)
    assert conversation.title == "Первый вопрос" and await store.conversation_author(thread_id) == "tester"
    await store.upsert_conversation(thread_id, metadata={"a": 1, "b": 2})
    await store.upsert_conversation(thread_id, metadata={"a": None, "c": 3})
    loaded = await store.get_conversation(thread_id)
    assert loaded is not None and loaded.conversation.metadata_json == {"b": 2, "c": 3}
    assert loaded.user_identifier == "tester"

    created = dt.datetime(2026, 9, 16, 10, 0, tzinfo=dt.UTC)
    question = _message(thread_id, "user_message", "Когда сдаётся отчёт?", created_at=created)
    answer = _message(
        thread_id,
        "assistant_message",
        "До пятого числа [1].",
        created_at=created + dt.timedelta(seconds=5),
        trace_id="trace-1",
        metadata={"trace_id": "trace-1", "agent": {"question": "Когда сдаётся отчёт?"}},
    )
    await store.upsert_message(question)
    await store.upsert_message(answer)
    # повторная запись того же шага обновляет поля, а не дублирует строку
    await store.upsert_message(
        MessageValues(
            id=answer.id, conversation_id=thread_id, kind="assistant_message", output_text="До пятого [1]."
        )
    )
    await store.upsert_element(
        ElementValues(
            id=uuid.uuid4(),
            conversation_id=thread_id,
            message_id=answer.id,
            name="[1]",
            kind="text",
            display="side",
            content="**Приказ №144** — действует\n\nДо пятого числа.",
        )
    )
    loaded = await store.get_conversation(thread_id)
    assert loaded is not None
    assert [message.kind for message in loaded.messages] == ["user_message", "assistant_message"]
    stored_answer = loaded.messages[1]
    assert stored_answer.output_text == "До пятого [1]." and stored_answer.trace_id is None
    assert stored_answer.created_at == created + dt.timedelta(seconds=5)
    assert [element.name for element in loaded.elements] == ["[1]"]
    assert await store.message_trace_id(answer.id) is None

    await store.upsert_message(answer)  # trace_id вернулся вместе с metadata
    feedback = await store.upsert_feedback(message_id=answer.id, value=1, comment="полезно")
    assert feedback.trace_id == "trace-1" and feedback.conversation_id == thread_id
    updated = await store.upsert_feedback(
        message_id=answer.id, value=0, comment=None, feedback_id=uuid.uuid4()
    )
    assert updated.id == feedback.id and updated.value == 0
    loaded = await store.get_conversation(thread_id)
    assert loaded is not None and loaded.messages[1].feedback is not None
    assert loaded.messages[1].feedback.value == 0
    with pytest.raises(LookupError):
        await store.upsert_feedback(message_id=uuid.uuid4(), value=1, comment=None)

    await store.delete_conversation(thread_id)
    assert await store.get_conversation(thread_id) is None
    assert await store.get_element(thread_id, loaded.elements[0].id) is None
    assert await store.message_trace_id(answer.id) is None


async def test_list_conversations_pages_filters_and_orders_by_activity(store: ConversationStore) -> None:
    user = await store.upsert_user("lister", {})
    other = await store.upsert_user("other", {})
    ids = [uuid.uuid4() for _ in range(3)]
    for index, thread_id in enumerate(ids):
        await store.upsert_conversation(thread_id, title=f"Диалог {index}", user_id=user.id)
        await store.upsert_message(_message(thread_id, "assistant_message", f"ответ {index} про отчёт"))
    foreign = uuid.uuid4()
    await store.upsert_conversation(foreign, title="Чужой", user_id=other.id)
    # новое сообщение в первом диалоге делает его самым свежим
    await store.upsert_message(_message(ids[0], "assistant_message", "уникальный текст"))
    first_page = await store.list_conversations(user.id, limit=2)
    assert [item.conversation.title for item in first_page.items] == [
        "Диалог 0",
        "Диалог 2",
    ] and first_page.has_next
    second_page = await store.list_conversations(
        user.id, limit=2, cursor=first_page.items[-1].conversation.id
    )
    assert [item.conversation.title for item in second_page.items] == [
        "Диалог 1"
    ] and not second_page.has_next

    found = await store.list_conversations(user.id, limit=10, search="УНИКАЛЬНЫЙ")
    assert [item.conversation.title for item in found.items] == ["Диалог 0"]
    loaded = await store.get_conversation(ids[1])
    assert loaded is not None
    await store.upsert_feedback(message_id=loaded.messages[0].id, value=1, comment=None)
    positive = await store.list_conversations(user.id, limit=10, feedback=1)
    assert [item.conversation.title for item in positive.items] == ["Диалог 1"]
    assert (await store.list_conversations(user.id, limit=10, feedback=0)).items == []


async def test_chainlit_data_layer_round_trip(store: ConversationStore) -> None:
    init_http_context()  # вне websocket-сессии отложенные методы data layer выполняются сразу
    layer = AppDataLayer(store)
    persisted = await layer.create_user(User(identifier="ui-user", metadata={"provider": "credentials"}))
    assert persisted is not None and (await layer.get_user("ui-user")) is not None
    assert persisted.createdAt.endswith("Z") and parse_time(persisted.createdAt) is not None

    thread_id = str(uuid.uuid4())
    await layer.update_thread(thread_id, name="Когда сдаётся отчёт?", user_id=persisted.id)
    user_step: StepDict = {
        "id": str(uuid.uuid4()),
        "threadId": thread_id,
        "type": "user_message",
        "name": "ui-user",
        "output": "Когда сдаётся отчёт?",
        "createdAt": "2026-09-16T10:00:00.000000Z",
        "metadata": {},
    }
    tool_step: StepDict = {
        "id": str(uuid.uuid4()),
        "threadId": thread_id,
        "type": "tool",
        "name": "Ищу: «отчёт» → Найдено фрагментов: 3",
        "input": "",
        "output": "",
        "showInput": "json",
        "start": "2026-09-16T10:00:01.000000Z",
        "end": "2026-09-16T10:00:03.000000Z",
        "createdAt": "2026-09-16T10:00:01.000000Z",
        "metadata": {},
        "defaultOpen": False,
    }
    answer_id = str(uuid.uuid4())
    answer_step: StepDict = {
        "id": answer_id,
        "threadId": thread_id,
        "type": "assistant_message",
        "name": "Поиск по документам СЭД",
        "output": "До пятого числа [1].",
        "createdAt": "2026-09-16T10:00:05.000000Z",
        "metadata": {"trace_id": "trace-42", "agent": {"question": "Когда сдаётся отчёт?"}},
        "streaming": False,
        "tags": ["answer"],
    }
    for step in (user_step, tool_step, answer_step):
        await layer.create_step(step)
    await layer.update_step({**answer_step, "output": "До пятого числа [1]. Источники: [1] Приказ"})
    element = Text(
        name="[1]",
        content="**Приказ №144** — действует",
        display="side",
        for_id=answer_id,
        thread_id=thread_id,
    )
    await layer.create_element(element)
    feedback_id = await layer.upsert_feedback(
        Feedback(forId=answer_id, value=1, threadId=thread_id, comment="ок")
    )
    assert uuid.UUID(feedback_id)

    thread = await layer.get_thread(thread_id)
    assert (
        thread is not None
        and thread["name"] == "Когда сдаётся отчёт?"
        and thread["userIdentifier"] == "ui-user"
    )
    assert thread["userId"] == persisted.id
    assert [step["type"] for step in thread["steps"]] == ["user_message", "tool", "assistant_message"]
    stored_tool = thread["steps"][1]
    assert (
        stored_tool["name"] == "Ищу: «отчёт» → Найдено фрагментов: 3" and stored_tool["defaultOpen"] is False
    )
    assert stored_tool["start"] == "2026-09-16T10:00:01Z" and stored_tool["end"] == "2026-09-16T10:00:03Z"
    stored_answer = thread["steps"][2]
    assert stored_answer["output"].startswith("До пятого числа [1]. Источники") and stored_answer["tags"] == [
        "answer"
    ]
    assert stored_answer["metadata"] == answer_step["metadata"]
    assert stored_answer["feedback"] == {"forId": answer_id, "id": feedback_id, "value": 1, "comment": "ок"}
    assert await store.message_trace_id(uuid.UUID(answer_id)) == "trace-42"
    elements = thread["elements"] or []
    assert len(elements) == 1 and elements[0]["name"] == "[1]" and elements[0]["forId"] == answer_id
    assert elements[0]["url"] == element_content_url(thread_id, element.id)
    fetched = await layer.get_element(thread_id, element.id)
    assert fetched is not None and fetched["type"] == "text" and fetched["display"] == "side"
    stored_element = await store.get_element(uuid.UUID(thread_id), uuid.UUID(element.id))
    assert stored_element is not None and stored_element.content == "**Приказ №144** — действует"

    assert await layer.get_thread_author(thread_id) == "ui-user"
    listed = await layer.list_threads(Pagination(first=10), ThreadFilter(userId=persisted.id))
    assert [item["name"] for item in listed.data] == [
        "Когда сдаётся отчёт?"
    ] and not listed.pageInfo.hasNextPage
    assert listed.data[0]["steps"] == [] and format_time(parse_time(listed.data[0]["createdAt"])) is not None

    assert await layer.delete_feedback(feedback_id)
    await layer.delete_element(element.id, thread_id)
    await layer.delete_step(tool_step["id"])
    thread = await layer.get_thread(thread_id)
    assert thread is not None and [step["type"] for step in thread["steps"]] == [
        "user_message",
        "assistant_message",
    ]
    assert thread["steps"][1]["feedback"] is None and thread["elements"] == []
    await layer.delete_thread(thread_id)
    assert await layer.get_thread(thread_id) is None
    assert await layer.get_favorite_steps(persisted.id) == [] and await layer.build_debug_url() == ""


async def test_feedback_on_run_step_takes_trace_id_of_the_assistant_answer(store: ConversationStore) -> None:
    """Chainlit привязывает 👍/👎 к run-шагу хода; trace_id хранится в ответе ассистента внутри него (7.3)."""
    user = await store.upsert_user("rater", {})
    thread_id = uuid.uuid4()
    await store.upsert_conversation(thread_id, title="Ход с оценкой", user_id=user.id)
    question = _message(thread_id, "user_message", "Когда сдаётся отчёт?")
    await store.upsert_message(question)
    run = MessageValues(
        id=uuid.uuid4(), conversation_id=thread_id, kind="run", name="on_message", parent_id=question.id
    )
    # шаг потомка пишется раньше родителя — без внешнего ключа на parent_id это допустимо (миграция 0003)
    answer = _message(thread_id, "assistant_message", "До пятого [1].", trace_id="trace-run")
    answer = MessageValues(**{**answer.__dict__, "parent_id": run.id})
    await store.upsert_message(answer)
    await store.upsert_message(run)
    await store.upsert_element(
        ElementValues(
            id=uuid.uuid4(), conversation_id=thread_id, message_id=answer.id, name="[1]", kind="text"
        )
    )
    assert await store.message_trace_id(run.id) == "trace-run"
    feedback = await store.upsert_feedback(message_id=run.id, value=0, comment="мимо")
    assert feedback.trace_id == "trace-run" and feedback.message_id == run.id
    loaded = await store.get_conversation(thread_id)
    assert loaded is not None and [message.kind for message in loaded.messages] == [
        "user_message",
        "assistant_message",
        "run",
    ]
    # удаление сообщения убирает и его элементы (внешнего ключа больше нет)
    await store.delete_message(answer.id)
    loaded = await store.get_conversation(thread_id)
    assert loaded is not None and loaded.elements == [] and await store.message_trace_id(run.id) is None
