"""Поток вопроса для UI (7.1): шаги агента, стрим, ответ с цитатами; восстановление диалога из БД (7.1, FR-6).

Раннер — тот же, что в тестах агента: MCP-сервер в памяти и LLM по сценарию, Chainlit не нужен.
"""

from __future__ import annotations

from typing import Any

from agent.rendering import TOOL_SEARCH
from common.config import DEFAULT_CONFIG_PATH, load_app_config
from tests.unit.agent_fakes import tool_step
from tests.unit.test_agent_runner import QUERY, Harness
from tests.unit.test_agent_runner import harness as harness_fixture  # noqa: F401 — фикстура pytest «harness»
from ui.citations import SOURCES_NAME, citation_name
from ui.flow import (
    CACHE_KEY,
    NO_SEARCH_TITLE,
    NOTES_KEY,
    QUERIES_HEADER,
    REWRITE_KEY,
    REWRITE_TITLE,
    UNCHANGED_TITLE,
    Final,
    StepFinished,
    StepStarted,
    Token,
    UiEvent,
    run_question,
)
from ui.state import AGENT_METADATA_KEY, TRACE_ID_KEY, restore_session

CONFIG = load_app_config(DEFAULT_CONFIG_PATH)
QUESTION = "Когда сдаётся отчёт по охране труда?"
ANSWER = "Отчёт по охране труда сдаётся до пятого числа [S1]."


def _started(events: list[UiEvent], key: str) -> StepStarted:
    return next(event for event in events if isinstance(event, StepStarted) and event.key == key)


def _finished(events: list[UiEvent], key: str) -> StepFinished:
    return next(event for event in events if isinstance(event, StepFinished) and event.key == key)


def _final(events: list[UiEvent]) -> Final:
    final = events[-1]
    assert isinstance(final, Final)
    return final


def _thread_steps(question: str, final: Final) -> list[dict[str, Any]]:
    """Шаги диалога в том виде, в каком Chainlit отдаёт их при возобновлении."""
    return [
        {"type": "user_message", "output": question, "metadata": {}},
        {"type": "tool", "output": "", "metadata": {}},
        {"type": "assistant_message", "output": final.text, "metadata": final.metadata},
    ]


async def test_steps_stream_and_final_answer_with_citations(harness: Harness) -> None:
    runner = harness.runner(
        [tool_step(TOOL_SEARCH, query=QUERY), "Заметки: D1 (S1) отвечает на вопрос."], [ANSWER]
    )
    session = harness.session()
    events = [event async for event in run_question(runner, session, QUESTION)]

    # первый сигнал — шаг разбора вопроса, до вызовов LLM (AC-7.1)
    first = events[0]
    assert isinstance(first, StepStarted) and first.key == REWRITE_KEY and first.title == REWRITE_TITLE
    assert _finished(events, REWRITE_KEY).title == UNCHANGED_TITLE
    # шаг инструмента: статус при старте, итог после стрелки при завершении (FR-7)
    assert _started(events, "tool-1").title == f"Ищу: «{QUERY}»"
    tool = _finished(events, "tool-1")
    assert tool.ok and tool.title.startswith(f"Ищу: «{QUERY}» → Найдено фрагментов:")
    assert _finished(events, NOTES_KEY).output == "Заметки: D1 (S1) отвечает на вопрос."
    # стрим идёт с маркерами модели, готовый ответ — с номерами ссылок и блоком «Источники»
    assert "".join(event.text for event in events if isinstance(event, Token)) == ANSWER
    final = _final(events)
    assert final.text.startswith(
        "Отчёт по охране труда сдаётся до пятого числа [1].\n\nИсточники:\n[1] Приказ №144"
    )
    assert [citation.name for citation in final.citations] == [citation_name(1), SOURCES_NAME]
    assert "Приказ №144" in final.citations[0].content and "пятого" in final.citations[0].content
    assert final.citations[1].content.startswith("- [1] Приказ №144")
    assert 0 <= final.first_signal_s < 1.0 and final.seconds >= final.first_signal_s
    # metadata сообщения: запись хода со снимком свидетельств и trace_id (без Langfuse — пуст)
    record = final.metadata[AGENT_METADATA_KEY]
    assert final.metadata[TRACE_ID_KEY] is None and record["question"] == QUESTION
    assert record["document_aliases"][0] == "D1" and record["tool_calls"] == 1 and not record["refused"]
    assert [item["alias"] for item in record["evidence"]["documents"]][0] == "D1"
    assert "S1" in [item["alias"] for item in record["evidence"]["fragments"]]
    assert record["first_signal_s"] is not None


async def test_restored_dialogue_answers_follow_up_from_cache(harness: Harness) -> None:
    runner = harness.runner([tool_step(TOOL_SEARCH, query=QUERY), "Заметки: D1 (S1)."], [ANSWER])
    session = harness.session()
    final = _final([event async for event in run_question(runner, session, QUESTION)])
    chunk_id = session.registry.fragment_by_alias("S1")
    assert chunk_id is not None

    restored = restore_session(CONFIG, "thread-1", _thread_steps(QUESTION, final))
    assert restored.id == "thread-1"
    document = restored.registry.document_by_alias("D1")
    fragment = restored.registry.fragment_by_alias("S1")
    assert (
        document is not None
        and document.doc_id == harness.order_id
        and document.label == "Приказ №144 от 15.01.2026"
    )
    assert fragment is not None and fragment.chunk_id == chunk_id.chunk_id and "пятого" in fragment.text
    assert [turn.question for turn in restored.memory.turns] == [QUESTION]
    assert restored.memory.turns[0].answer == final.text
    assert restored.memory.turns[0].document_aliases[0] == "D1"

    follow_up = harness.runner(
        ["Заметки: отвечаю по ранее найденному."],
        ["Да, до пятого числа [S1]."],
        rewrite_steps=[
            '{"query": "срок сдачи отчёта по охране труда по приказу №144", "needs_search": true, '
            '"relevant_documents": ["D1"], "reason": "уточнение по найденному документу"}'
        ],
    )
    events = [event async for event in run_question(follow_up, restored, "а точно до пятого?")]
    assert _started(events, CACHE_KEY).title == "Использую ранее найденное: D1"
    assert "D1 Приказ №144" in (_finished(events, CACHE_KEY).output or "")
    assert not any(isinstance(event, StepStarted) and event.key.startswith("tool-") for event in events)
    final2 = _final(events)
    assert (
        final2.text.startswith("Да, до пятого числа [1].")
        and final2.answer.sources[0].chunk_id == chunk_id.chunk_id
    )
    assert _finished(events, REWRITE_KEY).output == "Пояснение: уточнение по найденному документу"


async def test_rewrite_step_titles_for_chat_mode_and_sub_queries(harness: Harness) -> None:
    chat = harness.runner(
        [],
        ["Здравствуйте! Спрашивайте о документах."],
        rewrite_steps=['{"query": "привет", "intent": "greeting"}'],
    )
    events = [event async for event in run_question(chat, harness.session(), "привет")]
    assert _finished(events, REWRITE_KEY).title == NO_SEARCH_TITLE
    assert not any(isinstance(event, StepStarted) and event.key.startswith("tool-") for event in events)
    assert _final(events).citations == []

    split = harness.runner(
        [tool_step(TOOL_SEARCH, query=QUERY), "Заметки: D1."],
        ["Срок — до пятого числа [S1]."],
        rewrite_steps=[
            '{"query": "срок отчёта и ответственный", "queries": ["срок отчёта по охране труда", '
            '"ответственный за охрану труда"], "needs_search": true}'
        ],
    )
    events = [event async for event in run_question(split, harness.session(), "Когда отчёт и кто отвечает?")]
    rewrite = _finished(events, REWRITE_KEY)
    assert rewrite.title == "Запрос с учётом диалога: «срок отчёта и ответственный»"
    assert (
        rewrite.output
        == f"{QUERIES_HEADER}\n1. срок отчёта по охране труда\n2. ответственный за охрану труда"
    )
