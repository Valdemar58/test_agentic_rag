"""Раннер агента (6.2): цикл инструментов через MCP в памяти с фейковым LLM, бюджет (AC-1.3), ответ.

Инструменты — настоящий MCP-сервер этапа 5 на встроенном Qdrant и respx-моке сервиса карточек;
LLM — сценарий шагов, чтобы проверять обвязку, а не модель.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any
from uuid import UUID

import httpx
import pytest
import respx
from mcp.types import Tool
from openai import APIStatusError

from agent.evidence import EvidenceRegistry
from agent.memory import Turn
from agent.prompts import BUDGET_CAVEAT, FORCED_SEARCH_NOTE, NO_ANSWER_PHRASE, NO_HITS_NOTES
from agent.rendering import CUT_MARK, TOOL_CARD, TOOL_CONTENT, TOOL_RELATED, TOOL_SEARCH
from agent.runner import (
    AgentRunner,
    AgentSession,
    AnswerReady,
    AnswerRestarted,
    AnswerVerified,
    CacheUsed,
    LoopText,
    QueryRewritten,
    ToolFinished,
    ToolStarted,
)
from agent.tools import (
    BUDGET_EXHAUSTED,
    CONTEXT_EXHAUSTED,
    CONTEXT_TIGHT,
    AgentTools,
    ToolResultData,
    ToolTransport,
    normalize_status,
)
from agent.tracing import Tracing
from common.config import DEFAULT_CONFIG_PATH, AppConfig, LlmRole, load_app_config
from mcp_server.cards import CardServiceClient
from mcp_server.content import DocumentReader
from mcp_server.glossary import EmptyGlossary
from mcp_server.server import Services, build_server
from tessa_export.fake import link, make_snapshot, stable_uuid
from tests.unit.agent_fakes import InMemoryTransport, RecordingTracing, ScriptedLLM, tool_step
from tests.unit.index_data import InMemoryCorpus, standard_corpus

CONFIG = load_app_config(DEFAULT_CONFIG_PATH)
CARDS_URL = "http://cards.test"
ORDER_109 = stable_uuid("agent", "order109")
QUERY = "отчёт по охране труда до пятого числа"
TODAY = dt.date(2026, 9, 16)
VERIFY_OK = '{"problems": []}'


@dataclass
class Harness:
    tools: AgentTools
    corpus: InMemoryCorpus
    order_id: str
    llms: dict[str, ScriptedLLM]

    def runner(
        self,
        loop_steps: list[Any],
        answer_steps: list[str],
        rewrite_steps: list[str] | None = None,
        summary_steps: list[str] | None = None,
        verify_steps: list[str] | None = None,
        config: AppConfig = CONFIG,
        tracing: Tracing | None = None,
    ) -> AgentRunner:
        self.llms = {
            "tool_loop": ScriptedLLM(steps=loop_steps),
            "answer": ScriptedLLM(steps=answer_steps),
            # без сценария переписывание отдаёт не-JSON и поиск идёт по исходному вопросу
            "rewrite": ScriptedLLM(steps=rewrite_steps or []),
            "summary": ScriptedLLM(steps=summary_steps or []),
            # без сценария проверка черновика замечаний не находит
            "verify": ScriptedLLM(steps=verify_steps or [], fallback=VERIFY_OK),
        }

        def factory(role: LlmRole) -> ScriptedLLM:
            return self.llms[role]

        return AgentRunner(config, self.tools, factory, tracing=tracing, today=lambda: TODAY)

    def session(self, config: AppConfig = CONFIG) -> AgentSession:
        return AgentSession(config)


def _card_server(cards: dict[UUID, dict[str, Any]]) -> Any:
    def serve(request: httpx.Request) -> httpx.Response:
        card_id = UUID(httpx.Response(200, content=request.content).json()["card_id"])
        if card_id not in cards:
            return httpx.Response(404, json={"error": "resource_not_found", "message": "card не найден"})
        return httpx.Response(200, json=cards[card_id])

    return serve


@pytest.fixture(name="harness")  # имя явно: фикстуру импортирует и живой тест трейсинга
async def harness() -> AsyncIterator[Harness]:
    corpus = standard_corpus()
    order_id = UUID(corpus.docs["order_144"])
    cards = {
        order_id: make_snapshot(
            order_id,
            number="144",
            outgoing=[link(ORDER_109, ref_type_name="в отмену", ref_type_reverse_name="отменено")],
        ).card_data_json,
        ORDER_109: make_snapshot(
            ORDER_109, number="109", incoming=[link(order_id, ref_type_name=None, ref_type_reverse_name=None)]
        ).card_data_json,
    }
    services = Services(
        searcher=corpus.searcher(),
        cards=CardServiceClient(CARDS_URL, CONFIG.card_service, username="robot", password="x"),
        reader=DocumentReader(corpus.index.client, CONFIG.qdrant, CONFIG.retrieval),
        glossary=EmptyGlossary(),
    )
    tools = AgentTools(InMemoryTransport(build_server(CONFIG, services)))
    with respx.mock(assert_all_called=False) as router:
        router.post(f"{CARDS_URL}/core/cards/get").mock(side_effect=_card_server(cards))
        await tools.load()
        yield Harness(tools=tools, corpus=corpus, order_id=str(order_id), llms={})
    await services.aclose()


def _tool_messages(llm: ScriptedLLM) -> list[str]:
    """Уникальные сообщения инструментов, которые видел LLM (история повторяется на каждом шаге)."""
    seen: dict[str, None] = {}
    for messages in llm.inputs:
        for message in messages:
            if message.role == "tool":
                seen.setdefault(str(message.content), None)
    return list(seen)


async def test_search_then_answer_with_aliases_and_events(harness: Harness) -> None:
    runner = harness.runner(
        [tool_step(TOOL_SEARCH, query=QUERY), "Заметки: D1 (S1) отвечает на вопрос, документ действует."],
        ["Отчёт по охране труда сдаётся до пятого числа [S1]."],
    )
    session = harness.session()
    events = [event async for event in runner.run("Когда сдаётся отчёт по охране труда?", session)]
    assert [event.kind for event in events] == [
        "run_started",
        "query_rewritten",
        "tool_started",
        "tool_finished",
        "loop_notes",
        "answer_delta",
        "verify_started",
        "answer_verified",
        "answer_ready",
    ]
    started, finished = events[2], events[3]
    assert isinstance(started, ToolStarted) and started.status == f"Ищу: «{QUERY}»"
    assert (
        isinstance(finished, ToolFinished)
        and finished.ok
        and finished.summary.startswith("Найдено фрагментов:")
    )
    ready = events[-1]
    assert isinstance(ready, AnswerReady)
    answer = ready.answer
    # FR-4: ссылка модели [S1] стала [1], блок «Источники» построен по реестру и ведёт к чанку индекса
    assert answer.text.startswith(
        "Отчёт по охране труда сдаётся до пятого числа [1].\n\nИсточники:\n[1] Приказ №144"
    )
    assert len(answer.sources) == 1 and answer.sources[0].alias == "S1" and not answer.unresolved_markers
    assert not answer.refused and not answer.budget_exhausted
    assert answer.search_queries == [QUERY] and answer.tool_calls[0].name == TOOL_SEARCH
    assert answer.fragment_aliases[0] == "S1" and answer.document_aliases[0] == "D1"

    document = session.registry.document_by_alias("D1")
    assert document is not None and document.doc_id == harness.order_id
    assert document.label == "Приказ №144 от 15.01.2026" and document.doc_status == "active"
    fragment = session.registry.fragment_by_alias("S1")
    assert fragment is not None and "пятого" in fragment.text and fragment.doc_alias == "D1"
    assert answer.sources[0].chunk_id == fragment.chunk_id and answer.sources[0].doc_id == harness.order_id

    # LLM цикла видит компактный текст с псевдонимами, а не JSON с UUID
    tool_text = _tool_messages(harness.llms["tool_loop"])[0]
    assert (
        tool_text.startswith("Найдено фрагментов:")
        and "[S1] D1 Приказ №144 от 15.01.2026 (действует" in tool_text
    )
    assert harness.order_id not in tool_text
    # LLM ответа получает свидетельства со статусами, ориентир из заметок и вопрос; в системном промпте —
    # правило отказа; проверка черновика без замечаний ответ не меняет
    system, user = harness.llms["answer"].inputs[0]
    assert NO_ANSWER_PHRASE in str(system.content) and "[S1]" in str(system.content)
    assert str(user.content).endswith("Вопрос пользователя: Когда сдаётся отчёт по охране труда?")
    assert "[S1] (D1, действует)" in str(user.content) and "Ориентир шага поиска" in str(user.content)
    assert "D1 (S1) отвечает на вопрос, документ действует." in str(user.content)
    verified = events[-2]
    assert isinstance(verified, AnswerVerified) and verified.parsed and not verified.problems
    assert answer.verification is not None and not answer.verification.corrected
    verify_user = str(harness.llms["verify"].inputs[0][1].content)
    assert "Черновик ответа:\nОтчёт по охране труда сдаётся до пятого числа [S1]." in verify_user
    assert "[S1] (D1, действует)" in verify_user


async def test_aliases_resolve_to_ids_and_evidence_accumulates(harness: Harness) -> None:
    runner = harness.runner(
        [
            tool_step(TOOL_SEARCH, query=QUERY),
            tool_step(TOOL_CONTENT, doc_id="D1", section_id="S1"),
            tool_step(TOOL_CARD, doc_id="D1"),
            tool_step(TOOL_RELATED, doc_id="D1"),
            "Заметки: всё найдено.",
        ],
        ["Ответ [S1][S2]."],
    )
    session = harness.session()
    answer = await runner.ask("Что с отчётом?", session)
    calls = answer.tool_calls
    assert [call.name for call in calls] == [TOOL_SEARCH, TOOL_CONTENT, TOOL_CARD, TOOL_RELATED]
    fragment = session.registry.fragment_by_alias("S1")
    assert fragment is not None
    assert calls[1].arguments == {"doc_id": harness.order_id, "section_id": fragment.chunk_id}
    assert calls[1].summary == "Прочитано разделов: 1 из 1"
    assert calls[2].arguments == {"doc_id": harness.order_id}
    assert calls[2].summary == "Карточка: Приказ №144 от 15.01.2026 (действует)"
    assert calls[3].summary == "Связей: 1 (в отмену)"

    # поиск дал несколько фрагментов (S1…Sn), раздел из get_document_content — следующий псевдоним
    section_alias = answer.fragment_aliases[-1]
    section = session.registry.fragment_by_alias(section_alias)
    assert section is not None and section.kind == "section" and section.doc_alias == "D1"
    assert section.chunk_id == fragment.parent_id, "section_id=S1 (child) отдал раздел-родитель"
    document = session.registry.document_by_alias("D1")
    assert (
        document is not None
        and document.card_text
        and "статус: действует (Действующий)" in document.card_text
    )
    related = session.registry.document(str(ORDER_109))
    assert related is not None and related.alias.startswith("D")
    assert document.relations == [f"[{related.alias}] {related.label} — в отмену (исходящая)"]
    assert answer.document_aliases[0] == "D1" and related.alias in answer.document_aliases
    evidence = str(harness.llms["answer"].inputs[0][1].content)
    assert f"[{section_alias}] (D1, действует)" in evidence
    # сводка карточки (тема, согласующие, связи) — в свидетельствах, факты из неё цитируются как [D1]
    assert "    Тема: О назначении ответственных" in evidence
    assert f"    [{related.alias}] {related.label} — в отмену (исходящая)" in evidence


async def test_budget_never_exceeds_max_tool_calls(harness: Harness) -> None:
    limit = CONFIG.agent.max_tool_calls
    runner = harness.runner(
        [tool_step(TOOL_SEARCH, query=f"{QUERY} {index}") for index in range(limit + 2)]
        + ["Заметки: бюджет исчерпан."] * 3,
        ["Отчёт сдаётся до пятого числа [S1]."],
    )
    answer = await runner.ask("Когда сдаётся отчёт?", harness.session())
    assert len(answer.tool_calls) == limit == 8  # AC-1.3
    assert answer.budget_exhausted and BUDGET_CAVEAT in answer.text
    # LLM просил инструменты и сверх бюджета: вызовы не выполнены, вместо результата — пояснение
    assert len(harness.llms["tool_loop"].inputs) >= limit + 2
    assert BUDGET_EXHAUSTED.format(limit=limit) in _tool_messages(harness.llms["tool_loop"])
    system = str(harness.llms["answer"].inputs[0][0].content)
    assert "Бюджет поиска был исчерпан" in system


async def test_tool_error_is_returned_as_text_and_refusal_is_detected(harness: Harness) -> None:
    missing = "00000000-0000-0000-0000-000000000000"
    runner = harness.runner(
        [tool_step(TOOL_CARD, doc_id=missing), "Заметки: карточка не найдена."],
        [f"{NO_ANSWER_PHRASE}. Карточка запрошенного документа не найдена."],
    )
    events = [event async for event in runner.run("Что в карточке?", harness.session())]
    finished = next(event for event in events if isinstance(event, ToolFinished))
    assert not finished.ok and "не найдена" in finished.summary
    ready = events[-1]
    assert isinstance(ready, AnswerReady) and ready.answer.refused
    assert ready.answer.tool_calls[0].ok is False
    assert _tool_messages(harness.llms["tool_loop"])[0].startswith(f"Ошибка инструмента {TOOL_CARD}")


def test_registry_is_shared_across_questions_of_a_session() -> None:
    session = AgentSession(CONFIG)
    assert isinstance(session.registry, EvidenceRegistry)
    session.registry.register_document("doc-1", label="Приказ №1")
    assert session.registry.document_by_alias("D1") is not None


async def test_rewritten_query_drives_the_loop_and_turns_are_remembered(harness: Harness) -> None:
    rewritten = "срок сдачи отчёта по охране труда для филиалов"
    runner = harness.runner(
        [tool_step(TOOL_SEARCH, query=rewritten), "Заметки: срок тот же."],
        ["Для филиалов срок тот же — до пятого числа [S1]."],
        rewrite_steps=[
            f'{{"query": "{rewritten}", "needs_search": true, "relevant_documents": ["D1"], '
            '"abbreviations": [], "reason": "уточнение к прошлому вопросу"}'
        ],
    )
    session = harness.session()
    session.registry.register_document(
        harness.order_id, label="Приказ №144 от 15.01.2026", doc_status="active"
    )
    session.memory.add(Turn(question="Когда сдаётся отчёт по охране труда?", answer="До пятого числа [S1]."))

    events = [event async for event in runner.run("а для филиалов?", session)]
    rewritten_event = next(event for event in events if isinstance(event, QueryRewritten))
    assert rewritten_event.query == rewritten and rewritten_event.changed and rewritten_event.needs_search
    assert rewritten_event.relevant_documents == ["D1"]
    ready = events[-1]
    assert isinstance(ready, AnswerReady)
    assert ready.answer.rewritten_query == rewritten and ready.answer.search_queries == [rewritten]

    rewrite_input = str(harness.llms["rewrite"].inputs[0][1].content)
    assert "Пользователь: Когда сдаётся отчёт по охране труда?" in rewrite_input
    assert "[D1] Приказ №144 от 15.01.2026 (действует)" in rewrite_input
    assert rewrite_input.endswith("Новый вопрос пользователя: а для филиалов?")
    loop_user = str(harness.llms["tool_loop"].inputs[0][-1].content)
    assert "Вопрос пользователя: а для филиалов?" in loop_user
    assert f"Поисковый запрос с учётом диалога: {rewritten}" in loop_user

    assert [turn.question for turn in session.memory.turns] == [
        "Когда сдаётся отчёт по охране труда?",
        "а для филиалов?",
    ]
    assert session.memory.turns[-1].rewritten_query == rewritten


async def test_message_without_search_is_answered_from_history_without_tools(harness: Harness) -> None:
    runner = harness.runner(
        [tool_step(TOOL_SEARCH, query="не должно вызываться")],
        ["Здравствуйте! Я ищу ответы в приказах и договорах организации. Задайте вопрос."],
        rewrite_steps=['{"query": "привет", "intent": "greeting", "reason": "приветствие"}'],
    )
    events = [event async for event in runner.run("Привет!", harness.session())]
    assert not any(isinstance(event, ToolStarted) for event in events)
    ready = events[-1]
    assert isinstance(ready, AnswerReady)
    answer = ready.answer
    assert answer.needs_search is False and answer.tool_calls == [] and not answer.refused
    assert answer.text.startswith("Здравствуйте!")
    system, user = harness.llms["answer"].inputs[0]
    assert "не требует поиска" in str(system.content)
    assert "Сообщение пользователя: Привет!" in str(user.content)
    assert not harness.llms["tool_loop"].inputs


async def test_follow_up_uses_cached_evidence_without_new_search(harness: Harness) -> None:
    """FR-6: уточнение по уже найденному документу отвечается по кэшу сессии, поиск не повторяется."""
    session = harness.session()
    first = harness.runner(
        [tool_step(TOOL_SEARCH, query=QUERY), "Заметки: нашёл D1."], ["Отчёт сдаётся до пятого числа [S1]."]
    )
    await first.ask("Когда сдаётся отчёт по охране труда?", session)
    cached_fragments = [fragment.alias for fragment in session.registry.fragments_of(harness.order_id)]
    assert cached_fragments

    follow_up = harness.runner(
        ["Заметки: ранее найденных фрагментов достаточно, S1 отвечает на вопрос."],
        ["Да, срок — до пятого числа [S1]."],
        rewrite_steps=[
            '{"query": "срок сдачи отчёта по охране труда по приказу №144", "needs_search": true, '
            '"relevant_documents": ["D1"]}'
        ],
    )
    events = [event async for event in follow_up.run("а точно до пятого?", session)]
    cache = next(event for event in events if isinstance(event, CacheUsed))
    assert cache.document_aliases == ["D1"] and cache.fragment_aliases == cached_fragments
    assert not any(isinstance(event, ToolStarted) for event in events)
    ready = events[-1]
    assert isinstance(ready, AnswerReady)
    assert ready.answer.tool_calls == [] and ready.answer.fragment_aliases == cached_fragments
    loop_user = str(harness.llms["tool_loop"].inputs[0][-1].content)
    assert "Ранее найденные в этом диалоге свидетельства" in loop_user and "[S1]" in loop_user
    assert "пятого" in loop_user
    evidence = str(harness.llms["answer"].inputs[0][1].content)
    assert "[S1] (D1, действует)" in evidence


async def test_old_turns_are_compacted_after_the_answer(harness: Harness) -> None:
    small_memory = CONFIG.agent.memory.model_copy(update={"buffer_messages": 2})
    config = CONFIG.model_copy(update={"agent": CONFIG.agent.model_copy(update={"memory": small_memory})})
    session = harness.session(config)
    session.memory.add(Turn(question="Первый вопрос?", answer="Первый ответ."))
    runner = harness.runner(
        [tool_step(TOOL_SEARCH, query=QUERY), "Заметки."],
        ["Ответ [S1]."],
        summary_steps=["Сводка: обсуждали первый вопрос."],
        config=config,
    )
    await runner.ask("Второй вопрос?", session)
    assert session.memory.summary == "Сводка: обсуждали первый вопрос."
    assert [turn.question for turn in session.memory.turns] == ["Второй вопрос?"]
    summary_input = str(harness.llms["summary"].inputs[0][1].content)
    assert "Пользователь: Первый вопрос?" in summary_input and "Второй вопрос?" not in summary_input

    again = harness.runner([tool_step(TOOL_SEARCH, query=QUERY), "Заметки."], ["Ответ [S1]."], config=config)
    await again.ask("Третий вопрос?", session)
    rewrite_input = str(harness.llms["rewrite"].inputs[0][1].content)
    assert "Сводка предыдущего диалога: Сводка: обсуждали первый вопрос." in rewrite_input
    assert "Пользователь: Второй вопрос?" in rewrite_input


async def test_question_is_traced_as_rewrite_loop_and_answer_steps(harness: Harness) -> None:
    """FR-8 / AC-8.1: трейс вопроса содержит цепочку переписывание → цикл инструментов → ответ."""
    tracing = RecordingTracing()
    runner = harness.runner(
        [tool_step(TOOL_SEARCH, query=QUERY), "Заметки: D1."],
        ["Отчёт сдаётся до пятого числа [S1]."],
        rewrite_steps=['{"query": "срок сдачи отчёта по охране труда", "needs_search": true}'],
        tracing=tracing,
    )
    session = harness.session()
    answer = await runner.ask("Когда сдаётся отчёт?", session)
    assert answer.trace_id == "trace-1"
    root = tracing.questions[0]
    assert root["session_id"] == session.id and root["question"] == "Когда сдаётся отчёт?"
    assert root["output"]["answer"] == answer.text and root["output"]["sources"] == [answer.sources[0].line()]
    assert root["metadata"]["rewritten_query"] == "срок сдачи отчёта по охране труда"
    assert [(step["name"], step["kind"]) for step in tracing.steps] == [
        ("rewrite", "chain"),
        ("tool_loop", "agent"),
        ("answer", "chain"),
        ("verify", "chain"),
    ]
    rewrite, loop, compose, verify = tracing.steps
    assert rewrite["output"]["query"] == "срок сдачи отчёта по охране труда"
    assert "Поисковый запрос с учётом диалога: срок сдачи отчёта" in loop["input"]
    assert loop["output"]["tool_calls"][0]["query"] == QUERY and loop["metadata"]["search_queries"] == [QUERY]
    # шаг ответа хранит черновик с маркерами, шаг проверки — замечания, корень трейса — готовый текст
    assert compose["output"] == "Отчёт сдаётся до пятого числа [S1]."
    assert compose["input"]["notes"] == "Заметки: D1."
    assert verify["input"] == {"draft": "Отчёт сдаётся до пятого числа [S1]."}
    assert verify["output"] == {"problems": [], "answer": None} and verify["metadata"]["corrected"] is False
    assert root["metadata"]["verified"] is True and root["metadata"]["unresolved"] == []


async def test_repeated_calls_and_relation_type_filter_do_not_spend_budget(harness: Harness) -> None:
    """Живой прогон: модель перебирала типы связей и повторяла поиск — бюджет уходил впустую."""
    runner = harness.runner(
        [
            tool_step(TOOL_SEARCH, query=QUERY),
            tool_step(TOOL_RELATED, doc_id="D1", relation_type="в отмену"),
            tool_step(TOOL_RELATED, doc_id="D1", relation_type="дополнение"),
            tool_step(TOOL_SEARCH, query=QUERY),
            "Заметки: всё найдено.",
        ],
        ["Ответ [S1]."],
    )
    answer = await runner.ask("Что с отчётом?", harness.session())
    assert [call.name for call in answer.tool_calls] == [TOOL_SEARCH, TOOL_RELATED]
    assert answer.tool_calls[1].arguments == {"doc_id": harness.order_id}, "фильтр по типу связи снят"
    repeats = [
        text for text in _tool_messages(harness.llms["tool_loop"]) if text.startswith("Этот вызов уже")
    ]
    assert len(repeats) == 2 and "Связи документа [D1]" in repeats[0] and "Найдено фрагментов" in repeats[1]


async def test_context_guard_cuts_results_and_stops_tool_calls(harness: Harness) -> None:
    """Контекст цикла ограничен: результаты сверх остатка обрезаются, дальше инструменты не вызываются."""
    # результат первого поиска на синтетическом корпусе ≈ 800 символов: лимит меньше, чтобы он был обрезан
    limits = CONFIG.agent.tool_output.model_copy(
        update={"loop_context_chars": 500, "context_reserve_chars": 200}
    )
    config = CONFIG.model_copy(update={"agent": CONFIG.agent.model_copy(update={"tool_output": limits})})
    runner = harness.runner(
        [
            tool_step(TOOL_SEARCH, query=QUERY),
            tool_step(TOOL_SEARCH, query="контроль оставляю за собой"),
            tool_step(TOOL_CONTENT, doc_id="D1"),
            "Заметки: контекст кончился.",
        ],
        ["Отчёт сдаётся до пятого числа [S1]."],
        config=config,
    )
    answer = await runner.ask("Когда сдаётся отчёт?", harness.session(config))
    assert [call.name for call in answer.tool_calls] == [TOOL_SEARCH], "второй и третий вызовы не выполнены"
    assert answer.context_exhausted and not answer.budget_exhausted and BUDGET_CAVEAT in answer.text
    texts = _tool_messages(harness.llms["tool_loop"])
    assert texts[0].endswith(CONTEXT_TIGHT)
    assert len(texts[0]) <= 500 + len(CUT_MARK) + 1 + len(CONTEXT_TIGHT)
    assert texts[1] == CONTEXT_EXHAUSTED


async def test_model_error_in_loop_does_not_lose_the_answer(harness: Harness) -> None:
    """vLLM отклонил запрос (например, переполнен контекст) — ответ строится по уже собранному."""
    request = httpx.Request("POST", "http://vllm.test/v1/chat/completions")
    error = APIStatusError(
        "maximum context length exceeded", response=httpx.Response(400, request=request), body=None
    )
    runner = harness.runner(
        [tool_step(TOOL_SEARCH, query=QUERY), error, "не должно понадобиться"],
        ["Отчёт сдаётся до пятого числа [S1]."],
    )
    answer = await runner.ask("Когда сдаётся отчёт?", harness.session())
    assert answer.notes and "прерван ошибкой модели" in answer.notes
    assert answer.context_exhausted and BUDGET_CAVEAT in answer.text
    assert answer.sources and answer.sources[0].alias == "S1", "свидетельства первого поиска сохранены"


def test_without_tracing_answer_has_no_trace_id() -> None:
    session = AgentSession(CONFIG, session_id="dialog-1")
    assert session.id == "dialog-1" and AgentSession(CONFIG).id


async def test_loop_without_search_gets_forced_search_and_second_pass(harness: Harness) -> None:
    """Живой диалог 2026-09-16: цикл закончил без единого поиска и написал «ничего не найдено».

    Раннер ищет сам по переписанному запросу и по вопросу, затем запускает цикл ещё раз с результатами."""
    tracing = RecordingTracing()
    question = "Когда сдаётся отчёт?"
    runner = harness.runner(
        ["Вопрос не содержит ключевых слов документов СЭД.", "Заметки: S1 отвечает на вопрос."],
        ["Отчёт сдаётся до пятого числа [S1]."],
        rewrite_steps=[f'{{"query": "{QUERY}", "intent": "documents"}}'],
        tracing=tracing,
    )
    events = [event async for event in runner.run(question, harness.session())]
    kinds = [event.kind for event in events]
    assert kinds == [
        "run_started",
        "query_rewritten",
        "loop_text",
        "tool_started",
        "tool_finished",
        "tool_started",
        "tool_finished",
        "loop_notes",
        "answer_delta",
        "verify_started",
        "answer_verified",
        "answer_ready",
    ]
    note = events[2]
    assert isinstance(note, LoopText) and note.text == FORCED_SEARCH_NOTE
    started = [event for event in events if isinstance(event, ToolStarted)]
    assert [event.status for event in started] == [f"Ищу: «{QUERY}»", f"Ищу: «{question}»"]
    ready = events[-1]
    assert isinstance(ready, AnswerReady)
    answer = ready.answer
    assert answer.search_queries == [QUERY, question] and all(call.ok for call in answer.tool_calls)
    assert answer.notes == "Заметки: S1 отвечает на вопрос." and not answer.refused
    assert answer.sources and answer.sources[0].alias == "S1"
    # второй проход цикла получил результаты поиска в сообщении, а не в истории вызовов
    second_pass = str(harness.llms["tool_loop"].inputs[1][-1].content)
    assert "поиск выполнен за тебя" in second_pass and "Найдено фрагментов:" in second_pass
    assert "[S1] D1 Приказ №144" in second_pass
    assert [step["name"] for step in tracing.steps] == [
        "rewrite",
        "tool_loop",
        "forced_search",
        "tool_loop",
        "answer",
        "verify",
    ]
    assert tracing.steps[2]["output"]["fragments"][0] == "S1"


async def test_almost_right_arguments_are_normalized_instead_of_failing(harness: Harness) -> None:
    """Живой диалог 2026-09-16: `statuses: ["действует"]` и лишний `top_k` стоили по вызову из бюджета."""
    runner = harness.runner(
        [
            tool_step(TOOL_SEARCH, query=QUERY, filters={"statuses": ["Действует"], "doc_kinds": ["приказ"]}),
            tool_step(TOOL_CONTENT, doc_id="D1", section_id="S1", top_k=5),
            "Заметки: всё найдено.",
        ],
        ["Ответ [S1]."],
    )
    answer = await runner.ask("Когда сдаётся отчёт?", harness.session())
    search, content = answer.tool_calls
    assert search.ok and search.arguments["filters"] == {"statuses": ["active"], "doc_kinds": ["приказ"]}
    assert content.ok and "top_k" not in content.arguments and content.summary == "Прочитано разделов: 1 из 1"
    assert normalize_status("отменённый") == "cancelled" and normalize_status("проект") == "draft"
    assert normalize_status("active") == "active" and normalize_status("архив") == "архив"


class _EmptySearch:
    """Транспорт, у которого поиск ничего не находит: гибридный поиск без порога всегда отдаёт top_k."""

    def __init__(self, inner: ToolTransport) -> None:
        self._inner = inner

    async def list_tools(self) -> list[Tool]:
        return await self._inner.list_tools()

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> ToolResultData:
        if name == TOOL_SEARCH:
            return ToolResultData(is_error=False, text="", structured={"hits": [], "candidates": 0})
        return await self._inner.call_tool(name, arguments)

    async def aclose(self) -> None:
        await self._inner.aclose()


async def test_forced_search_without_hits_leads_to_honest_refusal(harness: Harness) -> None:
    harness.tools = AgentTools(_EmptySearch(harness.tools.transport))
    await harness.tools.load()
    runner = harness.runner(
        ["Заметки: не про документы."],
        [f"{NO_ANSWER_PHRASE}. Искали правила парковки велосипедов."],
    )
    answer = await runner.ask("Где парковать велосипед?", harness.session())
    assert [call.name for call in answer.tool_calls] == [TOOL_SEARCH], "запрос совпал с вопросом: один поиск"
    assert answer.tool_calls[0].ok and answer.fragment_aliases == []
    assert answer.notes == NO_HITS_NOTES and answer.refused
    assert len(harness.llms["tool_loop"].inputs) == 1, "без находок второй проход цикла не нужен"


LUNCH_NOTES = (
    "— перерыв для отдыха и питания 45 минут в диапазоне с 12:00 до 15:00 [S1]\n"
    "Таким образом, вернуться с обеда нужно не позднее 13:00.\n"
    "Отчёт сдаётся до пятого числа [S1]. Продолжительность перерыва не включается в рабочее время."
)
LUNCH_DRAFT = "Вернуться нужно не позднее 13:00 [S1]. Перерыв длится 45 минут [S1]."
LUNCH_FIXED = "Перерыв длится 45 минут [S1]."
LUNCH_CLAIM = "Вернуться нужно не позднее 13:00 [S1]."
LUNCH_VERDICT = (
    f'{{"problems": [{{"claim": "{LUNCH_CLAIM}", "reason": "во фрагменте нет времени возвращения"}}]}}'
)


async def test_unsupported_claim_is_corrected_by_the_verifier(harness: Harness) -> None:
    """Живой диалог 2026-09-17: вывод заметок «не позднее 13:00» дошёл до ответа как факт документа.

    В промпт ответа из заметок попадают только предложения с псевдонимами и без вводных слов вывода,
    а черновик ответа проверяется по свидетельствам: неподтверждённое предложение вычёркивается."""
    tracing = RecordingTracing()
    runner = harness.runner(
        [tool_step(TOOL_SEARCH, query=QUERY), LUNCH_NOTES],
        [LUNCH_DRAFT],
        verify_steps=[LUNCH_VERDICT],
        tracing=tracing,
    )
    events = [event async for event in runner.run("Когда вернуться с обеда?", harness.session())]
    ready = events[-1]
    assert isinstance(ready, AnswerReady)
    answer = ready.answer
    assert answer.text.startswith(f"{LUNCH_FIXED.replace('[S1]', '[1]')}\n\nИсточники:\n[1] Приказ №144")
    assert answer.verification is not None and answer.verification.corrected
    assert [(p.claim, p.action) for p in answer.verification.problems] == [(LUNCH_CLAIM, "removed")]
    assert answer.verify_seconds >= 0 and answer.notes == LUNCH_NOTES, "заметки целиком остаются в ответе"
    verified = next(event for event in events if isinstance(event, AnswerVerified))
    assert verified.corrected and verified.parsed and len(verified.problems) == 1
    # ориентир для ответа: строка с выводом и предложение без псевдонима отброшены
    answer_user = str(harness.llms["answer"].inputs[0][1].content)
    assert "45 минут в диапазоне с 12:00 до 15:00 [S1]" in answer_user
    assert "Таким образом" not in answer_user and "не включается в рабочее время" not in answer_user
    assert "Отчёт сдаётся до пятого числа [S1]." in answer_user
    # проверяющий видел черновик, свидетельства и вопрос; в трейсе — черновик, замечания и замена
    verify_user = str(harness.llms["verify"].inputs[0][1].content)
    assert verify_user.endswith(f"Черновик ответа:\n{LUNCH_DRAFT}") and "[S1] (D1, действует)" in verify_user
    verify = next(step for step in tracing.steps if step["name"] == "verify")
    assert verify["input"] == {"draft": LUNCH_DRAFT} and verify["output"]["answer"] == LUNCH_FIXED
    assert verify["output"]["problems"][0]["claim"] == LUNCH_CLAIM
    assert tracing.questions[0]["metadata"]["corrected"] is True
    assert answer.text == tracing.questions[0]["output"]["answer"]


async def test_verifier_failures_keep_the_draft_and_refusals_are_not_verified(harness: Harness) -> None:
    draft = "Отчёт сдаётся до пятого числа [S1]."
    loop = [tool_step(TOOL_SEARCH, query=QUERY), "Заметки [S1]."]
    garbage = harness.runner(loop, [draft], verify_steps=["не JSON"])
    answer = await garbage.ask("Когда отчёт?", harness.session())
    assert answer.text.startswith("Отчёт сдаётся до пятого числа [1].")
    assert answer.verification is not None and not answer.verification.parsed
    assert not answer.verification.corrected and answer.verification.problems == []

    # замечание к единственному предложению: вычёркивание опустошило бы ответ — после повтора остаётся
    # черновик, замечания сохраняются
    whole = f'{{"problems": [{{"claim": "{draft}", "reason": "нет срока"}}]}}'
    kept = harness.runner(loop, [draft, draft], verify_steps=[whole, whole])
    answer = await kept.ask("Когда отчёт?", harness.session())
    assert answer.text.startswith("Отчёт сдаётся до пятого числа [1].")
    assert answer.verification is not None and answer.verification.parsed
    assert not answer.verification.corrected and answer.verification.emptied
    assert [(p.claim, p.action) for p in answer.verification.problems] == [(draft, "kept")]

    # отказ не проверяется, проверяющий не вызывается
    refusal = harness.runner(loop, [f"{NO_ANSWER_PHRASE}. Срока нет."])
    answer = await refusal.ask("Когда отчёт?", harness.session())
    assert answer.refused and answer.verification is None and not harness.llms["verify"].inputs

    # проверка выключена конфигом — черновик уходит как есть, событий проверки нет
    no_verify = CONFIG.agent.verify.model_copy(update={"enabled": False})
    disabled = CONFIG.model_copy(update={"agent": CONFIG.agent.model_copy(update={"verify": no_verify})})
    plain = harness.runner(loop, [draft], verify_steps=[LUNCH_VERDICT], config=disabled)
    events = [event async for event in plain.run("Когда отчёт?", harness.session(disabled))]
    assert not any(isinstance(event, AnswerVerified) for event in events)
    ready = events[-1]
    assert isinstance(ready, AnswerReady) and ready.answer.verification is None
    assert not harness.llms["verify"].inputs


async def test_empty_answer_is_retried_once_before_giving_up(harness: Harness) -> None:
    """Живой прогон 2026-09-17: размышления съели весь лимит токенов, текст ответа пуст (55 с впустую)."""
    tracing = RecordingTracing()
    loop = [tool_step(TOOL_SEARCH, query=QUERY), "Заметки [S1]."]
    runner = harness.runner(loop, ["", "Отчёт сдаётся до пятого числа [S1]."], tracing=tracing)
    answer = await runner.ask("Когда отчёт?", harness.session())
    assert answer.text.startswith("Отчёт сдаётся до пятого числа [1].")
    assert len(harness.llms["answer"].inputs) == 2, "шаг ответа повторён один раз"
    compose = next(step for step in tracing.steps if step["name"] == "answer")
    assert compose["metadata"]["attempts"] == 2

    hopeless = harness.runner(loop, ["", "   ", "не должно вызываться [S1]."])
    answer = await hopeless.ask("Когда отчёт?", harness.session())
    assert answer.text == "" and len(harness.llms["answer"].inputs) == 2 and answer.verification is None


async def test_fully_rejected_draft_is_recomposed_once_with_the_remarks(harness: Harness) -> None:
    """Живой прогон 2026-09-17: весь черновик («вернуться в 12:45» от начала окна) не подтверждён —
    вычёркивать нечего, ответ составляется заново с замечаниями проверки, второй черновик тоже проверяется."""
    tracing = RecordingTracing()
    wrong = "Вернуться нужно в 12:45 [S1]."
    right = "Отчёт сдаётся до пятого числа [S1]."
    reject_all = f'{{"problems": [{{"claim": "{wrong}", "reason": "во фрагменте нет 12:45"}}]}}'
    runner = harness.runner(
        [tool_step(TOOL_SEARCH, query=QUERY), "Заметки [S1]."],
        [wrong, right],
        verify_steps=[reject_all, VERIFY_OK],
        tracing=tracing,
    )
    events = [event async for event in runner.run("Когда вернуться, если ушёл в 12:45?", harness.session())]
    assert [event.kind for event in events][-8:] == [
        "answer_delta",
        "verify_started",
        "answer_verified",
        "answer_restarted",
        "answer_delta",
        "verify_started",
        "answer_verified",
        "answer_ready",
    ]
    ready = events[-1]
    assert isinstance(ready, AnswerReady)
    answer = ready.answer
    assert answer.text.startswith("Отчёт сдаётся до пятого числа [1].")
    assert answer.verification is not None and not answer.verification.emptied
    assert answer.verification.problems == []
    restarted = next(event for event in events if isinstance(event, AnswerRestarted))
    assert [problem.action for problem in restarted.problems] == ["kept"]
    verified = [event for event in events if isinstance(event, AnswerVerified)]
    assert [event.attempt for event in verified] == [1, 2]
    second_prompt = str(harness.llms["answer"].inputs[1][1].content)
    assert "проверка отклонила целиком" in second_prompt and wrong in second_prompt
    assert "во фрагменте нет 12:45" in second_prompt and NO_ANSWER_PHRASE in second_prompt
    names = [step["name"] for step in tracing.steps]
    assert names == ["rewrite", "tool_loop", "answer", "verify", "answer", "verify"]
    assert tracing.steps[2]["input"]["feedback"] == []
    assert tracing.steps[4]["input"]["feedback"] == [(wrong, "во фрагменте нет 12:45")]

    # повтор один: второй отклонённый черновик уходит как есть, с замечаниями
    stubborn = harness.runner(
        [tool_step(TOOL_SEARCH, query=QUERY), "Заметки [S1]."],
        [wrong, wrong],
        verify_steps=[reject_all, reject_all],
    )
    answer = await stubborn.ask("Когда вернуться, если ушёл в 12:45?", harness.session())
    assert answer.text.startswith("Вернуться нужно в 12:45 [1].")
    assert len(harness.llms["answer"].inputs) == 2
    assert answer.verification is not None and answer.verification.emptied


async def test_model_links_block_is_stripped_before_verification(harness: Harness) -> None:
    """Живой прогон 2026-09-17: после вычёркивания всего черновика оставалась строка «Ссылки: [S6]»,
    она сходила за текст со ссылкой, и повтор ответа не запускался."""
    wrong = "Вернуться нужно в 12:45 [S1]."
    reject_all = f'{{"problems": [{{"claim": "{wrong}", "reason": "во фрагменте нет 12:45"}}]}}'
    runner = harness.runner(
        [tool_step(TOOL_SEARCH, query=QUERY), "Заметки [S1]."],
        [f"{wrong}\n\nСсылки: [S1]", "Отчёт сдаётся до пятого числа [S1].\n\n**Ссылки:** [S1]"],
        verify_steps=[reject_all, VERIFY_OK],
    )
    answer = await runner.ask("Когда вернуться, если ушёл в 12:45?", harness.session())
    assert answer.text.startswith("Отчёт сдаётся до пятого числа [1].\n\nИсточники:")
    assert len(harness.llms["answer"].inputs) == 2, "весь черновик отклонён — ответ составлен заново"
    for messages in harness.llms["verify"].inputs:
        assert "Ссылки" not in str(messages[1].content)


async def test_planned_search_runs_each_query_of_a_multipart_question(harness: Harness) -> None:
    """Прогон голден-сета 2026-09-18: модель делала 2–3 вызова из восьми и отвечала по первой находке."""
    rewrite = (
        '{"query": "отчёт и инструкции", "queries": ["отчёт по охране труда", "инструкции по охране труда"],'
        ' "intent": "documents"}'
    )
    runner = harness.runner(
        ["Заметки: обе части вопроса закрыты [S1]."],
        ["Ответ по обеим частям [S1]."],
        rewrite_steps=[rewrite],
        tracing=(tracing := RecordingTracing()),
    )
    events = [event async for event in runner.run("Отчёт и инструкции по охране труда?", harness.session())]

    searches = [
        event.arguments["query"]
        for event in events
        if isinstance(event, ToolStarted) and event.tool == TOOL_SEARCH
    ]
    assert searches[:2] == ["отчёт по охране труда", "инструкции по охране труда"], "оба запроса выполнены"
    assert [item["name"] for item in tracing.steps].count("planned_search") == 1
    loop_message = str(harness.llms["tool_loop"].inputs[0][-1].content)
    assert "поиск по каждой части уже выполнен" in loop_message
    ready = events[-1]
    assert isinstance(ready, AnswerReady) and not ready.answer.refused


async def test_weak_evidence_warning_goes_to_the_answer_prompt(harness: Harness) -> None:
    """Прогон голден-сета 2026-09-18: на вопросах без ответа в корпусе лучшая оценка поиска ≤ 0,10."""
    runner = harness.runner(
        [tool_step(TOOL_SEARCH, query="парковка велосипедов"), "Заметки: ничего по теме [S1]."],
        [f"{NO_ANSWER_PHRASE} про велосипеды в документах нет."],
    )
    events = [event async for event in runner.run("Компенсация за велосипед?", harness.session())]

    answer_prompt = str(harness.llms["answer"].inputs[0][-1].content)
    assert "Оценки релевантности найденных фрагментов низкие" in answer_prompt
    ready = events[-1]
    assert isinstance(ready, AnswerReady) and ready.answer.refused
