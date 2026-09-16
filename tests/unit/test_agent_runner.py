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

from agent.evidence import EvidenceRegistry
from agent.prompts import BUDGET_CAVEAT, NO_ANSWER_PHRASE
from agent.rendering import TOOL_CARD, TOOL_CONTENT, TOOL_RELATED, TOOL_SEARCH
from agent.runner import AgentRunner, AgentSession, AnswerReady, ToolFinished, ToolStarted
from agent.tools import BUDGET_EXHAUSTED, AgentTools
from common.config import DEFAULT_CONFIG_PATH, LlmRole, load_app_config
from mcp_server.cards import CardServiceClient
from mcp_server.content import DocumentReader
from mcp_server.glossary import EmptyGlossary
from mcp_server.server import Services, build_server
from tessa_export.fake import link, make_snapshot, stable_uuid
from tests.unit.agent_fakes import InMemoryTransport, ScriptedLLM, tool_step
from tests.unit.index_data import InMemoryCorpus, standard_corpus

CONFIG = load_app_config(DEFAULT_CONFIG_PATH)
CARDS_URL = "http://cards.test"
ORDER_109 = stable_uuid("agent", "order109")
QUERY = "отчёт по охране труда до пятого числа"
TODAY = dt.date(2026, 9, 16)


@dataclass
class Harness:
    tools: AgentTools
    corpus: InMemoryCorpus
    order_id: str
    llms: dict[str, ScriptedLLM]

    def runner(self, loop_steps: list[Any], answer_steps: list[str]) -> AgentRunner:
        self.llms = {
            "tool_loop": ScriptedLLM(steps=loop_steps),
            "answer": ScriptedLLM(steps=answer_steps),
            "rewrite": ScriptedLLM(),
            "summary": ScriptedLLM(),
        }

        def factory(role: LlmRole) -> ScriptedLLM:
            return self.llms[role]

        return AgentRunner(CONFIG, self.tools, factory, today=lambda: TODAY)

    def session(self) -> AgentSession:
        return AgentSession(CONFIG)


def _card_server(cards: dict[UUID, dict[str, Any]]) -> Any:
    def serve(request: httpx.Request) -> httpx.Response:
        card_id = UUID(httpx.Response(200, content=request.content).json()["card_id"])
        if card_id not in cards:
            return httpx.Response(404, json={"error": "resource_not_found", "message": "card не найден"})
        return httpx.Response(200, json=cards[card_id])

    return serve


@pytest.fixture
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
        "tool_started",
        "tool_finished",
        "loop_notes",
        "answer_delta",
        "answer_ready",
    ]
    started, finished = events[1], events[2]
    assert isinstance(started, ToolStarted) and started.status == f"Ищу: «{QUERY}»"
    assert (
        isinstance(finished, ToolFinished)
        and finished.ok
        and finished.summary.startswith("Найдено фрагментов:")
    )
    ready = events[-1]
    assert isinstance(ready, AnswerReady)
    answer = ready.answer
    assert answer.text == "Отчёт по охране труда сдаётся до пятого числа [S1]."
    assert not answer.refused and not answer.budget_exhausted
    assert answer.search_queries == [QUERY] and answer.tool_calls[0].name == TOOL_SEARCH
    assert answer.fragment_aliases[0] == "S1" and answer.document_aliases[0] == "D1"

    document = session.registry.document_by_alias("D1")
    assert document is not None and document.doc_id == harness.order_id
    assert document.label == "Приказ №144 от 15.01.2026" and document.doc_status == "active"
    fragment = session.registry.fragment_by_alias("S1")
    assert fragment is not None and "пятого" in fragment.text and fragment.doc_alias == "D1"

    # LLM цикла видит компактный текст с псевдонимами, а не JSON с UUID
    tool_text = _tool_messages(harness.llms["tool_loop"])[0]
    assert (
        tool_text.startswith("Найдено фрагментов:")
        and "[S1] D1 Приказ №144 от 15.01.2026 (действует" in tool_text
    )
    assert harness.order_id not in tool_text
    # LLM ответа получает вопрос, заметки и свидетельства со статусами; в системном промпте — правило отказа
    system, user = harness.llms["answer"].inputs[0]
    assert NO_ANSWER_PHRASE in str(system.content) and "[S1]" in str(system.content)
    assert "Вопрос пользователя: Когда сдаётся отчёт" in str(user.content)
    assert "[S1] (D1, действует)" in str(user.content) and "Заметки агента" in str(user.content)


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
    assert f"связь: [{related.alias}]" in evidence and f"[{section_alias}] (D1, действует)" in evidence


async def test_budget_never_exceeds_max_tool_calls(harness: Harness) -> None:
    limit = CONFIG.agent.max_tool_calls
    runner = harness.runner(
        [tool_step(TOOL_SEARCH, query=f"{QUERY} {index}") for index in range(limit + 2)]
        + ["Заметки: бюджет исчерпан."] * 3,
        ["Отчёт сдаётся до пятого числа [S1]."],
    )
    answer = await runner.ask("Когда сдаётся отчёт?", harness.session())
    assert len(answer.tool_calls) == limit == 8  # AC-1.3
    assert answer.budget_exhausted and answer.text.endswith(BUDGET_CAVEAT)
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
