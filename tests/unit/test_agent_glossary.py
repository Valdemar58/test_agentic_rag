"""Расшифровка аббревиатур по глоссарию до поиска (8.2, FR-5): термины, запрос, шаг раннера."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

import pytest

from agent.glossary import (
    Expansion,
    candidate_terms,
    expand_query,
    first_expansion,
    keep_abbreviations,
    short_definition,
)
from agent.prompts import LOOP_GLOSSARY_TITLE
from agent.rendering import TOOL_SEARCH
from agent.runner import AgentRunner, AgentSession, GlossaryUsed, QueryRewritten
from agent.tools import AgentTools
from common.config import DEFAULT_CONFIG_PATH, load_app_config
from mcp_server.cards import CardServiceClient
from mcp_server.content import DocumentReader
from mcp_server.glossary import GlossaryEntry, GlossaryResult
from mcp_server.server import Services, build_server
from tests.unit.agent_fakes import InMemoryTransport, RecordingTracing, ScriptedLLM, tool_step
from tests.unit.index_data import standard_corpus
from ui.flow import GLOSSARY_KEY, StepFinished, StepStarted, run_question

CONFIG = load_app_config(DEFAULT_CONFIG_PATH)
PVTR = "правила внутреннего трудового распорядка"


class FakeGlossary:
    """Глоссарий с одной записью: проверяем обвязку, а не поиск по коллекции."""

    def __init__(self) -> None:
        self.terms: list[str] = []

    def lookup(self, term: str) -> GlossaryResult:
        self.terms.append(term)
        if term.casefold() != "пвтр":
            return GlossaryResult(term=term, entries=[], note="нет такого термина")
        return GlossaryResult(
            term=term,
            entries=[
                GlossaryEntry(
                    term="ПВТР",
                    definition=f"{PVTR} Общества.",
                    doc_id="d1",
                    doc_label="Приказ №144 от 15.01.2026",
                    doc_status="active",
                    section_id="s1",
                )
            ],
        )


@pytest.fixture(name="parts")
async def parts() -> AsyncIterator[tuple[AgentTools, FakeGlossary]]:
    corpus = standard_corpus()
    glossary = FakeGlossary()
    services = Services(
        searcher=corpus.searcher(),
        cards=CardServiceClient("http://cards.test", CONFIG.card_service, username="robot", password="x"),
        reader=DocumentReader(corpus.index.client, CONFIG.qdrant, CONFIG.retrieval),
        glossary=glossary,
    )
    tools = AgentTools(InMemoryTransport(build_server(CONFIG, services)))
    await tools.load()
    yield tools, glossary
    await services.aclose()


def test_terms_and_query_expansion() -> None:
    assert candidate_terms("Что такое ПВТР и СИЗ?", [], 3) == ["ПВТР", "СИЗ"]
    assert candidate_terms("во сколько обед", ["ПВТР"], 3) == ["ПВТР"]
    # названные моделью идут первыми, повторы и лишнее сверх лимита отбрасываются
    assert candidate_terms("Нужен ли СИЗ по ПВТР?", ["пвтр"], 2) == ["пвтр", "СИЗ"]
    assert candidate_terms("Когда сдаётся отчёт?", [], 3) == []

    expansion = Expansion(term="ПВТР", definition=f"{PVTR} Общества. Утверждены приказом.", doc_label="П")
    assert expand_query("отпуск по ПВТР", [expansion], chars=120) == (
        f"отпуск по ПВТР (ПВТР — {PVTR} Общества)"
    )
    assert expand_query("отпуск", [], chars=120) == "отпуск"
    assert short_definition(f"{PVTR} Общества", 20) == "правила внутреннего"
    # живой диалог 2026-09-17: расшифровка обрывалась на союзе («…от места добычи или»)
    assert short_definition("сооружение для транспортировки газа от места добычи или", 45) == (
        "сооружение для транспортировки газа от места"
    )


def test_first_expansion_reads_tool_result() -> None:
    structured = json.loads(
        GlossaryResult(
            term="ПВТР",
            entries=[
                GlossaryEntry(term="ПВТР", definition=PVTR, doc_id="d1", doc_label="Приказ №1"),
                GlossaryEntry(term="ПВТР", definition="другое", doc_id="d2", doc_label="Приказ №2"),
            ],
        ).model_dump_json()
    )
    expansion = first_expansion("ПВТР", structured)
    assert expansion is not None and expansion.definition == PVTR and expansion.doc_label == "Приказ №1"
    assert first_expansion("ПВТР", {"term": "ПВТР", "entries": [], "note": "нет"}) is None
    assert first_expansion("ПВТР", None) is None


async def test_runner_expands_abbreviation_before_search(parts: tuple[AgentTools, FakeGlossary]) -> None:
    tools, glossary = parts
    tracing = RecordingTracing()
    llms = {
        "tool_loop": ScriptedLLM(steps=[tool_step(TOOL_SEARCH, query="ПВТР"), "Заметки: S1 по вопросу."]),
        "answer": ScriptedLLM(steps=["Ответ по правилам [S1]."]),
        "rewrite": ScriptedLLM(steps=[]),
        "summary": ScriptedLLM(steps=[]),
        "verify": ScriptedLLM(steps=[], fallback='{"problems": []}'),
    }
    runner = AgentRunner(CONFIG, tools, lambda role: llms[role], tracing=tracing)
    events = [event async for event in runner.run("Что написано в ПВТР?", AgentSession(CONFIG))]

    kinds = [event.kind for event in events]
    assert kinds.index("glossary_used") < kinds.index("query_rewritten"), "расшифровка идёт до поиска"
    used = next(event for event in events if isinstance(event, GlossaryUsed))
    assert [item.term for item in used.expansions] == ["ПВТР"]
    rewritten = next(event for event in events if isinstance(event, QueryRewritten))
    assert rewritten.query == f"Что написано в ПВТР? (ПВТР — {PVTR} Общества)"
    assert glossary.terms == ["ПВТР"], "инструмент вызван один раз, бюджет цикла не тронут"
    step = next(item for item in tracing.steps if item["name"] == "glossary")
    assert step["input"] == {"terms": ["ПВТР"]} and step["output"][0]["term"] == "ПВТР"
    # расшифровка видна и циклу: модель сама формулирует запрос к hybrid_search
    loop_message = str(llms["tool_loop"].inputs[0][-1].content)
    assert LOOP_GLOSSARY_TITLE in loop_message and f"ПВТР — {PVTR}" in loop_message


async def test_ui_shows_glossary_step(parts: tuple[AgentTools, FakeGlossary]) -> None:
    tools, _ = parts
    llms = {
        "tool_loop": ScriptedLLM(steps=[tool_step(TOOL_SEARCH, query="ПВТР"), "Заметки: S1 по вопросу."]),
        "answer": ScriptedLLM(steps=["Ответ по правилам [S1]."]),
        "rewrite": ScriptedLLM(steps=[]),
        "summary": ScriptedLLM(steps=[]),
        "verify": ScriptedLLM(steps=[], fallback='{"problems": []}'),
    }
    runner = AgentRunner(CONFIG, tools, lambda role: llms[role])
    events = [event async for event in run_question(runner, AgentSession(CONFIG), "Что написано в ПВТР?")]

    finished = next(
        event for event in events if isinstance(event, StepFinished) and event.key == GLOSSARY_KEY
    )
    assert finished.title == "Расшифровываю по глоссарию: ПВТР"
    assert finished.output is not None and PVTR in finished.output
    keys = [event.key for event in events if isinstance(event, StepStarted)]
    assert keys.index(GLOSSARY_KEY) < keys.index("tool-1"), "шаг глоссария идёт до поиска"


async def test_unknown_term_does_not_change_query(parts: tuple[AgentTools, FakeGlossary]) -> None:
    tools, glossary = parts
    llms = {
        "tool_loop": ScriptedLLM(steps=[tool_step(TOOL_SEARCH, query="СИЗ"), "Заметки: S1 по вопросу."]),
        "answer": ScriptedLLM(steps=["Ответ [S1]."]),
        "rewrite": ScriptedLLM(steps=[]),
        "summary": ScriptedLLM(steps=[]),
        "verify": ScriptedLLM(steps=[], fallback='{"problems": []}'),
    }
    runner = AgentRunner(CONFIG, tools, lambda role: llms[role])
    question = "Нужны ли СИЗ на объекте?"
    events = [event async for event in runner.run(question, AgentSession(CONFIG))]

    assert "glossary_used" not in [event.kind for event in events]
    rewritten = next(event for event in events if isinstance(event, QueryRewritten))
    assert rewritten.query == question and glossary.terms == ["СИЗ"]


def test_keep_abbreviations_restores_case_lost_by_the_model() -> None:
    # регистр восстанавливается по слову целиком, каждое сокращение отдельно
    assert (
        keep_abbreviations("Отпуск по ПВТР и ЛПУМГ", "правила отпуска по пвтр в лпумг")
        == "правила отпуска по ПВТР в ЛПУМГ"
    )
    # верное написание не трогаем
    assert keep_abbreviations("Что такое ПВТР", "что такое ПВТР") == "что такое ПВТР"
    assert keep_abbreviations("без сокращений", "без сокращений") == "без сокращений"


def test_keep_abbreviations_appends_a_mangled_or_lost_one() -> None:
    """Живой диалог 2026-09-25: «КОЭ» переписалось как «кое» — буква искажена, не регистр."""
    assert (
        keep_abbreviations("Что такое КОЭ", "что означает аббревиатура кое")
        == "что означает аббревиатура кое КОЭ"
    )
    # модель раскрыла сокращение словами — дописываем его, sparse-вектор ищет по точному слову
    assert keep_abbreviations("Что такое КОЭ", "комплекс очистки эмульсии") == (
        "комплекс очистки эмульсии КОЭ"
    )
    # «кое-что» не превращается в «КОЭ-что»
    assert keep_abbreviations("Что такое КОЭ", "кое-что про эмульсию") == "кое-что про эмульсию КОЭ"
