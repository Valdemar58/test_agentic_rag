"""Живые сценарии агента на стенде (6.7): AC-2.1 выбор инструмента, AC-1.1 несколько поисков,
AC-1.2 отказ, AC-6.1 уточняющие вопросы.

Нужны vLLM профиля runtime и MCP-сервер с индексом; корпус любой (синтетический или реальный) —
сценарии сформулированы по предметной области, а не по конкретным документам, и проверяют поведение
агента (какой инструмент выбран, сколько поисков, есть ли отказ, переписан ли запрос), а не текст
ответа. Запуск: `uv run pytest -m scenario`; ~10 минут на RTX 5080.
"""

from __future__ import annotations

import re
from collections.abc import AsyncIterator

import httpx
import pytest

from agent.rendering import TOOL_CARD, TOOL_CONTENT, TOOL_GLOSSARY, TOOL_RELATED, TOOL_SEARCH
from agent.runner import AgentRunner, AgentSession, QueryRewritten
from agent.service import build_runner
from common.config import AppConfig, load_app_config
from common.settings import Settings
from mcp_server.server import HEALTH_PATH

pytestmark = [pytest.mark.integration, pytest.mark.gpu, pytest.mark.scenario]

DOC_NUMBER_RE = re.compile(r"№\s*[\w/-]+")
SEARCH_QUESTION = "Какие требования установлены к отчётности по охране труда?"
NO_ANSWER_QUESTION = "Какие правила парковки личных велосипедов сотрудников установлены?"
MULTI_DOC_QUESTION = (
    "Какие инструкции по охране труда действуют сейчас и какими приказами они вводились раньше?"
)
CLARIFICATIONS: list[tuple[str, str, list[str]]] = [
    ("Какие инструкции по охране труда утверждены?", "а кто их согласовывал?", ["инструкц", "охран"]),
    ("Какие требования к средствам индивидуальной защиты?", "а для водителей?", ["защит", "водител"]),
    ("Что сказано в договорах о сроках оплаты?", "а о штрафах?", ["договор", "штраф"]),
    ("Какие приказы регулируют работу с газопроводами?", "а с какого числа?", ["газопровод"]),
    ("Кто отвечает за пожарную безопасность?", "а в филиалах?", ["пожарн", "филиал"]),
]


@pytest.fixture(scope="module")
def config() -> AppConfig:
    return load_app_config()


@pytest.fixture
async def runner(config: AppConfig) -> AsyncIterator[AgentRunner]:
    """Раннер на каждый тест: список инструментов MCP читается заново (дёшево), цикл событий — тестовый."""
    settings = Settings()
    llm_url = settings.resolve_llm_base_url(config)
    mcp_url = settings.resolve_mcp_url(config)
    try:
        httpx.get(f"{llm_url}/models", timeout=5).raise_for_status()
        httpx.get(mcp_url.rsplit("/", 1)[0] + HEALTH_PATH, timeout=5).raise_for_status()
    except httpx.HTTPError as exc:
        pytest.skip(f"стенд недоступен ({exc}); нужны профиль runtime и MCP-сервер")
    runner = await build_runner(config, settings)
    try:
        yield runner
    finally:
        await runner.aclose()


def _tools(answer_calls: list[str]) -> set[str]:
    return set(answer_calls)


async def test_ac21_content_question_starts_with_search(runner: AgentRunner, config: AppConfig) -> None:
    answer = await runner.ask(SEARCH_QUESTION, AgentSession(config))
    assert answer.tool_calls and answer.tool_calls[0].name == TOOL_SEARCH
    assert answer.search_queries and not answer.refused


async def test_ac21_follow_ups_pick_card_relations_and_content(
    runner: AgentRunner, config: AppConfig
) -> None:
    session = AgentSession(config)
    first = await runner.ask(SEARCH_QUESTION, session)
    assert first.document_aliases, "без найденных документов уточнения бессмысленны"
    # статус документа известен уже из поиска, а подписант и согласующие — только из карточки; если цикл
    # первого вопроса уже прочитал карточку, уточнение по FR-6 отвечается из кэша сессии без вызова
    card = await runner.ask("Кто подписал первый из найденных документов и кто его согласовывал?", session)
    card_calls = _tools([call.name for call in first.tool_calls + card.tool_calls])
    assert TOOL_CARD in card_calls and not card.refused
    related = await runner.ask("Какими документами он был отменён, изменён или дополнен?", session)
    assert {TOOL_RELATED, TOOL_SEARCH} & _tools([call.name for call in related.tool_calls])
    content = await runner.ask("Прочитай этот документ целиком и перескажи структуру по разделам", session)
    assert TOOL_CONTENT in _tools([call.name for call in content.tool_calls])


async def test_ac21_abbreviation_goes_to_glossary_first(runner: AgentRunner, config: AppConfig) -> None:
    answer = await runner.ask("Что означает аббревиатура ПВТР?", AgentSession(config))
    names = [call.name for call in answer.tool_calls]
    assert TOOL_GLOSSARY in names and names.index(TOOL_GLOSSARY) <= 1


async def test_ac21_filters_for_cancelled_kinds_and_dates(runner: AgentRunner, config: AppConfig) -> None:
    cancelled = await runner.ask("Какие приказы были отменены?", AgentSession(config))
    statuses = [
        (call.arguments.get("filters") or {}).get("statuses") or []
        for call in cancelled.tool_calls
        if call.name == TOOL_SEARCH
    ]
    assert any("cancelled" in item for item in statuses), statuses

    contracts = await runner.ask("Что сказано в договорах о сроках оплаты?", AgentSession(config))
    kinds = [
        " ".join((call.arguments.get("filters") or {}).get("doc_kinds") or []).casefold()
        for call in contracts.tool_calls
        if call.name == TOOL_SEARCH
    ]
    assert any("договор" in item for item in kinds), kinds

    dated = await runner.ask("Какие приказы изданы в 2026 году?", AgentSession(config))
    filters = [call.arguments.get("filters") or {} for call in dated.tool_calls if call.name == TOOL_SEARCH]
    dates = [str(item.get("date_from") or item.get("date_to") or "") for item in filters]
    assert any(item.startswith("2026") for item in dates), dates


async def test_ac11_multi_document_question_runs_several_searches(
    runner: AgentRunner, config: AppConfig
) -> None:
    answer = await runner.ask(MULTI_DOC_QUESTION, AgentSession(config))
    assert len(set(answer.search_queries)) >= 2, answer.search_queries


async def test_ac12_question_without_answer_is_refused(runner: AgentRunner, config: AppConfig) -> None:
    answer = await runner.ask(NO_ANSWER_QUESTION, AgentSession(config))
    assert answer.refused, answer.text[:300]


EVERYDAY_QUESTIONS = [
    "Во сколько я должен вернуться с обеда?",
    "Во сколько мне приходить на работу?",
    "Сколько дней отпуска мне положено?",
    "Могу ли я работать из дома?",
    "Что мне делать, если я заболел?",
    "Можно ли мне курить на территории?",
    "Кому я должен сообщить об опоздании?",
    "Нужно ли мне носить спецодежду?",
    "Когда мне выплатят зарплату?",
    "Можно ли играть в настольный теннис в офисе?",
    "Мне нужно ехать в командировку, что оформить?",
    "Как мне получить пропуск?",
]
SMALL_TALK = ["Привет! Что ты умеешь?", "Спасибо, всё понятно"]
LUNCH_QUESTION = EVERYDAY_QUESTIONS[0]
RULES_QUESTION = "Какие правила внутреннего трудового распорядка установлены?"
APPENDIX_FOLLOW_UP = "Каким должностям установлен суммированный учёт рабочего времени?"


async def test_everyday_first_person_questions_are_routed_to_search(
    runner: AgentRunner, config: AppConfig
) -> None:
    """Живой диалог 2026-09-16: «во сколько вернуться с обеда» уходил в режим беседы без единого поиска."""
    for question in EVERYDAY_QUESTIONS:
        rewritten = await runner.rewrite(question, AgentSession(config))
        assert rewritten.needs_search, f"«{question}» → {rewritten.intent}: {rewritten.reason}"
    for message in SMALL_TALK:
        rewritten = await runner.rewrite(message, AgentSession(config))
        assert not rewritten.needs_search, f"«{message}» → {rewritten.intent}: {rewritten.reason}"


async def test_lunch_question_is_answered_from_rules_with_citations(
    runner: AgentRunner, config: AppConfig
) -> None:
    answer = await runner.ask(LUNCH_QUESTION, AgentSession(config))
    assert answer.needs_search and answer.search_queries, "поиск обязателен, даже если модель его пропустила"
    assert not answer.refused and answer.sources, answer.text[:300]
    assert any(source.kind == "fragment" for source in answer.sources)


async def test_appendix_list_follow_up_cites_the_fragment_with_the_list(
    runner: AgentRunner, config: AppConfig
) -> None:
    """Живой диалог 2026-09-16: перечень должностей из приложения выдавался без ссылки на само приложение."""
    session = AgentSession(config)
    await runner.ask(RULES_QUESTION, session)
    answer = await runner.ask(APPENDIX_FOLLOW_UP, session)
    assert not answer.refused, answer.text[:300]
    cited = [
        source
        for source in answer.sources
        if source.kind == "fragment" and "риложение" in (source.breadcrumbs or "")
    ]
    assert cited, [source.line() for source in answer.sources]
    assert any(re.search(r"(?m)^\s*\d+\.\s", source.text or "") for source in cited), (
        "по ссылке открывается фрагмент с самим перечнем, а не с упоминанием приложения"
    )


@pytest.mark.parametrize(("question", "clarification", "stems"), CLARIFICATIONS)
async def test_ac61_clarification_is_rewritten_into_self_contained_query(
    runner: AgentRunner, config: AppConfig, question: str, clarification: str, stems: list[str]
) -> None:
    session = AgentSession(config)
    await runner.ask(question, session)
    events = [event async for event in runner.run(clarification, session)]
    rewritten = next(event for event in events if isinstance(event, QueryRewritten))
    assert rewritten.changed and rewritten.needs_search, rewritten
    query = rewritten.query.casefold()
    # самодостаточность: предмет прошлого вопроса (основы слов) или конкретные документы из прошлого ответа
    names_documents = DOC_NUMBER_RE.search(query) is not None
    missing = [stem for stem in stems if stem not in query]
    assert not missing or names_documents, f"в переписанном запросе «{rewritten.query}» нет {missing}"
