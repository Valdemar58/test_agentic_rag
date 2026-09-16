"""Переписывание запроса (6.3, FR-6): разбор JSON модели, промпт с историей и документами, отключение."""

from __future__ import annotations

from agent.evidence import EvidenceRegistry
from agent.memory import Turn, render_history
from agent.rewrite import FALLBACK_REASON, QueryRewriter, parse_rewrite
from common.config import DEFAULT_CONFIG_PATH, load_app_config
from tests.unit.agent_fakes import ScriptedLLM

CONFIG = load_app_config(DEFAULT_CONFIG_PATH)
QUESTION = "а для филиалов?"


def test_parse_takes_json_from_text_and_filters_aliases() -> None:
    text = (
        "Вот запрос:\n"
        '{"query": "требования к СИЗ для филиалов", "needs_search": true, '
        '"relevant_documents": ["D1", "S3", "не псевдоним"], "abbreviations": ["СИЗ"], "reason": "уточнение"}'
    )
    result = parse_rewrite(QUESTION, text, thinking="думал")
    assert result.query == "требования к СИЗ для филиалов" and result.changed
    assert result.needs_search and result.relevant_documents == ["D1"] and result.abbreviations == ["СИЗ"]
    assert result.reason == "уточнение" and result.thinking == "думал"


def test_parse_falls_back_to_the_question() -> None:
    for text in ("не json", '{"query": 5, "needs_search": [}', "", '["список"]'):
        result = parse_rewrite(QUESTION, text)
        assert result.query == QUESTION and result.needs_search and not result.changed
        assert result.reason == FALLBACK_REASON
    empty_query = parse_rewrite(QUESTION, '{"query": "", "needs_search": false}')
    assert empty_query.query == QUESTION and empty_query.needs_search is False


async def test_rewriter_prompt_contains_history_and_known_documents() -> None:
    llm = ScriptedLLM(steps=['{"query": "срок сдачи отчёта по охране труда для филиалов"}'])
    registry = EvidenceRegistry(max_documents=5)
    registry.register_document(
        "doc-1", label="Приказ №144 от 15.01.2026", doc_status="active", subject="Об охране"
    )
    registry.register_document("doc-2", label="Договор №Д-1", doc_status="active")
    turns = [
        Turn(
            question="Когда сдаётся отчёт по охране труда?",
            answer="До пятого числа [S1]." * 200,
            document_aliases=["D1"],
        )
    ]
    rewriter = QueryRewriter(llm, CONFIG.agent.rewrite)
    result = await rewriter.rewrite(QUESTION, turns, registry.documents(), summary="Обсуждали отчётность.")
    assert result.query == "срок сдачи отчёта по охране труда для филиалов" and result.changed
    system, user = llm.inputs[0]
    assert "JSON" in str(system.content) and "needs_search" in str(system.content)
    user_text = str(user.content)
    assert "Сводка предыдущего диалога: Обсуждали отчётность." in user_text
    assert "Пользователь: Когда сдаётся отчёт по охране труда?" in user_text
    assert "…(обрезано)" in user_text, "длинный прошлый ответ обрезан до answer_chars"
    assert "[D1] Приказ №144 от 15.01.2026 (действует) — «Об охране» — в последнем ответе" in user_text
    assert "[D2] Договор №Д-1 (действует)\n" in user_text, "документ не из последнего ответа — без пометки"
    assert user_text.endswith(f"Новый вопрос пользователя: {QUESTION}")


async def test_rewriter_disabled_passes_question_through() -> None:
    llm = ScriptedLLM(steps=['{"query": "другое"}'])
    settings = CONFIG.agent.rewrite.model_copy(update={"enabled": False})
    result = await QueryRewriter(llm, settings).rewrite(QUESTION, [], [], None)
    assert result.query == QUESTION and not llm.inputs


def test_render_history_without_turns() -> None:
    assert render_history([], answer_chars=100) == "(это первый вопрос в диалоге)"
