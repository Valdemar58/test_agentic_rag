"""Цитаты (6.5, FR-4): маркеры [S#]/[D#] → нумерованные ссылки и детерминированный блок «Источники»."""

from __future__ import annotations

from agent.citations import (
    CARD_SOURCE,
    KNOWN_DOCUMENT_SOURCE,
    SOURCES_TITLE,
    cite_answer,
    strip_model_sources,
)
from agent.evidence import EvidenceRegistry


def _registry() -> EvidenceRegistry:
    registry = EvidenceRegistry(max_documents=10)
    registry.register_document(
        "doc-144",
        label="Приказ №144 от 15.01.2026",
        doc_status="active",
        card_text="[D1] …\nТема: Об охране труда",
    )
    registry.register_fragment(
        "chunk-1",
        doc_id="doc-144",
        kind="hit",
        breadcrumbs="Приказ №144 от 15.01.2026 → Раздел 3. Контроль → п. 3.2",
        text="Отчёт сдаётся до пятого числа.",
        parent_id="parent-1",
        clause="3.2",
        page_no=2,
    )
    registry.register_document("doc-109", label="Приказ №109 от 10.02.2025", doc_status="cancelled")
    registry.register_fragment(
        "chunk-2",
        doc_id="doc-109",
        kind="hit",
        breadcrumbs="Приказ №109 от 10.02.2025 → п. 1",
        text="Отчёт сдаётся до десятого числа.",
    )
    return registry


def test_markers_are_renumbered_in_order_of_appearance_with_sources_block() -> None:
    registry = _registry()
    cited = cite_answer(
        "Срок — до пятого числа [S1]. Старый приказ отменён [S2][S1]; статус подтверждён карточкой [S1, D1]. "
        "Старый приказ известен только из поиска [D2].",
        registry,
    )
    assert cited.body == (
        "Срок — до пятого числа [1]. Старый приказ отменён [2][1]; статус подтверждён карточкой [1][3]. "
        "Старый приказ известен только из поиска [4]."
    )
    assert [(source.number, source.alias, source.kind) for source in cited.sources] == [
        (1, "S1", "fragment"),
        (2, "S2", "fragment"),
        (3, "D1", "document"),
        (4, "D2", "document"),
    ]
    first = cited.sources[0]
    assert first.chunk_id == "chunk-1" and first.parent_id == "parent-1" and first.doc_id == "doc-144"
    assert first.clause == "3.2" and first.page_no == 2 and first.text == "Отчёт сдаётся до пятого числа."
    assert cited.sources[2].chunk_id is None and cited.sources[2].text == "[D1] …\nТема: Об охране труда"
    assert cited.text == cited.body + "\n\n" + cited.sources_block
    assert cited.sources_block.splitlines() == [
        f"{SOURCES_TITLE}:",
        "[1] Приказ №144 от 15.01.2026 → Раздел 3. Контроль → п. 3.2 (действует), стр. 2",
        "[2] Приказ №109 от 10.02.2025 → п. 1 (отменён)",
        f"[3] Приказ №144 от 15.01.2026 (действует) — {CARD_SOURCE}",
        f"[4] Приказ №109 от 10.02.2025 (отменён) — {KNOWN_DOCUMENT_SOURCE}",
    ]
    assert cited.unresolved == []


def test_unknown_markers_are_dropped_and_model_sources_block_is_stripped() -> None:
    registry = _registry()
    text = (
        "Срок — до пятого числа [S1][S9]. Документ D1 действует, а D7 неизвестен [D7].  \n\n"
        "**Источники:**  \n[S1] Приказ №144\n[S9] что-то"
    )
    cited = cite_answer(text, registry)
    assert (
        cited.body
        == "Срок — до пятого числа [1]. Документ Приказ №144 от 15.01.2026 действует, а D7 неизвестен ."
    )
    assert cited.unresolved == ["S9", "D7"] and [source.alias for source in cited.sources] == ["S1"]
    assert "Источники:" in cited.text and cited.text.count(SOURCES_TITLE) == 1


def test_answer_without_markers_is_returned_as_is() -> None:
    cited = cite_answer("В документах ответа нет. Искались правила парковки.", _registry())
    assert cited.text == "В документах ответа нет. Искались правила парковки." and cited.sources == []
    assert cited.sources_block == ""


def test_strip_model_sources_handles_headers_and_keeps_body() -> None:
    assert strip_model_sources("Ответ [S1].\n\n### Источники\n- [S1] …") == "Ответ [S1]."
    assert strip_model_sources("Ответ.\nИсточники:\n[1] x") == "Ответ."
    assert strip_model_sources("Источники указаны в тексте [S1].") == "Источники указаны в тексте [S1]."


def test_strip_model_links_block_only_when_it_is_a_list_of_markers() -> None:
    """Живой прогон 2026-09-16: модель дублировала ссылки блоком «Ссылки» вопреки промпту."""
    assert strip_model_sources("Ответ [S1].\nСсылки:\n[S1], [S2], [S1].") == "Ответ [S1]."
    assert strip_model_sources("Ответ [S1].\n\n**Ссылки:** [S1][S2][D1]") == "Ответ [S1]."
    assert strip_model_sources("Ответ.\nСсылки:\n[S1] Приказ №176 → Раздел 6\n[S2] Приказ №99") == "Ответ."
    # живой прогон 2026-09-17: строки блока через дефис — «- [S6] (начало и окончание диапазона)»
    assert (
        strip_model_sources("Ответ [S6].\n\nСсылки:\n- [S6] (диапазон)\n- [S6] (продолжительность)")
        == "Ответ [S6]."
    )
    kept = "Ответ.\nСсылки на ПВТР в тексте приказа [S1].\nОни обязательны."
    assert strip_model_sources(kept) == kept
    prose = "Ответ.\nСсылки:\nсм. раздел 6 [S1]"
    assert strip_model_sources(prose) == prose, "после заголовка проза — блок не трогаем"
