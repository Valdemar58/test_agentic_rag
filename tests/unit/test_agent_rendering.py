"""Свидетельства для итогового ответа: порядок, контекст раздела один раз, обрезка вместо выбрасывания.

Живой диалог 2026-09-16: прочитанный раздел с перечнем должностей шёл последним и целиком выпал из бюджета
ответа, а четыре фрагмента одного раздела принесли четыре одинаковых контекста.
"""

from __future__ import annotations

from agent.evidence import EvidenceRegistry
from agent.rendering import (
    CUT_MARK,
    EVIDENCE_OVERFLOW_NOTE,
    NO_CARD_NOTE,
    NO_EVIDENCE,
    render_evidence,
)
from common.config import DEFAULT_CONFIG_PATH, load_app_config

SETTINGS = load_app_config(DEFAULT_CONFIG_PATH).agent.answer
DOC = "doc-1"
SECTION_6 = "parent-6"
SECTION_10 = "parent-10"
CONTEXT_6 = "6.1. Режим работы: " + "слово " * 120
CONTEXT_10 = "10.1. Ответственность: " + "текст " * 60


def registry_with_dialog() -> EvidenceRegistry:
    registry = EvidenceRegistry(max_documents=5)
    registry.register_document(DOC, label="Приказ №176 от 19.08.2024", doc_status="active", subject="ПВТР")
    registry.register_fragment(  # S1 — из кэша прошлого хода
        "chunk-1", doc_id=DOC, kind="hit", breadcrumbs="Раздел 1", text="1.3. Общие положения.", context="1.1"
    )
    registry.register_fragment(  # S2, S3 — свежие фрагменты одного раздела 6
        "chunk-2",
        doc_id=DOC,
        kind="hit",
        breadcrumbs="Раздел 6",
        text="скользящий график",
        parent_id=SECTION_6,
        context=CONTEXT_6,
    )
    registry.register_fragment(
        "chunk-3",
        doc_id=DOC,
        kind="hit",
        breadcrumbs="Раздел 6 → п. 6.4",
        text="6.4. особый характер",
        parent_id=SECTION_6,
        context=CONTEXT_6,
        clause="6.4",
    )
    registry.register_fragment(  # S4 — заголовок приложения из раздела 10, сам раздел прочитан (S5)
        "chunk-4",
        doc_id=DOC,
        kind="hit",
        breadcrumbs="Раздел 10",
        text="Перечень должностей",
        parent_id=SECTION_10,
        context=CONTEXT_10,
    )
    registry.register_fragment(
        SECTION_10,
        doc_id=DOC,
        kind="section",
        breadcrumbs="Раздел 10",
        text=CONTEXT_10 + " 1. Мастер склада.",
        parent_id=SECTION_10,
    )
    return registry


def _order(text: str) -> list[str]:
    return [line.split("]")[0] + "]" for line in text.splitlines() if line.startswith("[S")]


def test_sections_first_fresh_hits_then_cache_and_context_once_per_section() -> None:
    registry = registry_with_dialog()
    text = render_evidence(registry, ["S1", "S2", "S3", "S4", "S5"], ["D1"], SETTINGS, cached_aliases=["S1"])
    assert _order(text) == ["[S5]", "[S2]", "[S3]", "[S4]", "[S1]"]
    assert text.count("Контекст раздела: 6.1. Режим работы") == 1, "контекст раздела 6 показан один раз"
    assert "Контекст раздела: 10.1." not in text, "раздел 10 уже среди свидетельств целиком"
    assert "Контекст раздела: 1.1" in text
    assert f"[D1] Приказ №176 от 19.08.2024 — действует; тема: «ПВТР»{NO_CARD_NOTE}" in text


def test_oversized_fragment_is_cut_and_tiny_remainder_is_noted() -> None:
    registry = registry_with_dialog()
    # документы и раздел S5 занимают ≈ 660 символов, блок S2 с контекстом ≈ 790: при бюджете 1100 остаток
    # больше минимального блока — S2 обрезается; при 800 остаток крошечный — S2 не показывается
    generous = SETTINGS.model_copy(update={"evidence_max_chars": 1100})
    text = render_evidence(registry, ["S2", "S5"], ["D1"], generous)
    assert _order(text) == ["[S5]", "[S2]"]
    assert CUT_MARK in text and EVIDENCE_OVERFLOW_NOTE not in text, "второй блок обрезан, а не выброшен"
    assert len(text) <= 1100 + len(CUT_MARK) + len("\n") * 8
    tight = SETTINGS.model_copy(update={"evidence_max_chars": 800})
    text = render_evidence(registry, ["S2", "S5"], ["D1"], tight)
    assert _order(text) == ["[S5]"] and text.endswith(EVIDENCE_OVERFLOW_NOTE)


def test_without_fragments_and_documents_there_is_no_evidence() -> None:
    assert render_evidence(EvidenceRegistry(max_documents=5), [], [], SETTINGS) == NO_EVIDENCE
    registry = registry_with_dialog()
    text = render_evidence(registry, ["S9"], ["D1"], SETTINGS)
    assert _order(text) == [] and text.endswith("Фрагменты:")


def test_documents_section_is_bounded_by_half_the_budget() -> None:
    """Прогон 2026-09-18: сводки карточек договора с допсоглашениями заняли 40 000 символов промпта."""
    registry = EvidenceRegistry(20)
    aliases: list[str] = []
    for number in range(6):
        document = registry.register_document(
            f"doc-{number}",
            label=f"Договорной документ №{number}",
            doc_status="active",
            card_text="сводка\n" + "\n".join(f"    поле {index}: значение" for index in range(40)),
        )
        aliases.append(document.alias)
    settings = SETTINGS.model_copy(update={"evidence_max_chars": 4000})
    rendered = render_evidence(registry, [], aliases, settings)
    documents_part = rendered.split("Фрагменты:")[0]
    assert len(documents_part) <= settings.evidence_max_chars, "раздел документов держится в половине бюджета"
    assert rendered.count("поле 39") < len(aliases), "сводки карточек сверх предела не печатаются"
    assert all(f"[{alias}]" in rendered for alias in aliases), "сам документ виден всегда"
