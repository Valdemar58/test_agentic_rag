"""Элементы цитат для UI (7.2, FR-4): имя совпадает с маркером ссылки, содержимое — фрагмент или карточка."""

from __future__ import annotations

from agent.citations import Source
from ui.citations import (
    CARD_NOT_REQUESTED,
    CARD_TITLE,
    CONTEXT_TITLE,
    SOURCES_NAME,
    citation_name,
    citations_for,
)


def _fragment(number: int, *, context: str | None = None) -> Source:
    return Source(
        number=number,
        alias="S1",
        kind="fragment",
        doc_id="doc-1",
        chunk_id="chunk-1",
        parent_id="parent-1",
        label="Приказ №144 от 15.01.2026",
        doc_status="действует",
        breadcrumbs="Приказ №144 от 15.01.2026 → 2. Отчётность → 2.1",
        clause="2.1",
        page_no=3,
        text="Отчёт сдаётся до пятого числа.",
        context=context,
    )


def _document(number: int, *, card_text: str | None) -> Source:
    return Source(
        number=number,
        alias="D1",
        kind="document",
        doc_id="doc-1",
        chunk_id=None,
        label="Приказ №144 от 15.01.2026",
        doc_status="действует",
        breadcrumbs=None,
        text=card_text,
    )


def test_fragment_citation_shows_document_place_and_exact_text() -> None:
    citation, summary = citations_for(
        [_fragment(1, context="2. Отчётность. 2.1 Отчёт сдаётся до пятого числа.")]
    )
    assert citation.name == citation_name(1) == "[1]"
    # сводный элемент последним: боковая панель Chainlit называется его именем
    assert summary.name == SOURCES_NAME == "Список источников"
    assert summary.content == "- [1] Приказ №144 от 15.01.2026 → 2. Отчётность → 2.1 (действует), стр. 3"
    lines = citation.content.splitlines()
    assert lines[0] == "**Приказ №144 от 15.01.2026** — действует"
    assert "Раздел: Приказ №144 от 15.01.2026 → 2. Отчётность → 2.1" in lines
    assert "Пункт: 2.1" in lines and "Страница: 3" in lines
    assert "Отчёт сдаётся до пятого числа." in lines
    assert CONTEXT_TITLE in lines and lines[-1].startswith("2. Отчётность.")


def test_fragment_without_context_has_no_context_block() -> None:
    citation, _ = citations_for([_fragment(2)])
    assert CONTEXT_TITLE not in citation.content and citation.content.endswith(
        "Отчёт сдаётся до пятого числа."
    )
    assert citations_for([]) == []


def test_document_citation_shows_card_or_says_it_was_not_requested() -> None:
    with_card, without_card, summary = citations_for(
        [_document(1, card_text="Подписант: Иванов; согласующие: Петров"), _document(2, card_text=None)]
    )
    assert summary.content.splitlines() == [
        "- [1] Приказ №144 от 15.01.2026 (действует) — карточка документа",
        "- [2] Приказ №144 от 15.01.2026 (действует) — документ СЭД (карточка не запрашивалась)",
    ]
    assert (
        with_card.name == "[1]"
        and CARD_TITLE in with_card.content
        and "Подписант: Иванов" in with_card.content
    )
    assert without_card.name == "[2]" and CARD_NOT_REQUESTED in without_card.content
