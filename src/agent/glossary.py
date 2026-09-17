"""Расшифровка аббревиатур вопроса по глоссарию до поиска (FR-5, пункт «б»).

Термины для расшифровки называет шаг переписывания (поле `abbreviations`), а прописные сокращения
вопроса дополнительно находит regexp: модель называет их не всегда. Каждый термин ищется в глоссарии
прямым вызовом MCP мимо цикла — бюджет вызовов инструментов (FR-1) на это не тратится. Найденная
расшифровка не подменяет запрос, а добавляется к нему: поиску нужны и сокращение (sparse-вектор ищет
по словам), и полная форма. Из нескольких определений одного термина берётся первое — инструмент
отдаёт записи действующих документов первыми.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import BaseModel, Field

# сокращение: две и более прописные буквы подряд, допускаются цифры и дефис («ПВТР», «1С:УАТ» → «УАТ»)
ABBREVIATION_RE = re.compile(r"\b[А-ЯЁA-Z]{2}[А-ЯЁA-Z0-9-]{0,8}\b")
ENTRIES_KEY = "entries"
EDGE_CHARS = " ,;:.-–—"
# служебные слова: на них обрезанная расшифровка обрываться не должна
DANGLING_WORDS = frozenset(
    "и или а но от до для с со в во к ко на по при из у о об за что как это его её их".split()
)


class Expansion(BaseModel):
    term: str = Field(description="Аббревиатура или термин из вопроса")
    definition: str = Field(description="Расшифровка из глоссария")
    doc_label: str = Field(default="", description="Документ-источник расшифровки")


def candidate_terms(question: str, named: Sequence[str], limit: int) -> list[str]:
    """Термины для глоссария: сначала названные моделью, затем прописные сокращения вопроса."""
    terms: list[str] = []
    seen: set[str] = set()
    for term in [*named, *ABBREVIATION_RE.findall(question)]:
        cleaned = " ".join(str(term).split())
        key = cleaned.casefold()
        if not cleaned or key in seen:
            continue
        seen.add(key)
        terms.append(cleaned)
        if len(terms) >= limit:
            break
    return terms


def first_expansion(term: str, structured: Mapping[str, Any] | None) -> Expansion | None:
    """Первая запись ответа `glossary_lookup` → расшифровка; пустой ответ и мусор — None."""
    entries = (structured or {}).get(ENTRIES_KEY)
    if not isinstance(entries, list) or not entries:
        return None
    entry = entries[0]
    if not isinstance(entry, dict) or not str(entry.get("definition") or "").strip():
        return None
    return Expansion(
        term=str(entry.get("term") or term),
        definition=" ".join(str(entry["definition"]).split()),
        doc_label=str(entry.get("doc_label") or ""),
    )


def short_definition(definition: str, chars: int) -> str:
    """Расшифровка для запроса: первое предложение, не длиннее `chars` символов.

    Обрыв на союзе или предлоге («…от места добычи или») в запрос не попадает: такие слова с конца
    снимаются вместе со знаками."""
    text = definition.split(". ")[0].strip().rstrip(".")
    if len(text) > chars:
        text = text[:chars].rsplit(" ", 1)[0]
    words = text.split()
    while words and words[-1].casefold().strip(EDGE_CHARS) in DANGLING_WORDS:
        words.pop()
    return " ".join(words).rstrip(EDGE_CHARS)


def expand_query(query: str, expansions: Sequence[Expansion], *, chars: int) -> str:
    """Запрос с расшифровками в скобках: «отпуск по ПВТР (ПВТР — правила внутреннего распорядка)»."""
    parts = [f"{item.term} — {short_definition(item.definition, chars)}" for item in expansions]
    return f"{query} ({'; '.join(parts)})" if parts else query
