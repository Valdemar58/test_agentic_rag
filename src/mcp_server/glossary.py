"""`glossary_lookup` (FR-2.5, FR-5): расшифровка терминов и аббревиатур по глоссарию организации.

До этапа 8 глоссарий пуст: коллекция ещё не строится, инструмент отвечает «не найдено» с
пояснением. Модель ответа и протокол зафиксированы сейчас, чтобы агент и тесты этапа 6 не менялись.
"""

from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel, Field

EMPTY_NOTE = "глоссарий ещё не построен (FR-5, этап 8): термин не найден"


class GlossaryEntry(BaseModel):
    term: str = Field(description="Термин или аббревиатура, как в документе")
    definition: str = Field(description="Определение или расшифровка")
    doc_id: str = Field(description="Документ-источник")
    doc_label: str = Field(description="Документ: вид, номер, дата")
    section_id: str | None = Field(default=None, description="Раздел-источник для get_document_content")


class GlossaryResult(BaseModel):
    term: str
    entries: list[GlossaryEntry]
    note: str | None = None


class GlossaryLookup(Protocol):
    def lookup(self, term: str) -> GlossaryResult: ...


class EmptyGlossary:
    """Заглушка до этапа 8."""

    def lookup(self, term: str) -> GlossaryResult:
        return GlossaryResult(term=term, entries=[], note=EMPTY_NOTE)
