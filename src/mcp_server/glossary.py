"""`glossary_lookup` (FR-2.5, FR-5): расшифровка терминов и аббревиатур по глоссарию организации.

Термин ищется в коллекции `glossary` (её собирает `ingest.glossary`): сначала точным совпадением
нормализованного ключа, потом векторами bge-m3 — но найденное векторами берётся, только если термин
записи действительно похож на запрошенный (вхождение или difflib не ниже `glossary.match_ratio`),
иначе инструмент отдавал бы «похожие по смыслу» определения чужих терминов. Записи из действующих
документов идут первыми, статус документа-источника виден агенту. Пока коллекция не собрана,
инструмент отвечает «не найдено» с пояснением — модель ответа и протокол те же, что с этапа 6.
"""

from __future__ import annotations

import difflib
import logging
from typing import Protocol

from pydantic import BaseModel, Field

from common.config import DocStatus, GlossarySettings
from ingest.embeddings import Embedder
from ingest.glossary import GlossaryRecord, normalize_term
from ingest.glossary_index import GlossaryIndex

logger = logging.getLogger(__name__)

EMPTY_NOTE = "глоссарий ещё не построен (FR-5, этап 8): термин не найден"
MISSING_NOTE = "термина «{term}» в глоссарии нет: ищи по самому термину через hybrid_search"
EMPTY_TERM_NOTE = "пустой запрос: укажи термин или аббревиатуру"
UNAVAILABLE_NOTE = "глоссарий недоступен: ищи по самому термину через hybrid_search"
# короче этого термин считается сокращением: только точное совпадение, без вхождений и похожести
MIN_STEM_CHARS = 5
# «документы» и «документ»: одно слово с разным окончанием
MAX_STEM_DIFF = 3


class GlossaryEntry(BaseModel):
    term: str = Field(description="Термин или аббревиатура, как в документе")
    definition: str = Field(description="Определение или расшифровка")
    doc_id: str = Field(description="Документ-источник")
    doc_label: str = Field(description="Документ: вид, номер, дата")
    doc_status: DocStatus | None = Field(
        default=None, description="Статус документа-источника: active, cancelled, draft"
    )
    section_id: str | None = Field(default=None, description="Раздел-источник для get_document_content")


class GlossaryResult(BaseModel):
    term: str
    entries: list[GlossaryEntry]
    note: str | None = None


class GlossaryLookup(Protocol):
    def lookup(self, term: str) -> GlossaryResult: ...


class EmptyGlossary:
    """Глоссарий выключен в конфиге или коллекция не собрана."""

    def lookup(self, term: str) -> GlossaryResult:
        return GlossaryResult(term=term, entries=[], note=EMPTY_NOTE)


def _entry(record: GlossaryRecord) -> GlossaryEntry:
    return GlossaryEntry(
        term=record.term,
        definition=record.definition,
        doc_id=record.doc_id,
        doc_label=record.doc_label,
        doc_status=record.doc_status,
        section_id=record.section_id,
    )


def matches(key: str, candidate: str, ratio: float) -> bool:
    """Термин записи отвечает запросу.

    Короткий термин — сокращение, и для него годится только точное совпадение: живой диалог 2026-09-17
    показал, что вхождение и похожесть дают чужое определение («ЛПУМГ» → «МГ», «ЛПУ» → «ПУ» с difflib
    0.8). Для слов длиннее `MIN_STEM_CHARS` допускается разное окончание («документы» и «документ»),
    для словосочетаний — вхождение («средства индивидуальной защиты» в «… защиты работника»). Вхождение
    одного слова в словосочетание не считается совпадением: «документы» — не «организационно-
    распорядительные документы». Всё остальное решает похожесть не ниже порога."""
    if not key or not candidate:
        return False
    if key == candidate:
        return True
    short, long = (key, candidate) if len(key) <= len(candidate) else (candidate, key)
    if len(short) < MIN_STEM_CHARS:
        return False
    if " " in short:
        if short in long:
            return True
    elif " " not in long and long.startswith(short) and len(long) - len(short) <= MAX_STEM_DIFF:
        return True
    return difflib.SequenceMatcher(None, key, candidate).ratio() >= ratio


def order(records: list[GlossaryRecord]) -> list[GlossaryRecord]:
    """Действующие документы первыми; одинаковые определения одного термина не повторяются."""
    ordered = sorted(records, key=lambda record: (record.doc_status != "active", record.doc_label))
    seen: set[tuple[str, str]] = set()
    unique: list[GlossaryRecord] = []
    for record in ordered:
        key = (record.term_key, normalize_term(record.definition))
        if key in seen:
            continue
        seen.add(key)
        unique.append(record)
    return unique


class QdrantGlossary:
    """Глоссарий из коллекции Qdrant: точное совпадение термина, затем поиск векторами."""

    def __init__(self, index: GlossaryIndex, embedder: Embedder, settings: GlossarySettings) -> None:
        self._index = index
        self._embedder = embedder
        self._settings = settings

    def lookup(self, term: str) -> GlossaryResult:
        key = normalize_term(term)
        if not key:
            return GlossaryResult(term=term, entries=[], note=EMPTY_TERM_NOTE)
        limit = self._settings.lookup_top_k
        try:
            if not self._index.exists():
                return GlossaryResult(term=term, entries=[], note=EMPTY_NOTE)
            records = self._index.by_term(key, limit)
            if not records:
                found = self._index.search(self._embedder.encode([term])[0], limit)
                records = [
                    record for record in found if matches(key, record.term_key, self._settings.match_ratio)
                ]
        except Exception as exc:  # noqa: BLE001 — справочный инструмент не должен ронять ответ агента
            logger.warning("glossary_lookup «%s»: глоссарий недоступен (%s)", term[:60], exc)
            return GlossaryResult(term=term, entries=[], note=UNAVAILABLE_NOTE)
        entries = [_entry(record) for record in order(records)[:limit]]
        logger.info("glossary_lookup «%s»: записей %d", term[:60], len(entries))
        return GlossaryResult(
            term=term, entries=entries, note=None if entries else MISSING_NOTE.format(term=term)
        )
