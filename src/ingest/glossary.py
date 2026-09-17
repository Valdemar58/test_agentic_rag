"""Глоссарий организации (FR-5): разделы «Термины и определения» и «Сокращения» → записи глоссария.

Источник — уже построенный индекс документов, а не повторный разбор файлов: берутся чанки, у которых
заголовок раздела (или крошка пути) совпадает с `glossary.headings`, и из их текста код выделяет
кандидатов «термин — определение»: строки перечня («ПВТР — правила внутреннего трудового распорядка;»),
строки с двоеточием и строки markdown-таблицы «| термин | определение |». Дальше раздел подтверждает LLM
(роль `glossary`, ТЗ FR-5): она отвечает, глоссарий ли это вообще и какие строки терминами не являются, —
а выбрасывает их снова код. Порядок тот же, что при проверке ответа (`agent/verify.py`): модель называет,
код применяет; ничего дописать она не может. Раздел, ответ по которому не разобран, в глоссарий не
попадает — подтверждение обязательно.

Сборка идёт при поднятом профиле runtime (нужна Qwen), то есть после инжеста: dots.mocr к этому времени
выгружен, а векторы записей считает bge-m3 на CPU, как в рантайме (§2 ТЗ).
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import re
import time
import uuid
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from llama_index.core.llms import LLM, ChatMessage
from openai import OpenAIError
from pydantic import BaseModel, ConfigDict, Field

from common.config import DocStatus, GlossarySettings
from ingest.embeddings import Embedder
from ingest.metadata import ChunkPayload, document_label

logger = logging.getLogger(__name__)

GLOSSARY_NAMESPACE = uuid.UUID("2b0f5c2a-9e44-4b71-8a1d-1f0c3d6e7a95")
# поля payload, по которым чанк опознаётся как часть раздела глоссария (читаются отдельным проходом)
HEADING_FIELDS = ("heading", "section_path", "chunk_level")
ELLIPSIS = "…"

DASH_CHARS = "—–‒―−-"
# маркер записи перечня в начале строки: «-», «•», «1.», «2)», «2.1.»
_MARKER_RE = re.compile(r"^\s*(?:[-–—•*▪]|\d+[.)]|\d+(?:\.\d+)+\.?)\s+")
_DASH_RE = re.compile(rf"\s+[{DASH_CHARS}]{{1,2}}\s+")
_COLON_RE = re.compile(r":\s+")
_TABLE_ROW_RE = re.compile(r"^\s*\|(?P<cells>.+)\|\s*$")
_TABLE_RULE_RE = re.compile(r"^[\s|:-]+$")
_LETTER_RE = re.compile(r"[^\W\d_]")
_UPPER_PAIR_RE = re.compile(r"[А-ЯЁA-Z]{2}")
_LOWER_START_RE = re.compile(r"^[а-яёa-z]")
# пробел перед знаком препинания — артефакт разбора («авансовый отчёт .»)
_PUNCT_SPACE_RE = re.compile(r"\s+([.,;:])")
PLAIN_HYPHEN = "-"
# номер и слово «раздел» в начале заголовка: «Раздел 2. Термины и определения» → «термины и определения»
_HEADING_NUMBER_RE = re.compile(r"^\s*(?:раздел\s+|глава\s+|п\.\s*)?\d+(?:\.\d+)*[.)]?\s*", re.IGNORECASE)
_EDGE_RE = re.compile(r"^[\s«»\"'*_#.,;:()]+|[\s«»\"'*_#.,;:()]+$")
_SPACES_RE = re.compile(r"\s+")
# шапка таблицы терминов: такие строки — не записи глоссария
TABLE_HEADER_CELLS = frozenset(
    {"термин", "термины", "сокращение", "сокращения", "аббревиатура", "обозначение", "понятие", "определение"}
)
# JSON без экранирования переводов строк и только первый объект — как в agent/verify.py
_DECODER = json.JSONDecoder(strict=False)

CONFIRM_SYSTEM_PROMPT = """Ты проверяешь выписку из раздела документа организации, похожего на глоссарий.
Тебе дают строки «термин — определение», которые выделены из текста раздела автоматически.

Ответь строго одним объектом JSON и ничем больше:
{"glossary": true или false, "reject": [номера строк], "reason": "коротко, одной фразой"}

glossary — true, если раздел действительно перечень терминов, сокращений или определений; false, если это
обычный текст (обязанности, порядок, сроки, перечень должностей или документов).
reject — номера строк, которые не являются парой «термин — определение»: обычное предложение пункта,
заголовок, ссылка на документ, фамилия, должность, единица измерения без пояснения. Если все строки —
термины, верни пустой список.
Ничего не переписывай, не дополняй и не добавляй новых строк: только номера."""

CONFIRM_USER_TEMPLATE = """Документ: {document}
Раздел: {section}

Строки:
{lines}"""


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class GlossaryRecord(_Frozen):
    """Запись глоссария — точка коллекции Qdrant (payload)."""

    record_id: str = Field(description="Детерминированный id точки: uuid5 от чанка и ключа термина")
    term: str = Field(description="Термин или аббревиатура, как в документе")
    term_key: str = Field(description="Нормализованный термин для точного поиска")
    definition: str = Field(description="Определение или расшифровка")
    doc_id: str
    doc_label: str = Field(description="Документ: вид, номер, дата")
    doc_kind: str
    doc_status: DocStatus
    section_id: str = Field(description="Раздел-источник (parent_id или chunk_id) для get_document_content")
    chunk_id: str
    file_row_id: str = Field(description="Файл-источник: по нему чистятся записи исчезнувших файлов")
    breadcrumbs: str

    @property
    def text(self) -> str:
        """Текст записи для эмбеддинга и показа агенту."""
        return f"{self.term} — {self.definition}"


@dataclass(frozen=True)
class Candidate:
    term: str
    definition: str
    chunk: ChunkPayload


@dataclass(frozen=True)
class Section:
    """Раздел глоссария одного файла: чанки под одним заголовком и кандидаты из них."""

    document: str
    heading: str
    candidates: list[Candidate]


@dataclass(frozen=True)
class SectionVerdict:
    glossary: bool
    reject: set[int] = field(default_factory=set)
    reason: str = ""
    parsed: bool = True


class ChunkSource(Protocol):
    """Чтение индекса документов: отдельный проход по полям заголовка и чтение чанков по id."""

    def scroll_payload_fields(self, fields: Sequence[str]) -> Iterator[tuple[str, Mapping[str, Any]]]: ...

    def get_chunks(self, ids: Sequence[str]) -> list[ChunkPayload]: ...


class GlossaryStore(Protocol):
    """Коллекция глоссария: `ingest.glossary_index.GlossaryIndex`, в тестах — фейк в памяти."""

    def ensure_collection(self) -> None: ...

    def upsert(self, records: Sequence[GlossaryRecord], embeddings: Sequence[Any]) -> int: ...

    def prune(self, keep: set[str]) -> int: ...


class TermConfirmer(Protocol):
    async def confirm(self, section: Section) -> SectionVerdict: ...


# ---------- разбор кандидатов ----------


def normalize_term(text: str) -> str:
    """Ключ термина: без регистра, ё→е, без кавычек и знаков по краям («ПВТР;» и «"ПВТР"» → «пвтр»)."""
    return _SPACES_RE.sub(" ", _EDGE_RE.sub("", text.casefold().replace("ё", "е"))).strip()


def normalize_heading(text: str) -> str:
    """Заголовок без номера раздела и знаков по краям, в нижнем регистре."""
    return normalize_term(_HEADING_NUMBER_RE.sub("", text or ""))


def is_glossary_heading(text: str, headings: Sequence[str]) -> bool:
    """Заголовок раздела глоссария: нормализованный заголовок содержит одну из настроенных фраз."""
    normalized = normalize_heading(text)
    return bool(normalized) and any(phrase in normalized for phrase in headings)


def is_glossary_chunk(fields: Mapping[str, Any], headings: Sequence[str]) -> bool:
    """Чанк из раздела глоссария: совпал заголовок чанка или любая крошка его пути."""
    if str(fields.get("chunk_level") or "child") != "child":
        return False
    path = fields.get("section_path") or []
    crumbs = [str(item) for item in path] if isinstance(path, list) else []
    return any(is_glossary_heading(text, headings) for text in [str(fields.get("heading") or ""), *crumbs])


def _cut(text: str, limit: int) -> str:
    return text if len(text) <= limit else f"{text[:limit].rstrip()}{ELLIPSIS}"


def _table_cells(line: str) -> list[str] | None:
    match = _TABLE_ROW_RE.match(line)
    if match is None or _TABLE_RULE_RE.match(line):
        return None
    return [cell.strip() for cell in match.group("cells").split("|")]


def _compound_word(term: str, definition: str, separator: str) -> bool:
    """Дефис внутри слова, а не тире перед определением: «информационно - справочные документы».

    Реальный корпус: обычный дефис служит и разделителем («ТН - транспортная накладная»), поэтому
    к нему требование: термин из нескольких слов, аббревиатура или определение с заглавной буквы."""
    return (
        separator.strip() == PLAIN_HYPHEN
        and len(term.split()) == 1
        and not _UPPER_PAIR_RE.search(term)
        and bool(_LOWER_START_RE.match(definition))
    )


def split_entry(line: str) -> tuple[str, str] | None:
    """Строка → (термин, определение) по первому тире, двоеточию или двум колонкам таблицы."""
    cells = _table_cells(line)
    if cells is not None:
        filled = [cell for cell in cells if cell]
        return (filled[0], filled[1]) if len(filled) >= 2 else None
    text = _MARKER_RE.sub("", line.strip())
    if not text:
        return None
    for pattern in (_DASH_RE, _COLON_RE):
        match = pattern.search(text)
        if match is None:
            continue
        term, definition = text[: match.start()], text[match.end() :]
        return None if _compound_word(term, definition, match.group(0)) else (term, definition)
    return None


def _balanced(term: str) -> str:
    """Незакрытая скобка — хвост соседнего оборота («приказ (распоряжение»): отбрасывается."""
    return term[: term.index("(")].strip() if term.count("(") > term.count(")") else term


def _valid(term: str, definition: str, settings: GlossarySettings) -> bool:
    if not term or not _LETTER_RE.search(term) or len(term.split()) > settings.term_max_words:
        return False
    return len(definition) >= settings.definition_min_chars and bool(_LETTER_RE.search(definition))


def parse_candidates(chunk: ChunkPayload, settings: GlossarySettings) -> list[Candidate]:
    """Кандидаты «термин — определение» из текста чанка; строки без разделителя пропускаются."""
    candidates: list[Candidate] = []
    for line in chunk.body.splitlines():
        parts = split_entry(line)
        if parts is None or parts[0].rstrip().endswith(","):
            # «Должностное лицо, ответственное за охрану труда, — назначается приказом»: слева придаточное
            # предложение пункта, а не термин
            continue
        term = _balanced(_EDGE_RE.sub("", parts[0]).strip())
        definition = _PUNCT_SPACE_RE.sub(r"\1", parts[1].strip()).rstrip(";,").strip()
        if normalize_term(term) in TABLE_HEADER_CELLS or not _valid(term, definition, settings):
            continue
        candidates.append(
            Candidate(term=term, definition=_cut(definition, settings.definition_max_chars), chunk=chunk)
        )
    return candidates


def _date(value: str | None) -> dt.date | None:
    try:
        return dt.date.fromisoformat(value[:10]) if value else None
    except ValueError:
        return None


def _record(candidate: Candidate) -> GlossaryRecord:
    chunk = candidate.chunk
    key = normalize_term(candidate.term)
    return GlossaryRecord(
        record_id=str(uuid.uuid5(GLOSSARY_NAMESPACE, f"{chunk.chunk_id}:{key}")),
        term=candidate.term,
        term_key=key,
        definition=candidate.definition,
        doc_id=chunk.doc_id,
        doc_label=document_label(chunk.doc_kind, chunk.doc_number, _date(chunk.doc_date)),
        doc_kind=chunk.doc_kind,
        doc_status=chunk.doc_status,
        section_id=chunk.parent_id or chunk.chunk_id,
        chunk_id=chunk.chunk_id,
        file_row_id=chunk.file_row_id,
        breadcrumbs=chunk.breadcrumbs,
    )


def _document_of(chunk: ChunkPayload) -> str:
    """Реквизиты документа для промпта подтверждения: корневая крошка или вид документа."""
    return document_label(chunk.doc_kind, chunk.doc_number, _date(chunk.doc_date))


def _heading_of(chunk: ChunkPayload, headings: Sequence[str]) -> str:
    """Заголовок раздела глоссария: сам заголовок чанка или подошедшая крошка пути."""
    for text in [chunk.heading or "", *reversed(chunk.section_path)]:
        if is_glossary_heading(text, headings):
            return text
    return chunk.heading or (chunk.section_path[-1] if chunk.section_path else "")


# ---------- подтверждение разделов ----------


def parse_verdict(text: str) -> SectionVerdict:
    """JSON подтверждения → решение по разделу; что не разбирается — раздел не подтверждён."""
    start = (text or "").find("{")
    if start < 0:
        return SectionVerdict(glossary=False, parsed=False)
    try:
        data, _ = _DECODER.raw_decode(text, start)
        if not isinstance(data, dict):
            raise TypeError("ожидался объект JSON")
        raw_reject = data.get("reject") or []
        if not isinstance(raw_reject, list):
            raise TypeError("reject должен быть списком")
        reject = {int(item) for item in raw_reject if str(item).strip().lstrip("-").isdigit()}
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        logger.warning("Глоссарий: ответ подтверждения не разобран (%s): %.200s", exc, text)
        return SectionVerdict(glossary=False, parsed=False)
    return SectionVerdict(
        glossary=bool(data.get("glossary", True)), reject=reject, reason=str(data.get("reason") or "")
    )


def confirm_user_message(section: Section) -> str:
    lines = "\n".join(
        f"{number}. {candidate.term} — {candidate.definition}"
        for number, candidate in enumerate(section.candidates, start=1)
    )
    return CONFIRM_USER_TEMPLATE.format(
        document=section.document, section=section.heading or "без заголовка", lines=lines
    )


class LlmTermConfirmer:
    """Подтверждение раздела Qwen3 (роль `glossary`): решение по разделу и номера лишних строк."""

    def __init__(self, llm: LLM) -> None:
        self._llm = llm

    async def confirm(self, section: Section) -> SectionVerdict:
        messages = [
            ChatMessage(role="system", content=CONFIRM_SYSTEM_PROMPT),
            ChatMessage(role="user", content=confirm_user_message(section)),
        ]
        try:
            response = await self._llm.achat(messages)
        except (OpenAIError, ValueError) as exc:
            logger.error("Глоссарий: подтверждение раздела «%s» не выполнено: %s", section.heading, exc)
            return SectionVerdict(glossary=False, parsed=False)
        return parse_verdict(response.message.content or "")


# ---------- сборка ----------


@dataclass
class GlossaryReport:
    sections: int = 0
    confirmed: int = 0
    rejected_sections: int = 0
    unconfirmed: int = 0
    candidates: int = 0
    rejected_entries: int = 0
    records: int = 0
    documents: int = 0
    removed: int = 0
    seconds: float = 0.0

    def summary_lines(self) -> list[str]:
        return [
            f"Глоссарий: разделов найдено {self.sections}, подтверждено {self.confirmed}, "
            f"отклонено моделью {self.rejected_sections}, без ответа модели {self.unconfirmed}",
            f"Кандидатов {self.candidates}, из них выброшено моделью {self.rejected_entries}",
            f"Записей в коллекции: {self.records} из {self.documents} документов "
            f"(удалено устаревших {self.removed}), {self.seconds:.0f} с",
        ]


class GlossaryBuilder:
    """Индекс документов → записи глоссария: разделы, кандидаты, подтверждение, запись в Qdrant."""

    def __init__(
        self,
        settings: GlossarySettings,
        *,
        source: ChunkSource,
        store: GlossaryStore,
        embedder: Embedder,
        confirmer: TermConfirmer,
    ) -> None:
        self._settings = settings
        self._source = source
        self._store = store
        self._embedder = embedder
        self._confirmer = confirmer

    def sections(self) -> list[Section]:
        """Разделы глоссария индекса: чанки с подходящим заголовком, сгруппированные по разделу файла."""
        headings = [normalize_heading(item) for item in self._settings.headings]
        ids = [
            chunk_id
            for chunk_id, fields in self._source.scroll_payload_fields(HEADING_FIELDS)
            if is_glossary_chunk(fields, headings)
        ]
        groups: dict[tuple[str, str], list[ChunkPayload]] = {}
        for chunk in self._source.get_chunks(ids):
            groups.setdefault((chunk.file_row_id, chunk.parent_id or chunk.chunk_id), []).append(chunk)
        sections: list[Section] = []
        for chunks in groups.values():
            chunks.sort(key=lambda chunk: chunk.chunk_index)
            candidates: list[Candidate] = []
            for chunk in chunks:
                candidates.extend(parse_candidates(chunk, self._settings))
            if not candidates:
                continue
            first = chunks[0]
            sections.append(
                Section(
                    document=_document_of(first),
                    heading=_heading_of(first, headings),
                    candidates=candidates[: self._settings.max_entries_per_section],
                )
            )
        return sections

    async def build(self) -> GlossaryReport:
        """Полная пересборка глоссария: подтверждение разделов, запись точек, уборка устаревших."""
        started = time.perf_counter()
        report = GlossaryReport()
        sections = self.sections()
        report.sections = len(sections)
        records: dict[str, GlossaryRecord] = {}
        for section in sections:
            report.candidates += len(section.candidates)
            verdict = await self._confirmer.confirm(section)
            if not verdict.parsed:
                report.unconfirmed += 1
                continue
            if not verdict.glossary:
                report.rejected_sections += 1
                logger.info("Глоссарий: раздел «%s» не подтверждён — %s", section.heading, verdict.reason)
                continue
            report.confirmed += 1
            for number, candidate in enumerate(section.candidates, start=1):
                if number in verdict.reject:
                    report.rejected_entries += 1
                    continue
                record = _record(candidate)
                records.setdefault(record.record_id, record)
        ordered = list(records.values())
        self._store.ensure_collection()
        if ordered:
            embeddings = self._embedder.encode([record.text for record in ordered])
            self._store.upsert(ordered, embeddings)
        report.removed = self._store.prune(set(records))
        report.records = len(ordered)
        report.documents = len({record.doc_id for record in ordered})
        report.seconds = time.perf_counter() - started
        return report
