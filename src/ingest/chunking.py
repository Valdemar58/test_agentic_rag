"""Чанкинг DoclingDocument (FR-3): структурный основной, фиксированный фоллбэк, таблицы отдельно.

Структурный чанкер идёт по элементам DoclingDocument и ведёт «хлебные крошки»:
корневые крошки от вызывающего (например, «Приказ №144 от 15.01.2026» или «… → Приложение …»)
→ заголовки разделов (SectionHeader/Title; для dots.mocr уровень берётся из «#»/«##» в тексте)
→ нумерованные разделы («1. Утверждение» → «Раздел 1. Утверждение»)
→ пункт («3.2. …» → «п. 3.2»; номер берётся из маркера ListItem или из начала текста).
Чанк уровня child — абзац/пункт: соседние абзацы с одинаковыми крошками склеиваются, пока чанк
короче `min_tokens`; абзац длиннее `max_tokens` режется по предложениям с перекрытием.
Таблицы — отдельные чанки в markdown с крошками своего раздела; длинная таблица режется по строкам
с повторением шапки. Документ без заголовков и нумерации — фоллбэк: окна `max_tokens` с
перекрытием `overlap_tokens` по предложениям. Токены считаются токенайзером bge-m3.

`section_key` чанка — путь разделов без пункта: по нему собирается parent (задача 4.4).
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal

from common.config import ChunkingSettings
from ingest.tokens import TokenCounter

CHUNK_NAMESPACE = uuid.UUID("6f1c4a4e-2d5b-4d7e-9a3c-7b8e1f2a3c4d")
ChunkKind = Literal["text", "table"]
Strategy = Literal["structural", "fixed"]
SECTION_PREFIX = "Раздел"
CLAUSE_PREFIX = "п."
TABLE_HEADER_LINES = 2
SKIPPED_LABELS = frozenset({"page_header", "page_footer", "picture"})
HEADER_LABELS = frozenset({"section_header", "title"})

_NUMBER_RE = re.compile(r"^\s*(\d+(?:\.\d+)*)[.)]?(?=\s|$)")
_HASHES_RE = re.compile(r"^\s*(#+)\s*")
_SPACES_RE = re.compile(r"[ \t ]+")
_SENTENCE_RE = re.compile(r"(?<=[.!?;])\s+(?=[«\"(A-ZА-ЯЁ0-9])|\n+")


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    ordinal: int
    kind: ChunkKind
    strategy: Strategy
    breadcrumbs: tuple[str, ...]
    section_key: str
    heading: str | None
    clause: str | None
    body: str
    text: str
    tokens: int
    page_no: int | None
    item_refs: tuple[str, ...]


@dataclass
class _Unit:
    kind: Literal["header", "paragraph", "table"]
    text: str
    refs: list[str]
    page_no: int | None
    level: int = 0
    number: str | None = None


@dataclass
class _Draft:
    breadcrumbs: tuple[str, ...]
    section_key: str
    heading: str | None
    clause: str | None
    parts: list[str] = field(default_factory=list)
    refs: list[str] = field(default_factory=list)
    page_no: int | None = None
    tokens: int = 0


def _clean(text: str) -> str:
    return _SPACES_RE.sub(" ", text).strip()


def _label(item: Any) -> str:
    label = getattr(item, "label", "")
    return str(getattr(label, "value", label))


def _page(item: Any) -> int | None:
    prov = getattr(item, "prov", None)
    return int(prov[0].page_no) if prov else None


def _number_of(text: str) -> str | None:
    match = _NUMBER_RE.match(text)
    return match.group(1) if match else None


def sentences(text: str) -> list[str]:
    return [part.strip() for part in _SENTENCE_RE.split(text) if part and part.strip()]


class StructuralChunker:
    def __init__(self, settings: ChunkingSettings, counter: TokenCounter) -> None:
        self._settings = settings
        self._count = counter.count

    # ---------- извлечение единиц из DoclingDocument ----------

    def _units(self, document: Any) -> list[_Unit]:
        units: list[_Unit] = []
        skip_deeper_than: int | None = None
        for item, level in document.iterate_items(with_groups=True):
            if skip_deeper_than is not None:
                if level > skip_deeper_than:
                    continue
                skip_deeper_than = None
            label = _label(item)
            if label == "table":
                markdown = _clean_table(item.export_to_markdown(document))
                if markdown:
                    units.append(_Unit("table", markdown, [item.self_ref], _page(item)))
                skip_deeper_than = level
                continue
            if label in SKIPPED_LABELS:
                skip_deeper_than = level
                continue
            raw = getattr(item, "text", None)
            if not raw or not raw.strip():
                continue
            if label in HEADER_LABELS:
                hashes = _HASHES_RE.match(raw)
                header_level = len(hashes.group(1)) if hashes else int(getattr(item, "level", 1) or 1)
                text = _clean(_HASHES_RE.sub("", raw))
                units.append(
                    _Unit("header", text, [item.self_ref], _page(item), header_level, _number_of(text))
                )
                continue
            text = _clean(raw)
            number = None
            marker = getattr(item, "marker", None)
            if label == "list_item" and getattr(item, "enumerated", False) and marker:
                number = _number_of(marker)
                if number is not None and _number_of(text) is None:
                    text = f"{marker.strip()} {text}"
            if number is None:
                number = _number_of(text)
            units.append(_Unit("paragraph", text, [item.self_ref], _page(item), number=number))
        return units

    def _is_section_title(self, unit: _Unit) -> bool:
        """Короткий абзац с одним числом («1. Утверждение») — заголовок раздела, а не пункт."""
        if unit.number is None or "." in unit.number:
            return False
        rest = _NUMBER_RE.sub("", unit.text, count=1).strip()
        if not rest or rest.endswith((".", ";", ":")):
            return False
        return len(rest.split()) <= self._settings.section_title_max_words

    # ---------- сборка чанков ----------

    def chunk(self, document: Any, root_crumbs: tuple[str, ...], *, file_sha256: str) -> list[Chunk]:
        units = self._units(document)
        for unit in units:
            if unit.kind == "paragraph" and self._is_section_title(unit):
                unit.kind = "header"
        # заголовок документа — единственный заголовок Docling верхнего уровня в самом начале
        real_levels = [unit.level for unit in units if unit.kind == "header" and unit.level > 0]
        title: str | None = None
        if (
            real_levels
            and units[0].kind == "header"
            and units[0].level == min(real_levels)
            and real_levels.count(min(real_levels)) == 1
        ):
            title = units[0].text
            units = units[1:]
        structured = any(unit.kind == "header" for unit in units) or any(
            unit.kind == "paragraph" and unit.number for unit in units
        )
        if not structured:
            return self._fixed_chunks(units, root_crumbs, title, file_sha256)
        return self._structural_chunks(units, root_crumbs, file_sha256)

    def _structural_chunks(
        self, units: list[_Unit], root_crumbs: tuple[str, ...], file_sha256: str
    ) -> list[Chunk]:
        chunks: list[Chunk] = []
        # (уровень, крошка, текст заголовка, это нумерованный абзац-раздел, а не заголовок Docling)
        stack: list[tuple[int, str, str, bool]] = []
        draft: _Draft | None = None

        def flush() -> None:
            nonlocal draft
            if draft is not None and draft.parts:
                chunks.extend(self._emit(draft, "text", "structural", file_sha256, len(chunks)))
            draft = None

        for unit in units:
            if unit.kind == "header":
                flush()
                promoted = unit.level == 0
                if promoted:
                    # нумерованные разделы («1.», «2.») — на уровень ниже последнего настоящего заголовка
                    real_levels = [entry[0] for entry in stack if not entry[3]]
                    level = (real_levels[-1] if real_levels else 0) + 1
                else:
                    level = unit.level
                while stack and stack[-1][0] >= level:
                    stack.pop()
                crumb = f"{SECTION_PREFIX} {unit.text}" if unit.number else unit.text
                stack.append((level, crumb, unit.text, promoted))
                continue
            section = tuple(item[1] for item in stack)
            section_key = self._settings.breadcrumb_separator.join(section)
            heading = stack[-1][2] if stack else None
            if unit.kind == "table":
                flush()
                table_draft = _Draft(
                    root_crumbs + section, section_key, heading, None, [unit.text], unit.refs, unit.page_no
                )
                chunks.extend(self._emit(table_draft, "table", "structural", file_sha256, len(chunks)))
                continue
            clause = unit.number
            crumbs = root_crumbs + section + ((f"{CLAUSE_PREFIX} {clause}",) if clause else ())
            unit_tokens = self._count(unit.text)
            if draft is not None and (
                draft.breadcrumbs != crumbs or draft.tokens >= self._settings.min_tokens
            ):
                flush()
            if draft is None:
                draft = _Draft(crumbs, section_key, heading, clause, page_no=unit.page_no)
            draft.parts.append(unit.text)
            draft.refs.extend(unit.refs)
            draft.tokens += unit_tokens
        flush()
        return chunks

    def _fixed_chunks(
        self, units: list[_Unit], root_crumbs: tuple[str, ...], title: str | None, file_sha256: str
    ) -> list[Chunk]:
        chunks: list[Chunk] = []
        crumbs = root_crumbs + ((title,) if title else ())
        text_units = [unit for unit in units if unit.kind != "table"]
        if text_units:
            draft = _Draft(crumbs, "", title, None, page_no=text_units[0].page_no)
            draft.parts = [unit.text for unit in text_units]
            draft.refs = [ref for unit in text_units for ref in unit.refs]
            chunks.extend(self._emit(draft, "text", "fixed", file_sha256, 0))
        for unit in units:
            if unit.kind == "table":
                table_draft = _Draft(crumbs, "", title, None, [unit.text], unit.refs, unit.page_no)
                chunks.extend(self._emit(table_draft, "table", "fixed", file_sha256, len(chunks)))
        return chunks

    # ---------- нарезка по токенам ----------

    def _emit(
        self, draft: _Draft, kind: ChunkKind, strategy: Strategy, sha256: str, start: int
    ) -> list[Chunk]:
        prefix = self._settings.breadcrumb_separator.join(draft.breadcrumbs)
        budget = max(self._settings.max_tokens - self._count(prefix), self._settings.overlap_tokens + 1)
        body = "\n".join(draft.parts)
        if kind == "table":
            bodies = self._split_table(body, budget)
        else:
            bodies = self.split_fixed(body, budget)
        chunks: list[Chunk] = []
        for offset, part in enumerate(bodies):
            ordinal = start + offset
            text = f"{prefix}\n{part}" if prefix else part
            chunks.append(
                Chunk(
                    chunk_id=str(uuid.uuid5(CHUNK_NAMESPACE, f"{sha256}:{ordinal}")),
                    ordinal=ordinal,
                    kind=kind,
                    strategy=strategy,
                    breadcrumbs=draft.breadcrumbs,
                    section_key=draft.section_key,
                    heading=draft.heading,
                    clause=draft.clause,
                    body=part,
                    text=text,
                    tokens=self._count(text),
                    page_no=draft.page_no,
                    item_refs=tuple(draft.refs),
                )
            )
        return chunks

    def split_fixed(self, text: str, max_tokens: int | None = None) -> list[str]:
        """Окна не длиннее max_tokens по предложениям с перекрытием не меньше overlap_tokens."""
        limit = max_tokens or self._settings.max_tokens
        if self._count(text) <= limit:
            return [text]
        overlap = min(self._settings.overlap_tokens, limit - 1)
        pieces: list[str] = []
        for sentence in sentences(text):
            pieces.extend(self._split_long(sentence, limit))
        windows: list[str] = []
        window: list[str] = []
        window_tokens = 0
        for piece in pieces:
            piece_tokens = self._count(piece)
            if window and window_tokens + piece_tokens > limit:
                windows.append(" ".join(window))
                kept: list[str] = []
                kept_tokens = 0
                for previous in reversed(window):
                    previous_tokens = self._count(previous)
                    if kept_tokens >= overlap or kept_tokens + previous_tokens + piece_tokens > limit:
                        break
                    kept.insert(0, previous)
                    kept_tokens += previous_tokens
                window, window_tokens = kept, kept_tokens
            window.append(piece)
            window_tokens += piece_tokens
        if window:
            windows.append(" ".join(window))
        return windows or [text]

    def _split_long(self, sentence: str, limit: int) -> list[str]:
        if self._count(sentence) <= limit:
            return [sentence]
        words = sentence.split()
        pieces: list[str] = []
        current: list[str] = []
        for word in words:
            current.append(word)
            if self._count(" ".join(current)) > limit and len(current) > 1:
                current.pop()
                pieces.append(" ".join(current))
                current = [word]
        if current:
            pieces.append(" ".join(current))
        return pieces

    def _split_table(self, markdown: str, limit: int) -> list[str]:
        """Длинная таблица режется по строкам; шапка (первые строки markdown) повторяется."""
        if self._count(markdown) <= limit:
            return [markdown]
        lines = markdown.splitlines()
        header, rows = lines[:TABLE_HEADER_LINES], lines[TABLE_HEADER_LINES:]
        header_tokens = self._count("\n".join(header))
        parts: list[str] = []
        current: list[str] = []
        current_tokens = header_tokens
        for row in rows:
            row_tokens = self._count(row)
            if current and current_tokens + row_tokens > limit:
                parts.append("\n".join(header + current))
                current, current_tokens = [], header_tokens
            current.append(row)
            current_tokens += row_tokens
        if current:
            parts.append("\n".join(header + current))
        return parts or [markdown]


def _clean_table(markdown: str) -> str:
    lines = [line.rstrip() for line in markdown.splitlines() if line.strip()]
    return "\n".join(lines)
