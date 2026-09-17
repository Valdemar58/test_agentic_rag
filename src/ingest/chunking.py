"""Чанкинг DoclingDocument (FR-3): структурный основной, фиксированный фоллбэк, таблицы отдельно.

Структурный чанкер идёт по элементам DoclingDocument и ведёт «хлебные крошки»:
корневые крошки от вызывающего (например, «Приказ №144 от 15.01.2026» или «… → Приложение …»)
→ заголовки разделов (SectionHeader/Title; для dots.mocr уровень берётся из «#»/«##» в тексте)
→ нумерованные разделы («1. Утверждение» → «Раздел 1. Утверждение», «2.6 Гарантии» → «п. 2.6 Гарантии»)
→ пункт («3.2. …» → «п. 3.2»; номер берётся из маркера ListItem или из начала текста).
Номером пункта считается «1.», «1)», «1.1.» или «1.1» перед текстом; даты («04.09.2026»), суммы
(«26 702 руб.») и годы («2026 г.») номерами не считаются. Абзац из одного номера («1.6.») отдаёт
номер следующему абзацу. Крошки длиннее `breadcrumb_max_words` слов обрезаются.
Чанк уровня child — абзац/пункт: соседние абзацы с одинаковыми крошками склеиваются, пока чанк
короче `min_tokens`; абзац длиннее `max_tokens` режется по предложениям с перекрытием.
Таблицы — отдельные чанки в markdown с крошками своего раздела; длинная таблица режется по строкам
с повторением шапки, а строка шире лимита — в записи «колонка: значение». Документ без заголовков
и нумерации — фоллбэк: окна `max_tokens` с перекрытием `overlap_tokens` по предложениям.
Токены считаются токенайзером bge-m3.

Приложения (ручная проверка реального корпуса 2026-09-16: в 17 документах 69 приложений лежали внутри
последнего раздела): короткий абзац «Приложение № N» не в начале документа вместе с названием из
следующих коротких абзацев становится заголовком верхнего уровня («Приложение № 1. Перечень должностей…»);
нумерованный перечень внутри приложения («1. Мастер…», «2. Специалист…») — записи одного чанка, а не
пункты «п. 1», «п. 2». Строки оглавления (табуляция или отточие и номер страницы) пропускаются.

Перечень после двоеточия (живой диалог 2026-09-17: «…45 минут в следующем диапазоне:» и строки «начало
диапазона — 12:00; окончание — 15:00» оказались в разных чанках, и ответ терял условие): ненумерованные
абзацы после абзаца с двоеточием наследуют его пункт и остаются в том же чанке, пока записи кончаются
«;» или «,» либо идут элементами списка Docling; запись с точкой в конце закрывает перечень.

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
APPENDIX_PREFIX = "Приложение"
ELLIPSIS = "…"
TABLE_HEADER_LINES = 2
RECORD_SEPARATOR = ": "
SKIPPED_LABELS = frozenset({"page_header", "page_footer", "picture"})
HEADER_LABELS = frozenset({"section_header", "title"})
# «Приложение № 1», «Приложение 2», «Приложение N 1а», «Приложение № 3. Лист оповещения»
_APPENDIX_RE = re.compile(r"^\s*Приложение\s*(?:№|N|#)?\s*(\d+[а-яa-z]?)(?=[\s.:;)]|$)", re.IGNORECASE)
# строка оглавления: заголовок, табуляция или отточие, номер страницы («6.\tРежим…\t14», «Приложение № 1\t25»)
_TOC_RE = re.compile(r"(?:\t|\.{3,}|…)\s*\d{1,3}\s*$")
# абзац-привязка после «Приложение № N»: «к приказу…», «к Правилам…» — не название приложения
_ATTRIBUTION_RE = re.compile(r"^\s*к\s", re.IGNORECASE)

# «1.1. текст», «1.1 текст», «1. текст», «1) текст»; но не «26 702 руб.», не «04.09.2026», не «2026 г.»
_NUMBER_RE = re.compile(r"^\s*(?:(\d+(?:\.\d+)+)\.?|(\d+)[.)])(?=\s+\S)")
# заголовок Docling: допускается и «2 СРОКИ И УСЛОВИЯ» без точки после номера
_HEADER_NUMBER_RE = re.compile(r"^\s*(\d+(?:\.\d+)*)[.)]?(?=\s+\S)")
# абзац или маркер списка из одного номера: «1.6.», «1.6», «3.», «3)»
_BARE_NUMBER_RE = re.compile(r"^\s*(?:(\d+(?:\.\d+)+)\.?|(\d+)[.)])\s*$")
# компонент из четырёх и более цифр — год или дата («04.09.2026»), а не номер пункта;
# компонент с ведущим нулём — дата («07.09.26») или число («1.000»), номера пунктов так не пишут
_LONG_COMPONENT_RE = re.compile(r"\d{4,}|(?:^|\.)0\d")
LIST_OPENER = ":"
LIST_CONTINUATION = (":", ";", ",")
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
    parent_id: str | None = None


@dataclass
class _Unit:
    kind: Literal["header", "paragraph", "table"]
    text: str
    refs: list[str]
    page_no: int | None
    level: int = 0
    number: str | None = None
    appendix: bool = False
    list_item: bool = False


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
    list_mode: bool = False
    colon_list: bool = field(default=False, metadata={"doc": "В чанке есть абзац-вводка с двоеточием"})
    open_list: bool = field(default=False, metadata={"doc": "Последняя запись перечня не закрыта точкой"})

    def add(self, unit: _Unit, tokens: int) -> None:
        self.parts.append(unit.text)
        self.refs.extend(unit.refs)
        self.tokens += tokens
        tail = unit.text.rstrip()
        self.colon_list = self.colon_list or tail.endswith(LIST_OPENER)
        self.open_list = tail.endswith(LIST_CONTINUATION)

    def continues_list(self, unit: _Unit) -> bool:
        """Ненумерованный абзац после вводки с двоеточием — запись того же перечня."""
        return self.colon_list and unit.number is None and (self.open_list or unit.list_item)


@dataclass(frozen=True)
class _Level:
    """Открытый заголовок в стеке разделов."""

    level: int
    crumb: str
    text: str
    promoted: bool = field(metadata={"doc": "Нумерованный абзац-раздел, а не заголовок Docling"})
    appendix: bool = False


def _clean(text: str) -> str:
    return _SPACES_RE.sub(" ", text).strip()


def _label(item: Any) -> str:
    label = getattr(item, "label", "")
    return str(getattr(label, "value", label))


def _page(item: Any) -> int | None:
    prov = getattr(item, "prov", None)
    return int(prov[0].page_no) if prov else None


def _clause_number(match: re.Match[str] | None) -> str | None:
    if match is None:
        return None
    number = match.group(1) or match.group(2)
    return None if number is None or _LONG_COMPONENT_RE.search(number) else number


def number_of(text: str) -> str | None:
    """Номер пункта в начале текста («1.1. …» → «1.1»), иначе None."""
    return _clause_number(_NUMBER_RE.match(text))


def header_number_of(text: str) -> str | None:
    """Номер в заголовке: как у пункта, но допускается «2 СРОКИ И УСЛОВИЯ» без точки."""
    match = _HEADER_NUMBER_RE.match(text)
    if match is None or _LONG_COMPONENT_RE.search(match.group(1)):
        return None
    return match.group(1)


def bare_number(text: str) -> str | None:
    """Текст, состоящий из одного номера («1.6.» → «1.6»), иначе None."""
    return _clause_number(_BARE_NUMBER_RE.match(text))


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
        carried_number: str | None = None  # номер из абзаца вида «1.6.» без текста
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
            if _TOC_RE.search(raw):
                # строка оглавления: иначе «6. Режим… 14» — ложный раздел, «Приложение № 1 25» — приложение
                continue
            if label in HEADER_LABELS:
                hashes = _HASHES_RE.match(raw)
                header_level = len(hashes.group(1)) if hashes else int(getattr(item, "level", 1) or 1)
                text = _clean(_HASHES_RE.sub("", raw))
                units.append(
                    _Unit("header", text, [item.self_ref], _page(item), header_level, header_number_of(text))
                )
                continue
            text = _clean(raw)
            number = None
            marker = getattr(item, "marker", None)
            if label == "list_item" and getattr(item, "enumerated", False) and marker:
                number = bare_number(marker)
                if number is not None and number_of(text) is None:
                    text = f"{marker.strip()} {text}"
            if number is None:
                number = number_of(text)
            if number is None and (bare := bare_number(text)) is not None:
                carried_number = bare
                continue
            if number is None and carried_number is not None:
                number, text = carried_number, f"{carried_number}. {text}"
            carried_number = None
            units.append(
                _Unit(
                    "paragraph",
                    text,
                    [item.self_ref],
                    _page(item),
                    number=number,
                    list_item=label == "list_item",
                )
            )
        return units

    def _appendices(self, units: list[_Unit]) -> list[_Unit]:
        """Абзац «Приложение № N» (не первый в документе) с названием → заголовок приложения.

        Абзацы-привязки («к приказу…», «к Правилам…») остаются в тексте; названием считаются до двух
        следующих коротких ненумерованных абзацев (или остаток строки: «Приложение № 3. Лист оповещения»).
        Первый абзац документа не трогаем: «Приложение 1 к приказу…» в начале файла-вложения говорит о
        файле целиком, а не открывает раздел."""
        result: list[_Unit] = []
        index = 0
        while index < len(units):
            unit = units[index]
            match = _APPENDIX_RE.match(unit.text) if unit.kind == "paragraph" and index > 0 else None
            if match is None or len(unit.text.split()) > self._settings.appendix_max_words:
                result.append(unit)
                index += 1
                continue
            title_parts: list[str] = []
            rest = unit.text[match.end() :].strip(" .:;—-")
            if rest:
                title_parts.append(rest)
            refs = list(unit.refs)
            attributions: list[_Unit] = []
            index += 1
            while index < len(units):
                following = units[index]
                if following.kind != "paragraph" or following.number is not None:
                    break
                if _APPENDIX_RE.match(following.text):
                    break  # следующее приложение без текста («Форма…» — только таблица или картинка)
                if _ATTRIBUTION_RE.match(following.text) and not title_parts:
                    attributions.append(following)
                    index += 1
                    continue
                words = len(following.text.split())
                sentence = following.text.rstrip().endswith((".", ";", ":"))  # тело, а не название
                if words > self._settings.appendix_title_max_words or len(title_parts) >= 2 or sentence:
                    break
                title_parts.append(following.text)
                refs.extend(following.refs)
                index += 1
            title = " ".join(title_parts)
            text = f"{APPENDIX_PREFIX} № {match.group(1)}" + (f". {title}" if title else "")
            result.append(_Unit("header", text, refs, unit.page_no, appendix=True))
            result.extend(attributions)
        return result

    def _rest_after_number(self, text: str) -> str:
        return _HEADER_NUMBER_RE.sub("", text, count=1).strip()

    def _is_section_title(self, unit: _Unit) -> bool:
        """Короткий абзац с одним числом («1. Утверждение») — заголовок раздела, а не пункт."""
        if unit.number is None or "." in unit.number:
            return False
        rest = self._rest_after_number(unit.text)
        if not rest or rest.endswith((".", ";", ":")):
            return False
        return len(rest.split()) <= self._settings.section_title_max_words

    def _short(self, text: str) -> str:
        words = text.split()
        limit = self._settings.breadcrumb_max_words
        return text if len(words) <= limit else " ".join(words[:limit]) + ELLIPSIS

    def _header_crumb(self, unit: _Unit) -> str:
        """«3. Контроль» → «Раздел 3. Контроль»; «2.6 Гарантии» → «п. 2.6 Гарантии»; длинный → «п. 2.6»."""
        if unit.number is None:
            return self._short(unit.text)
        if "." not in unit.number:
            return self._short(f"{SECTION_PREFIX} {unit.text}")
        rest = self._rest_after_number(unit.text)
        if rest and len(rest.split()) <= self._settings.section_title_max_words:
            return f"{CLAUSE_PREFIX} {unit.number} {rest}"
        return f"{CLAUSE_PREFIX} {unit.number}"

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
        units = self._appendices(units)
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
        stack: list[_Level] = []
        draft: _Draft | None = None
        # приложения — на верхнем уровне заголовков документа (сиблинги разделов, а не их продолжение)
        top_level = min((unit.level for unit in units if unit.kind == "header" and unit.level > 0), default=1)

        def flush() -> None:
            nonlocal draft
            if draft is not None and draft.parts:
                chunks.extend(self._emit(draft, "text", "structural", file_sha256, len(chunks)))
            draft = None

        for unit in units:
            if unit.kind == "header":
                flush()
                promoted = unit.level == 0 and not unit.appendix
                if unit.appendix:
                    level = top_level
                elif promoted:
                    # нумерованные разделы («1.», «2.») — на уровень ниже последнего настоящего заголовка
                    real_levels = [entry.level for entry in stack if not entry.promoted]
                    level = (real_levels[-1] if real_levels else 0) + 1
                else:
                    level = unit.level
                while stack and stack[-1].level >= level:
                    stack.pop()
                stack.append(_Level(level, self._header_crumb(unit), unit.text, promoted, unit.appendix))
                continue
            section = tuple(item.crumb for item in stack)
            section_key = self._settings.breadcrumb_separator.join(section)
            heading = stack[-1].text if stack else None
            if unit.kind == "table":
                flush()
                table_draft = _Draft(
                    root_crumbs + section, section_key, heading, None, [unit.text], unit.refs, unit.page_no
                )
                chunks.extend(self._emit(table_draft, "table", "structural", file_sha256, len(chunks)))
                continue
            unit_tokens = self._count(unit.text)
            if draft is not None and draft.continues_list(unit):
                # записи перечня после «…в следующем диапазоне:» остаются в чанке вводки с её пунктом
                draft.add(unit, unit_tokens)
                continue
            clause = unit.number
            # внутри приложения «1. Мастер…», «2. Специалист…» — записи перечня, а не пункты документа
            list_entry = clause is not None and "." not in clause and any(item.appendix for item in stack)
            if list_entry:
                clause = None
            crumbs = root_crumbs + section + ((f"{CLAUSE_PREFIX} {clause}",) if clause else ())
            if draft is not None and (
                draft.breadcrumbs != crumbs
                or (draft.tokens >= self._settings.min_tokens and not (list_entry and draft.list_mode))
            ):
                flush()
            if draft is None:
                draft = _Draft(crumbs, section_key, heading, clause, page_no=unit.page_no)
            draft.add(unit, unit_tokens)
            draft.list_mode = draft.list_mode or list_entry
        flush()
        return chunks

    def _fixed_chunks(
        self, units: list[_Unit], root_crumbs: tuple[str, ...], title: str | None, file_sha256: str
    ) -> list[Chunk]:
        chunks: list[Chunk] = []
        crumbs = root_crumbs + ((self._short(title),) if title else ())
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
        bodies = self.split_table(body, budget) if kind == "table" else self.split_fixed(body, budget)
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

    def split_table(self, markdown: str, limit: int | None = None) -> list[str]:
        """Таблица по строкам с повторением шапки; строка шире лимита — записями «колонка: значение»."""
        limit = limit or self._settings.max_tokens
        if self._count(markdown) <= limit:
            return [markdown]
        lines = markdown.splitlines()
        header, rows = lines[:TABLE_HEADER_LINES], lines[TABLE_HEADER_LINES:]
        header_tokens = self._count("\n".join(header))
        names = _cells(header[0]) if header else []
        if header_tokens >= limit:
            # шапка сама шире лимита (гигантская объединённая ячейка): шапка — текстом, строки — записями
            header_parts = self.split_fixed(" | ".join(name for name in names if name), limit)
            header_parts.extend(part for row in rows for part in self._row_records(names, row, limit))
            return header_parts or [markdown]
        parts: list[str] = []
        current: list[str] = []
        current_tokens = header_tokens
        for row in rows:
            row_tokens = self._count(row)
            if header_tokens + row_tokens > limit:
                if current:
                    parts.append("\n".join(header + current))
                    current, current_tokens = [], header_tokens
                parts.extend(self._row_records(names, row, limit))
                continue
            if current and current_tokens + row_tokens > limit:
                parts.append("\n".join(header + current))
                current, current_tokens = [], header_tokens
            current.append(row)
            current_tokens += row_tokens
        if current:
            parts.append("\n".join(header + current))
        return parts or [markdown]

    def _row_records(self, names: list[str], row: str, limit: int) -> list[str]:
        """Широкая строка таблицы → чанки из пар «колонка: значение», каждый в пределах лимита."""
        cells = _cells(row)
        pairs: list[str] = []
        for index, cell in enumerate(cells):
            if not cell:
                continue
            name = names[index] if index < len(names) and names[index] else f"колонка {index + 1}"
            pairs.append(f"{name}{RECORD_SEPARATOR}{cell}")
        parts: list[str] = []
        current: list[str] = []
        current_tokens = 0
        for pair in pairs:
            for piece in self._split_long(pair, limit):
                piece_tokens = self._count(piece)
                if current and current_tokens + piece_tokens > limit:
                    parts.append("\n".join(current))
                    current, current_tokens = [], 0
                current.append(piece)
                current_tokens += piece_tokens
        if current:
            parts.append("\n".join(current))
        return parts


def _cells(line: str) -> list[str]:
    stripped = line.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    return [cell.strip().strip("*").strip() for cell in stripped.split("|")]


def _clean_table(markdown: str) -> str:
    lines = [line.rstrip() for line in markdown.splitlines() if line.strip()]
    return "\n".join(lines)
