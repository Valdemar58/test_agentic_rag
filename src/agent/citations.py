"""Машиночитаемые цитаты (6.5, FR-4): маркеры [S#]/[D#] в ответе → источники с chunk_id/doc_id.

Модель ставит ссылки псевдонимами реестра; здесь они переводятся в идентификаторы индекса и СЭД и
нумеруются по порядку появления: в тексте остаются [1], [2], в конце — блок «Источники» с реквизитами
документа, разделом/пунктом и страницей. Неизвестные маркеры удаляются и попадают в `unresolved`, свой
блок «Источники» модели отбрасывается — так каждая ссылка ответа резолвится в реальный фрагмент или
карточку (M3), а блок источников строится детерминированно и одинаково для UI и eval.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field

from agent.evidence import ALIAS_RE, DOC_PREFIX, EvidenceRegistry, Fragment, KnownDocument

SOURCES_TITLE = "Источники"
CARD_SOURCE = "карточка документа"
KNOWN_DOCUMENT_SOURCE = "документ СЭД (карточка не запрашивалась)"
# группа ссылок в скобках: [S1], [S1][S4], [S1, D2], [ S3; S4 ]
MARKER_GROUP_RE = re.compile(r"\[\s*((?:[DS]\d+)(?:\s*[,;/ ]\s*[DS]\d+)*)\s*\]")
ALIAS_IN_GROUP_RE = re.compile(r"[DS]\d+")
# псевдоним документа в прозе («документ D1 действует») — заменяется на реквизиты
BARE_DOC_RE = re.compile(r"(?<![\[\w])(D\d+)(?![\]\w])")
# блок «Источники», который модель иногда дописывает сама, — до конца текста
MODEL_SOURCES_RE = re.compile(r"\n[ \t]*(?:\*\*|#+\s*)?Источники\s*:?(?:\*\*)?[ \t]*\n.*\Z", re.DOTALL)
SourceKind = Literal["fragment", "document"]


class Source(BaseModel):
    number: int = Field(description="Номер в тексте ответа: [1], [2]…")
    alias: str = Field(description="Псевдоним реестра (S3, D1)")
    kind: SourceKind
    doc_id: str
    chunk_id: str | None = Field(description="ID чанка индекса для фрагмента; у карточки нет")
    parent_id: str | None = None
    label: str = Field(description="Документ: вид, номер, дата")
    doc_status: str | None
    breadcrumbs: str | None = Field(description="Путь к фрагменту: документ → раздел → пункт")
    clause: str | None = None
    page_no: int | None = None
    text: str | None = Field(description="Текст фрагмента или сводка карточки — для показа по клику")
    context: str | None = Field(default=None, description="Текст раздела-родителя фрагмента, если есть")

    def line(self) -> str:
        """Строка блока «Источники» (FR-4: название/номер, дата, раздел/пункт)."""
        status = f" ({self.doc_status})" if self.doc_status else ""
        if self.kind == "document":
            origin = CARD_SOURCE if self.text else KNOWN_DOCUMENT_SOURCE
            return f"[{self.number}] {self.label}{status} — {origin}"
        where = self.breadcrumbs or self.label
        page = f", стр. {self.page_no}" if self.page_no else ""
        return f"[{self.number}] {where}{status}{page}"


class CitedAnswer(BaseModel):
    text: str = Field(description="Ответ с номерами ссылок и блоком «Источники»")
    body: str = Field(description="Ответ без блока «Источники»")
    sources: list[Source]
    unresolved: list[str] = Field(description="Маркеры, которых нет в реестре (удалены из текста)")

    @property
    def sources_block(self) -> str:
        if not self.sources:
            return ""
        return "\n".join([f"{SOURCES_TITLE}:", *(source.line() for source in self.sources)])


def _fragment_source(number: int, fragment: Fragment, document: KnownDocument | None) -> Source:
    return Source(
        number=number,
        alias=fragment.alias,
        kind="fragment",
        doc_id=fragment.doc_id,
        chunk_id=fragment.chunk_id,
        parent_id=fragment.parent_id,
        label=document.label if document else "",
        doc_status=document.status if document else None,
        breadcrumbs=fragment.breadcrumbs,
        clause=fragment.clause,
        page_no=fragment.page_no,
        text=fragment.text,
        context=fragment.context,
    )


def _document_source(number: int, document: KnownDocument) -> Source:
    return Source(
        number=number,
        alias=document.alias,
        kind="document",
        doc_id=document.doc_id,
        chunk_id=None,
        label=document.label,
        doc_status=document.status,
        breadcrumbs=None,
        text=document.card_text,
    )


def strip_model_sources(text: str) -> str:
    """Убирает блок «Источники», дописанный моделью: он строится здесь детерминированно."""
    return MODEL_SOURCES_RE.sub("", text).rstrip()


def cite_answer(text: str, registry: EvidenceRegistry) -> CitedAnswer:
    body = strip_model_sources(text.strip())
    sources: list[Source] = []
    numbers: dict[str, int] = {}
    unresolved: list[str] = []

    def number_for(alias: str) -> int | None:
        if alias in numbers:
            return numbers[alias]
        source: Source | None = None
        if alias.startswith(DOC_PREFIX):
            document = registry.document_by_alias(alias)
            if document is not None:
                source = _document_source(len(sources) + 1, document)
        else:
            fragment = registry.fragment_by_alias(alias)
            if fragment is not None:
                source = _fragment_source(len(sources) + 1, fragment, registry.document(fragment.doc_id))
        if source is None:
            if alias not in unresolved:
                unresolved.append(alias)
            return None
        sources.append(source)
        numbers[alias] = source.number
        return source.number

    def replace_group(match: re.Match[str]) -> str:
        rendered = []
        for alias in ALIAS_IN_GROUP_RE.findall(match.group(1)):
            number = number_for(alias)
            if number is not None:
                rendered.append(f"[{number}]")
        return "".join(rendered)

    body = MARKER_GROUP_RE.sub(replace_group, body)

    def replace_bare_document(match: re.Match[str]) -> str:
        alias = match.group(1)
        if not ALIAS_RE.match(alias):
            return alias
        document = registry.document_by_alias(alias)
        return document.label if document is not None and document.label else alias

    body = BARE_DOC_RE.sub(replace_bare_document, body)
    body = re.sub(r"[ \t]+\n", "\n", body).strip()
    cited = CitedAnswer(text=body, body=body, sources=sources, unresolved=unresolved)
    if sources:
        cited.text = f"{body}\n\n{cited.sources_block}"
    return cited
