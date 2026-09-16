"""Метаданные чанка из полей карточки СЭД (FR-3, §13.4).

Спецификация — `docs/chunk_metadata_mapping.md` (О6, на согласовании): каждая строка таблицы
«поле чанка → источник в карточке» здесь — явное присваивание. Метаданные не выводятся из текста
документа; из конвейера приходят только координаты в структуре (крошки, раздел, пункт, страница),
хэш файла и роль файла. `acl_groups` — заглушка `[]` (N8), поле обязательно в схеме с первого дня.
"""

from __future__ import annotations

import datetime as dt
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from common.config import DocStatus, StatusRuleSettings
from ingest.cards import APPROVAL_SECTION, INCOMING_SECTION, OUTGOING_SECTION, CardRecord
from ingest.chunking import Chunk
from ingest.corpus import CorpusDocument
from ingest.files import FilePlan
from ingest.parents import ParentChunk
from tessa_export.manifest import LinksGraph

ChunkLevel = Literal["child", "parent"]
ChunkKind = Literal["structural", "fallback", "table"]

APPROVERS_SECTION = "Approv"
RESPONSIBLE_SECTION = "ResponsibleErrand"
DIRECTION_SECTION = "DirectionActivityDCI"
NUMBER_PREFIX = "№"


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Relation(_Frozen):
    doc_id: str = Field(description="ID связанной карточки")
    relation: str | None = Field(description="Тип связи с точки зрения этой карточки")
    direction: Literal["outgoing", "incoming"]
    doc_type: str | None = Field(default=None, description="Тип связанного документа, если есть в секции")


class DocumentMetadata(_Frozen):
    """Поля документа (одинаковые у всех чанков карточки)."""

    doc_id: str
    tessa_card_id: str
    card_type_name: str | None
    card_type_caption: str | None
    doc_kind: str
    doc_number: str | None
    doc_date: dt.date | None
    doc_date_ts: int | None
    doc_status: DocStatus
    doc_status_name: str | None
    state_id: int | None
    state_name: str | None
    approval_state: str | None
    approval_state_name: str | None
    department: str | None
    department_id: str | None
    author: str | None
    relations: list[Relation]
    acl_groups: list[str] = Field(default_factory=list, description="Заглушка (N8): [] = не заполнено")
    subject: str | None
    comment: str | None
    signed_by: str | None
    direction_activity: list[str]
    approvers: list[str]
    responsible: list[str]
    validity_period: str | None
    card_version: int | None
    card_modified: dt.datetime | None

    @property
    def label(self) -> str:
        """Корневая крошка: «Приказ №144 от 15.01.2026» (вид, номер, дата — что есть)."""
        return document_label(self.doc_kind, self.doc_number, self.doc_date)


def document_label(doc_kind: str, doc_number: str | None, doc_date: dt.date | None) -> str:
    parts = [doc_kind]
    if doc_number:
        parts.append(f"{NUMBER_PREFIX}{doc_number}")
    if doc_date:
        parts.append(f"от {doc_date:%d.%m.%Y}")
    return " ".join(parts)


class FileMetadata(_Frozen):
    file_sha256: str
    file_name: str
    file_row_id: str
    file_category: str | None
    file_role: str
    parse_route: str
    also_in: list[str] = Field(default_factory=list, description="Карточки с тем же файлом (дубли)")


class ChunkPayload(_Frozen):
    """Payload точки Qdrant: документ + файл + координаты чанка. Ключи — как в маппинге."""

    doc_id: str
    tessa_card_id: str
    card_type_name: str | None
    card_type_caption: str | None
    doc_kind: str
    doc_number: str | None
    doc_date: str | None
    doc_date_ts: int | None
    doc_status: DocStatus
    doc_status_name: str | None
    state_id: int | None
    state_name: str | None
    approval_state: str | None
    approval_state_name: str | None
    department: str | None
    department_id: str | None
    author: str | None
    relations: list[Relation]
    acl_groups: list[str]
    subject: str | None
    comment: str | None
    signed_by: str | None
    direction_activity: list[str]
    approvers: list[str]
    responsible: list[str]
    validity_period: str | None
    card_version: int | None
    card_modified: str | None
    file_sha256: str
    file_name: str
    file_row_id: str
    file_category: str | None
    file_role: str
    parse_route: str
    also_in: list[str]
    chunk_id: str
    chunk_level: ChunkLevel
    chunk_kind: ChunkKind
    chunk_index: int
    parent_id: str | None
    child_ids: list[str]
    section_path: list[str]
    clause: str | None
    heading: str | None
    breadcrumbs: str
    page_no: int | None
    text: str
    body: str
    tokens: int


def _text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _date(value: Any) -> dt.date | None:
    if value is None:
        return None
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    text = str(value).strip()
    if not text:
        return None
    try:
        return dt.datetime.fromisoformat(text.replace("Z", "+00:00")).date()
    except ValueError:
        return None


def _names(card: CardRecord, section: str, column: str) -> list[str]:
    return [name for row in card.rows(section) if (name := _text(row.get(column)))]


def _relations(card: CardRecord, graph: LinksGraph) -> list[Relation]:
    card_id = str(card.id)
    relations = [
        Relation(
            doc_id=str(row["DocID"]),
            relation=_text(row.get("RefTypeName")),
            direction="outgoing",
            doc_type=_text(row.get("DocTypeName")),
        )
        for row in card.rows(OUTGOING_SECTION)
        if row.get("DocID")
    ]
    # тип входящей связи — обратное имя типа из карточки-источника (ребро графа экспорта)
    incoming_types = {
        str(edge.from_id): edge.reverse_type or edge.relation_type
        for edge in graph.edges
        if str(edge.to_id) == card_id and edge.source == "outgoing"
    }
    relations.extend(
        Relation(
            doc_id=str(row["DocID"]),
            relation=incoming_types.get(str(row["DocID"])),
            direction="incoming",
            doc_type=_text(row.get("DocTypeName")),
        )
        for row in card.rows(INCOMING_SECTION)
        if row.get("DocID")
    )
    return relations


def document_metadata(
    document: CorpusDocument, graph: LinksGraph, status: StatusRuleSettings
) -> DocumentMetadata:
    """Метаданные документа корпуса: карточка плюс вид из манифеста как последний запасной вариант."""
    return card_metadata(document.card, graph, status, fallback_kind=document.entry.doc_kind)


def card_metadata(
    card: CardRecord, graph: LinksGraph, status: StatusRuleSettings, *, fallback_kind: str = "Без вида"
) -> DocumentMetadata:
    """Строка за строкой по таблице маппинга (docs/chunk_metadata_mapping.md, §1–2).

    Используется инжестом (граф связей экспорта даёт типы входящих связей) и MCP-инструментом
    карточки (граф пустой: связи там берутся из сервиса карточек)."""
    common = card.common_text
    doc_date = _date(card.common_field("DocDate")) or _date(card.common_field("CreationDate"))
    state_id = _int(card.common_field("StateID"))
    approval = card.fields(APPROVAL_SECTION)
    approval_state = _text(approval.get("StateName")) or common("StateName")
    approval_state_id = _int(approval.get("StateID"))
    if approval_state_id is None and not _text(approval.get("StateName")):
        approval_state_id = state_id
    return DocumentMetadata(
        doc_id=str(card.id),
        tessa_card_id=str(card.id),
        card_type_name=card.type_name,
        card_type_caption=card.type_caption,
        doc_kind=common("DocTypeTitle") or card.type_caption or fallback_kind,
        doc_number=common("FullNumber") or common("SecondaryFullNumber"),
        doc_date=doc_date,
        doc_date_ts=int(dt.datetime(doc_date.year, doc_date.month, doc_date.day, tzinfo=dt.UTC).timestamp())
        if doc_date
        else None,
        doc_status=status.resolve(common("StatusID"), state_id),
        doc_status_name=common("StatusNameStatus"),
        state_id=state_id,
        state_name=common("StateName"),
        approval_state=approval_state,
        approval_state_name=status.state_names.get(approval_state_id, approval_state)
        if approval_state_id is not None
        else approval_state,
        department=common("DepartmentName"),
        department_id=common("DepartmentID"),
        author=common("AuthorName") or common("RegistratorName") or _text(card.created_by_name),
        relations=_relations(card, graph),
        acl_groups=[],
        subject=common("Subject"),
        comment=common("Comment"),
        signed_by=common("SignedByName"),
        direction_activity=_names(card, DIRECTION_SECTION, "DirectionActivityName"),
        approvers=_names(card, APPROVERS_SECTION, "UserName"),
        responsible=_names(card, RESPONSIBLE_SECTION, "UserName"),
        validity_period=common("ValidityPeriod"),
        card_version=_int(getattr(card, "version", None)),
        card_modified=card.modified,
    )


def file_metadata(plan: FilePlan, parse_route: str) -> FileMetadata:
    return FileMetadata(
        file_sha256=plan.file.sha256,
        file_name=plan.file.name,
        file_row_id=str(plan.file.row_id),
        file_category=plan.file.category,
        file_role=plan.role or "skipped",
        parse_route=parse_route,
        also_in=[str(card_id) for card_id in plan.also_in],
    )


def root_crumbs(document: DocumentMetadata, plan: FilePlan) -> tuple[str, ...]:
    """Корень крошек: основной файл — только документ; приложение — документ и имя файла."""
    return _file_root_crumbs(document, plan.role or "skipped", plan.file.name)


def _file_root_crumbs(document: DocumentMetadata, role: str, file_name: str) -> tuple[str, ...]:
    if role == "main":
        return (document.label,)
    return (document.label, f"Приложение «{file_name}»")


def _chunk_kind(chunk: Chunk) -> ChunkKind:
    if chunk.kind == "table":
        return "table"
    return "structural" if chunk.strategy == "structural" else "fallback"


def child_payload(
    document: DocumentMetadata, file: FileMetadata, chunk: Chunk, separator: str
) -> ChunkPayload:
    root_length = len(_file_root_crumbs(document, file.file_role, file.file_name))
    section_path = list(chunk.breadcrumbs[root_length:])
    if chunk.clause is not None and section_path and section_path[-1].endswith(chunk.clause):
        section_path = section_path[:-1]
    return ChunkPayload(
        **_document_fields(document),
        **file.model_dump(),
        chunk_id=chunk.chunk_id,
        chunk_level="child",
        chunk_kind=_chunk_kind(chunk),
        chunk_index=chunk.ordinal,
        parent_id=chunk.parent_id,
        child_ids=[],
        section_path=section_path,
        clause=chunk.clause,
        heading=chunk.heading,
        breadcrumbs=separator.join(chunk.breadcrumbs),
        page_no=chunk.page_no,
        text=chunk.text,
        body=chunk.body,
        tokens=chunk.tokens,
    )


def parent_payload(
    document: DocumentMetadata, file: FileMetadata, parent: ParentChunk, separator: str
) -> ChunkPayload:
    return ChunkPayload(
        **_document_fields(document),
        **file.model_dump(),
        chunk_id=parent.chunk_id,
        chunk_level="parent",
        chunk_kind="structural",
        chunk_index=parent.ordinal,
        parent_id=None,
        child_ids=list(parent.child_ids),
        section_path=list(
            parent.breadcrumbs[len(_file_root_crumbs(document, file.file_role, file.file_name)) :]
        ),
        clause=None,
        heading=parent.heading,
        breadcrumbs=separator.join(parent.breadcrumbs),
        page_no=parent.page_no,
        text=parent.text,
        body=parent.body,
        tokens=parent.tokens,
    )


def _document_fields(document: DocumentMetadata) -> dict[str, Any]:
    data = document.model_dump()
    data["doc_date"] = document.doc_date.isoformat() if document.doc_date else None
    data["card_modified"] = document.card_modified.isoformat() if document.card_modified else None
    return data
