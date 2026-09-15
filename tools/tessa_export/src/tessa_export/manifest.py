"""Манифест экспорта и граф связей (§8.1.5 ТЗ).

manifest.json: для каждого документа — все пути входа (seed / по связи, тип, глубина), вид карточки,
статус, файлы с форматом, признаком текстового слоя, размерами и хэшами; исключённые документы,
ошибки, непройденные связи и сводная статистика. links_graph.json: рёбра from_id, to_id,
relation_type только между документами сета; висячие рёбра — отдельно.
"""

from __future__ import annotations

import re
from collections import Counter
from datetime import UTC, date, datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from tessa_export import __name__ as _package_name
from tessa_export.config import CoverageSettings, ExportConfig, StatusSettings
from tessa_export.files import FileRecord
from tessa_export.models import CardSnapshot
from tessa_export.storage import CARDS_DIR, CARDS_RAW_DIR
from tessa_export.walker import EntryPath, LinkEdge, WalkResult

MANIFEST_VERSION = "2"
APPROVAL_SECTION = "KrApprovalCommonInfoVirtual"
DocStatus = Literal["active", "cancelled", "draft"]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EntryPathEntry(_Model):
    kind: Literal["seed", "link"]
    depth: int
    via_card_id: UUID | None = None
    relation: str | None = None
    direction: Literal["outgoing", "incoming"] | None = None


class FileEntry(_Model):
    row_id: UUID
    name: str
    extension: str
    category: str | None
    version_number: int
    is_virtual: bool
    downloaded: bool
    path: str | None = Field(default=None, description="Путь внутри архива")
    sha256: str | None = None
    size: int | None = None
    content_type: str | None = None
    skipped_reason: str | None = None
    skipped_detail: str = ""
    smoke_ok: bool | None = None
    smoke_error: str | None = None
    smoke_note: str | None = Field(
        default=None, description="Файл открыт, но с оговоркой (например, python-docx его не разбирает)"
    )
    has_text_layer: bool | None = None
    page_count: int | None = None
    has_tables: bool | None = None
    has_terms_section: bool | None = None
    duplicate_of: str | None = Field(default=None, description="card_id/row_id первого файла с тем же sha256")


class DocumentEntry(_Model):
    card_id: UUID
    type_name: str | None
    type_caption: str | None
    doc_type_title: str | None
    doc_kind: str = Field(description="Категория Тессы: DocTypeTitle, иначе TypeCaption")
    coverage_kinds: list[str] = Field(description="Категории 8.2 по содержанию (coverage_kind_map)")
    number: str | None
    doc_date: date | None
    subject: str | None
    department: str | None
    status_id: str | None
    status_name: str | None
    state_id: int | None
    doc_status: DocStatus = Field(description="Статус по правилу status конфига: active/cancelled/draft")
    is_cancelled: bool
    state_name: str | None
    approval_state: str | None
    depth: int
    entry_paths: list[EntryPathEntry]
    card_path: str
    card_raw_path: str
    files: list[FileEntry]


class ExcludedEntry(_Model):
    card_id: UUID
    reason: str
    depth: int
    type_name: str | None
    doc_type_title: str | None
    entry_paths: list[EntryPathEntry]


class ErrorEntry(_Model):
    card_id: UUID
    depth: int
    kind: str
    message: str
    entry_paths: list[EntryPathEntry]


class SkippedLinkEntry(_Model):
    from_card_id: UUID | None
    to_card_id: UUID
    direction: str | None
    relation: str | None
    reason: str
    detail: str


class ManifestStats(_Model):
    documents: int
    excluded: int
    errors: int
    skipped_links: int
    files_total: int
    files_downloaded: int
    files_skipped: int
    skipped_by_reason: dict[str, int]
    skipped_by_extension: dict[str, int]
    duplicate_files: int
    doc_kinds: dict[str, int] = Field(description="Категория Тессы → документов")
    coverage_kinds: dict[str, int] = Field(description="Категория 8.2 → документов (с пересечениями)")
    doc_statuses: dict[str, int] = Field(description="Статус документа → документов")
    card_types: dict[str, int]
    status_values: dict[str, str | None] = Field(description="StatusID → StatusNameStatus, все встреченные")
    relation_types: dict[str, str | None] = Field(description="RefTypeName → RefTypeReverseName")


class Manifest(_Model):
    version: str = MANIFEST_VERSION
    created_at: datetime
    tool: str
    source: str = Field(default="tessa", description="tessa — онлайн-экспорт; иначе офлайн-фильтр архива")
    synthetic: bool = Field(default=False, description="True для режима самопроверки: данные синтетические")
    seed_ids: list[UUID]
    traversal: dict[str, Any]
    allowed_extensions: list[str]
    exclude_rules: int
    documents: list[DocumentEntry]
    excluded: list[ExcludedEntry]
    errors: list[ErrorEntry]
    skipped_links: list[SkippedLinkEntry]
    stats: ManifestStats


class LinkEdgeEntry(_Model):
    from_id: UUID
    to_id: UUID
    relation_type: str | None
    reverse_type: str | None
    relation_type_id: UUID | None
    source: Literal["outgoing", "incoming"]


class LinksGraph(_Model):
    version: str = MANIFEST_VERSION
    edges: list[LinkEdgeEntry] = Field(description="Рёбра между документами сета")
    dangling_edges: list[LinkEdgeEntry] = Field(description="Рёбра к документам вне сета")


def classify_coverage_kinds(
    doc_type_title: str | None, type_caption: str | None, subject: str | None, coverage: CoverageSettings
) -> list[str]:
    """Категории 8.2 по регэкспам конфига над видом документа, типом карточки и темой.

    Категории содержательные (решение заказчика 2026-09-14): приказ «Об утверждении Инструкции…»
    попадает и в «Приказы», и в «Инструкции». Без совпадений — other_kind_label."""
    texts = [text for text in (doc_type_title, type_caption, subject) if text]
    kinds = [
        kind
        for kind, patterns in coverage.coverage_kind_map.items()
        if any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns for text in texts)
    ]
    return kinds or [coverage.other_kind_label]


def resolve_doc_status(status_id: str | None, state_id: int | None, status: StatusSettings) -> DocStatus:
    """Статус документа по StatusID и StateID (правило и таблицы состояний в конфиге, StatusSettings)."""
    if status_id and status_id.lower() in {str(item) for item in status.cancelled_status_ids}:
        return "cancelled"
    if state_id in status.cancelled_state_ids:
        return "cancelled"
    if state_id in status.active_state_ids:
        return "active"
    return "draft"


def _to_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _to_date(value: Any) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value).strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date()
    except ValueError:
        return None


def _entry_paths(paths: list[EntryPath]) -> list[EntryPathEntry]:
    return [
        EntryPathEntry(
            kind=path.kind,
            depth=path.depth,
            via_card_id=path.via_card_id,
            relation=path.relation,
            direction=path.direction,
        )
        for path in paths
    ]


def _file_entry(record: FileRecord) -> FileEntry:
    return FileEntry(
        row_id=record.file.row_id,
        name=record.file.name,
        extension=record.file.extension,
        category=record.file.category,
        version_number=record.file.version_number,
        is_virtual=record.file.is_virtual,
        downloaded=record.downloaded,
        path=record.relative_path,
        sha256=record.sha256,
        size=record.size,
        content_type=record.content_type,
        skipped_reason=record.skipped_reason,
        skipped_detail=record.skipped_detail,
        smoke_ok=record.smoke_ok,
        smoke_error=record.smoke_error,
        smoke_note=record.smoke_note,
        has_text_layer=record.has_text_layer,
        page_count=record.page_count,
        has_tables=record.has_tables,
        has_terms_section=record.has_terms_section,
    )


def document_entry(
    snapshot: CardSnapshot,
    depth: int,
    entry_paths: list[EntryPath],
    records: list[FileRecord],
    coverage: CoverageSettings,
    status: StatusSettings,
) -> DocumentEntry:
    status_id = snapshot.common_text("StatusID")
    state_id = _to_int(snapshot.common_field("StateID"))
    doc_type_title = snapshot.common_text("DocTypeTitle")
    subject = snapshot.common_text("Subject")
    doc_status = resolve_doc_status(status_id, state_id, status)
    approval = snapshot.sections.get(APPROVAL_SECTION)
    approval_state = None
    if approval is not None and approval.fields:
        raw_state = approval.fields.get("StateName")
        approval_state = str(raw_state) if raw_state else None
    return DocumentEntry(
        card_id=snapshot.card_id,
        type_name=snapshot.type_name,
        type_caption=snapshot.type_caption,
        doc_type_title=doc_type_title,
        doc_kind=doc_type_title or snapshot.type_caption or "Без вида",
        coverage_kinds=classify_coverage_kinds(doc_type_title, snapshot.type_caption, subject, coverage),
        number=snapshot.common_text("FullNumber") or snapshot.common_text("SecondaryFullNumber"),
        doc_date=_to_date(snapshot.common_field("DocDate"))
        or _to_date(snapshot.common_field("CreationDate")),
        subject=subject,
        department=snapshot.common_text("DepartmentName"),
        status_id=status_id,
        status_name=snapshot.common_text("StatusNameStatus"),
        state_id=state_id,
        doc_status=doc_status,
        is_cancelled=doc_status == "cancelled",
        state_name=snapshot.common_text("StateName"),
        approval_state=approval_state,
        depth=depth,
        entry_paths=_entry_paths(entry_paths),
        card_path=f"{CARDS_DIR}/{snapshot.card_id}.json",
        card_raw_path=f"{CARDS_RAW_DIR}/{snapshot.card_id}.json",
        files=[_file_entry(record) for record in records],
    )


def mark_duplicates(documents: list[DocumentEntry]) -> int:
    """Помечает файлы с повторяющимся sha256 (в порядке документов); возвращает число дублей."""
    first_seen: dict[str, str] = {}
    duplicates = 0
    for document in documents:
        for file in document.files:
            if not file.sha256:
                continue
            origin = first_seen.get(file.sha256)
            if origin is None:
                first_seen[file.sha256] = f"{document.card_id}/{file.row_id}"
            else:
                file.duplicate_of = origin
                duplicates += 1
    return duplicates


def build_manifest(
    result: WalkResult,
    file_records: dict[UUID, list[FileRecord]],
    config: ExportConfig,
    *,
    synthetic: bool = False,
    now: datetime | None = None,
    source: str = "tessa",
) -> Manifest:
    coverage = config.coverage
    documents = [
        document_entry(
            visited.snapshot,
            visited.depth,
            visited.entry_paths,
            file_records.get(card_id, []),
            coverage,
            config.status,
        )
        for card_id, visited in result.cards.items()
    ]
    duplicates = mark_duplicates(documents)

    all_files = [file for document in documents for file in document.files]
    skipped = [file for file in all_files if not file.downloaded]
    status_values: dict[str, str | None] = {}
    relation_types: dict[str, str | None] = {}
    for visited in result.cards.values():
        snapshot = visited.snapshot
        status_id = snapshot.common_text("StatusID")
        if status_id:
            status_values.setdefault(status_id, snapshot.common_text("StatusNameStatus"))
        for link in snapshot.outgoing:
            if link.ref_type_name:
                relation_types.setdefault(link.ref_type_name, link.ref_type_reverse_name)

    stats = ManifestStats(
        documents=len(documents),
        excluded=len(result.excluded),
        errors=len(result.errors),
        skipped_links=len(result.skipped_links),
        files_total=len(all_files),
        files_downloaded=len(all_files) - len(skipped),
        files_skipped=len(skipped),
        skipped_by_reason=dict(Counter(file.skipped_reason or "" for file in skipped)),
        skipped_by_extension=dict(
            Counter(file.extension or "без расширения" for file in skipped if file.skipped_reason == "format")
        ),
        duplicate_files=duplicates,
        doc_kinds=dict(Counter(document.doc_kind for document in documents)),
        coverage_kinds=dict(Counter(kind for document in documents for kind in document.coverage_kinds)),
        doc_statuses=dict(Counter(document.doc_status for document in documents)),
        card_types=dict(Counter(document.type_name or "?" for document in documents)),
        status_values=status_values,
        relation_types=relation_types,
    )
    return Manifest(
        created_at=now or datetime.now(UTC),
        tool=_package_name,
        source=source,
        synthetic=synthetic,
        seed_ids=list(result.seed_ids),
        traversal=config.traversal.model_dump(),
        allowed_extensions=list(config.files.allowed_extensions),
        exclude_rules=len(config.exclude_rules),
        documents=documents,
        excluded=[
            ExcludedEntry(
                card_id=item.card_id,
                reason=item.reason,
                depth=item.depth,
                type_name=item.type_name,
                doc_type_title=item.doc_type_title,
                entry_paths=_entry_paths(item.entry_paths),
            )
            for item in result.excluded.values()
        ],
        errors=[
            ErrorEntry(
                card_id=item.card_id,
                depth=item.depth,
                kind=item.kind,
                message=item.message,
                entry_paths=_entry_paths(item.entry_paths),
            )
            for item in result.errors
        ],
        skipped_links=[
            SkippedLinkEntry(
                from_card_id=item.from_card_id,
                to_card_id=item.to_card_id,
                direction=item.direction,
                relation=item.relation,
                reason=item.reason,
                detail=item.detail,
            )
            for item in result.skipped_links
        ],
        stats=stats,
    )


def _edge_entry(edge: LinkEdge) -> LinkEdgeEntry:
    return LinkEdgeEntry(
        from_id=edge.from_card_id,
        to_id=edge.to_card_id,
        relation_type=edge.relation_type,
        reverse_type=edge.reverse_type,
        relation_type_id=edge.relation_type_id,
        source=edge.source,
    )


def build_links_graph(result: WalkResult) -> LinksGraph:
    return LinksGraph(
        edges=[_edge_entry(edge) for edge in result.edges],
        dangling_edges=[_edge_entry(edge) for edge in result.dangling_edges],
    )
