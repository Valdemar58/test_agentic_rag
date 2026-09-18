"""Фейковый шлюз и конструктор синтетических карточек для тестов и режима самопроверки.

Снимки собираются в той же структуре, что отдаёт SdkGateway: секции с реальными именами полей
Тессы (DocumentCommonInfo, OutgoingRefDocs, IncomingRefDocs), сырой JSON с типовыми суффиксами
и представление по форме CardData. Данные синтетические.
"""

from __future__ import annotations

import mimetypes
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any
from uuid import NAMESPACE_URL, UUID, uuid5

from tessa_export.config import CANCELLED_STATUS_ID, ViewParameter
from tessa_export.models import (
    COMMON_SECTION,
    INCOMING_SECTION,
    OUTGOING_SECTION,
    CardNotFoundError,
    CardSnapshot,
    DownloadedContent,
    FileInfo,
    GatewayError,
    LinkInfo,
    SectionSnapshot,
    ViewMeta,
    ViewPage,
)
from tessa_export.sample_files import minimal_docx_bytes, minimal_image_bytes, minimal_pdf_bytes

FILE_TYPE_ID = UUID("ab387c69-fd62-0655-bbc3-b879e433a143")
SYSTEM_USER_ID = UUID("11111111-1111-1111-1111-111111111111")
CANCEL_REF_TYPE_ID = UUID("efaa9300-4b08-47f8-8891-f7389bb4dc5a")
ACTIVE_STATUS_ID = UUID("0c8f3d3e-3b7f-4b7e-9a5c-1f2e3d4c5b6a")
DOC_CATEGORY_ID = UUID("d72084fc-7f8b-4b25-a4f1-153b784374aa")


def stable_uuid(*parts: str) -> UUID:
    """Детерминированный UUID для воспроизводимых фикстур."""
    return uuid5(NAMESPACE_URL, "tessa-export:" + ":".join(parts))


def link(
    doc_id: UUID,
    *,
    ref_type_name: str | None = "в отмену",
    ref_type_reverse_name: str | None = "отменено",
    description: str | None = None,
    doc_type_name: str | None = "Приказ",
    order: int | None = 1,
) -> LinkInfo:
    return LinkInfo(
        doc_id=doc_id,
        description=description or f"документ {str(doc_id)[:8]}",
        doc_type_name=doc_type_name,
        ref_type_id=CANCEL_REF_TYPE_ID if ref_type_name else None,
        ref_type_name=ref_type_name,
        ref_type_reverse_name=ref_type_reverse_name,
        order=order,
    )


def make_file(
    card_id: UUID,
    name: str,
    *,
    size: int = 1024,
    category: str | None = "Документ",
    is_virtual: bool = False,
    version_number: int = 1,
) -> FileInfo:
    return FileInfo(
        row_id=stable_uuid(str(card_id), "file", name),
        version_row_id=stable_uuid(str(card_id), "version", name),
        name=name,
        size=size,
        version_number=version_number,
        category=category,
        is_virtual=is_virtual,
        type_name="KrVirtualFileType" if is_virtual else "File",
    )


def _link_row(item: LinkInfo, *, typed: bool) -> dict[str, Any]:
    row: dict[str, Any] = {
        "RowID": stable_uuid("row", str(item.doc_id), item.ref_type_name or ""),
        "DocID": str(item.doc_id),
        "DocDescription": item.description,
    }
    if typed:
        row.update(
            {
                "DocTypeName": item.doc_type_name,
                "Order": item.order,
                "RefTypeID": str(item.ref_type_id) if item.ref_type_id else None,
                "RefTypeName": item.ref_type_name,
                "RefTypeReverseName": item.ref_type_reverse_name,
            }
        )
    return row


def _raw_link_row(row: dict[str, Any], *, typed: bool) -> dict[str, Any]:
    raw: dict[str, Any] = {
        "RowID::uid": str(row["RowID"]),
        "DocID::uid": row["DocID"],
        "DocDescription": row["DocDescription"],
    }
    if typed:
        raw.update(
            {
                "DocTypeName": row["DocTypeName"],
                "Order::int": row["Order"],
                "RefTypeID::uid": row["RefTypeID"],
                "RefTypeName": row["RefTypeName"],
                "RefTypeReverseName": row["RefTypeReverseName"],
            }
        )
    return raw


def make_snapshot(
    card_id: UUID,
    *,
    type_name: str = "OrderMKC",
    type_caption: str = "Приказ",
    doc_type_title: str | None = None,
    number: str = "1",
    doc_date: datetime | None = None,
    subject: str = "О назначении ответственных",
    status_id: UUID | None = ACTIVE_STATUS_ID,
    status_name: str | None = "Действующий",
    state_id: int = 6,
    state_name: str = "$KrStates_Doc_Registered",
    department: str | None = "Отдел охраны труда",
    outgoing: list[LinkInfo] | None = None,
    incoming: list[LinkInfo] | None = None,
    files: list[FileInfo] | None = None,
    extra_fields: dict[str, Any] | None = None,
) -> CardSnapshot:
    """Синтетическая карточка документа в структуре реального OrderMKC."""
    doc_date = doc_date or datetime(2026, 1, 15, tzinfo=UTC)
    doc_type_title = doc_type_title or type_caption
    department_id = stable_uuid("department", department) if department else None
    fields: dict[str, Any] = {
        "FullNumber": number,
        "DocDate": doc_date,
        "Subject": subject,
        "DocTypeID": str(stable_uuid("doctype", doc_type_title)),
        "DocTypeTitle": doc_type_title,
        "TypeDocumentNameTypeDocument": doc_type_title,
        "DepartmentID": str(department_id) if department_id else None,
        "DepartmentName": department,
        "AuthorID": str(SYSTEM_USER_ID),
        "AuthorName": "С.С. Сотрудник1",
        "RegistratorName": "С.С. Сотрудник2",
        "SignedByName": "С.С. Сотрудник3",
        "StatusID": str(status_id) if status_id else None,
        "StatusNameStatus": status_name,
        "StateID": state_id,
        "StateName": state_name,
        "Comment": None,
    }
    if extra_fields:
        fields.update(extra_fields)
    outgoing = list(outgoing or [])
    incoming = list(incoming or [])
    files = list(files or [])
    outgoing_rows = [_link_row(item, typed=True) for item in outgoing]
    incoming_rows = [_link_row(item, typed=False) for item in incoming]

    sections = {
        COMMON_SECTION: SectionSnapshot(fields=fields),
        OUTGOING_SECTION: SectionSnapshot(rows=outgoing_rows or None),
        INCOMING_SECTION: SectionSnapshot(rows=incoming_rows or None),
    }
    raw_fields = {
        "FullNumber": number,
        "DocDate::dtm": doc_date.isoformat().replace("+00:00", "Z"),
        "Subject": subject,
        "DocTypeID::uid": fields["DocTypeID"],
        "DocTypeTitle": doc_type_title,
        "TypeDocumentNameTypeDocument": doc_type_title,
        "DepartmentID::uid": fields["DepartmentID"],
        "DepartmentName": department,
        "AuthorID::uid": fields["AuthorID"],
        "AuthorName": fields["AuthorName"],
        "RegistratorName": fields["RegistratorName"],
        "SignedByName": fields["SignedByName"],
        "StatusID::uid": fields["StatusID"],
        "StatusNameStatus": status_name,
        "StateID::int": state_id,
        "StateName": state_name,
        "Comment": fields["Comment"],
    }
    type_id = stable_uuid("cardtype", type_name)
    created = datetime(2026, 1, 10, tzinfo=UTC)
    raw: dict[str, Any] = {
        "Info": None,
        "ValidationResult": {"Items": None},
        "CancelOpening": False,
        "Card": {
            "ID::uid": str(card_id),
            "TypeID::uid": str(type_id),
            "TypeName": type_name,
            "TypeCaption": type_caption,
            "Created::dtm": created.isoformat().replace("+00:00", "Z"),
            "CreatedByID::uid": str(SYSTEM_USER_ID),
            "CreatedByName": "System",
            "Modified::dtm": doc_date.isoformat().replace("+00:00", "Z"),
            "ModifiedByID::uid": str(SYSTEM_USER_ID),
            "ModifiedByName": "System",
            "Flags::int": 0,
            "Version::int": 1,
            "Sections": {
                COMMON_SECTION: {"Fields": raw_fields},
                OUTGOING_SECTION: {
                    ".table::int": 1,
                    "Rows": [_raw_link_row(row, typed=True) for row in outgoing_rows] or None,
                },
                INCOMING_SECTION: {
                    ".table::int": 1,
                    "Rows": [_raw_link_row(row, typed=False) for row in incoming_rows] or None,
                },
            },
            "Files": [
                {
                    ".deletionMode::int": 0,
                    ".flags::int": 0,
                    ".state::int": 0,
                    ".versionsLoaded": False,
                    "RowID::uid": str(item.row_id),
                    "TypeID::uid": str(FILE_TYPE_ID),
                    "TypeName": item.type_name,
                    "TypeCaption": "$CardTypes_TypesNames_File",
                    "CategoryID::uid": str(DOC_CATEGORY_ID) if item.category else None,
                    "CategoryCaption": item.category,
                    "CategoryOrder::int": 0,
                    "Name": item.name,
                    "VersionRowID::uid": str(item.version_row_id),
                    "VersionNumber::int": item.version_number,
                    "Hash": None,
                    "IsVirtual": item.is_virtual,
                    "StoreSource::int": 1,
                    "Size": item.size,
                }
                for item in files
            ],
            "Permissions": {"CardPermissions::int": 0, "Sections": None, "FilePermissions": None},
        },
        "SectionRows": {},
    }
    card_data_json: dict[str, Any] = {
        "id": str(card_id),
        "type_id": str(type_id),
        "type_name": type_name,
        "type_caption": type_caption,
        "created": created.isoformat(),
        "created_by_id": str(SYSTEM_USER_ID),
        "created_by_name": "System",
        "modified": doc_date.isoformat(),
        "modified_by_id": str(SYSTEM_USER_ID),
        "modified_by_name": "System",
        "flags": 0,
        "version": 1,
        "sections": {
            COMMON_SECTION: _card_data_section(fields={**fields, "DocDate": doc_date.isoformat()}),
            OUTGOING_SECTION: _card_data_section(rows=_json_rows(outgoing_rows)),
            INCOMING_SECTION: _card_data_section(rows=_json_rows(incoming_rows)),
        },
        "files": [
            {
                "row_id": str(item.row_id),
                "type_id": str(FILE_TYPE_ID),
                "type_name": item.type_name,
                "type_caption": "$CardTypes_TypesNames_File",
                "category_id": str(DOC_CATEGORY_ID) if item.category else None,
                "category_caption": item.category,
                "name": item.name,
                "version_row_id": str(item.version_row_id),
                "version_number": item.version_number,
                "is_virtual": item.is_virtual,
                "size": item.size,
                "state": 0,
                "card": None,
            }
            for item in files
        ],
        "permissions": {"card_permissions": 0, "sections": None, "file_permissions": None},
    }
    return CardSnapshot(
        card_id=card_id,
        type_id=type_id,
        type_name=type_name,
        type_caption=type_caption,
        sections=sections,
        files=files,
        outgoing=outgoing,
        incoming=incoming,
        raw=raw,
        card_data_json=card_data_json,
    )


def _card_data_section(
    *, fields: dict[str, Any] | None = None, rows: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    return {
        "name": None,
        "type": 0 if rows is None else 1,
        "rows": rows,
        "row_sorting_type": 0,
        "table_type": 0,
        "fields": fields,
    }


def _json_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]] | None:
    if not rows:
        return None
    return [
        {key: (str(value) if isinstance(value, UUID) else value) for key, value in row.items()}
        for row in rows
    ]


class FakeGateway:
    """Шлюз на заранее заданных снимках. Ошибки можно подставлять поштучно."""

    def __init__(self) -> None:
        self.cards: dict[UUID, CardSnapshot] = {}
        self.contents: dict[tuple[UUID, UUID], bytes] = {}
        self.card_errors: dict[UUID, Exception] = {}
        self.file_errors: dict[tuple[UUID, UUID], Exception] = {}
        self.get_calls: list[UUID] = []
        self.download_calls: list[tuple[UUID, UUID]] = []
        self.views: dict[str, ViewMeta] = {}
        self.view_data: dict[str, list[dict[str, Any]]] = {}
        self.view_paging: dict[str, bool] = {}
        self.view_calls: list[tuple[str, int | None, list[ViewParameter]]] = []
        self.closed = False
        self.connection_error: Exception | None = None

    def check_connection(self) -> None:
        if self.connection_error is not None:
            raise self.connection_error

    def add(self, snapshot: CardSnapshot, contents: dict[str, bytes] | None = None) -> CardSnapshot:
        """Регистрирует карточку; contents — содержимое файлов по имени."""
        self.cards[snapshot.card_id] = snapshot
        for file in snapshot.files:
            if contents and file.name in contents:
                self.contents[(snapshot.card_id, file.row_id)] = contents[file.name]
        return snapshot

    def get_card(self, card_id: UUID) -> CardSnapshot:
        self.get_calls.append(card_id)
        if card_id in self.card_errors:
            raise self.card_errors[card_id]
        if card_id not in self.cards:
            raise CardNotFoundError(f"карточка {card_id}: не найдена (фейк)")
        return self.cards[card_id]

    def download_file(self, card_id: UUID, file: FileInfo) -> DownloadedContent:
        key = (card_id, file.row_id)
        self.download_calls.append(key)
        if key in self.file_errors:
            raise self.file_errors[key]
        if key not in self.contents:
            raise GatewayError(f"файл «{file.name}» карточки {card_id}: содержимое не задано (фейк)")
        content_type = mimetypes.guess_type(file.name)[0] or "application/octet-stream"
        return DownloadedContent(content=self.contents[key], file_name=file.name, content_type=content_type)

    def add_view(
        self,
        alias: str,
        columns: list[str],
        rows: list[dict[str, Any]],
        *,
        caption: str | None = None,
        parameters: list[str] | None = None,
        paging: bool = True,
    ) -> None:
        """Регистрирует представление; paging=False имитирует представление без пагинации."""
        self.views[alias] = ViewMeta(alias, caption, list(columns), list(parameters or []))
        self.view_data[alias] = list(rows)
        self.view_paging[alias] = paging

    def list_views(self) -> list[ViewMeta]:
        return list(self.views.values())

    def view_page(
        self,
        alias: str,
        parameters: Sequence[ViewParameter] = (),
        *,
        subset: str | None = None,
        sorting: tuple[str, bool] | None = None,
        page_offset: int | None = None,
        page_limit: int | None = None,
    ) -> ViewPage:
        self.view_calls.append((alias, page_offset, list(parameters)))
        if alias not in self.views:
            raise GatewayError(f"представление «{alias}»: не найдено (фейк)")
        columns = self.views[alias].columns
        rows = self.view_data[alias]
        if page_limit is not None and self.view_paging.get(alias, True):
            start = ((page_offset or 1) - 1) * page_limit
            rows = rows[start : start + page_limit]
        elif page_limit is not None:
            rows = rows[:page_limit]
        return ViewPage(columns=list(columns), rows=[dict(row) for row in rows], row_count=len(rows))

    def close(self) -> None:
        self.closed = True


def build_demo_scenario() -> tuple[FakeGateway, list[UUID]]:
    """Сценарий для e2e-тестов и самопроверки: цикл A↔B, цепочка до глубины 3, два пути входа,
    карточка без файлов, виртуальный и неподдерживаемые файлы, кириллические имена.

    Возвращает шлюз и seed-список [A, G].
    """
    a, b, c, d, e, f, g = (stable_uuid("card", name) for name in "ABCDEFG")
    gateway = FakeGateway()

    order_docx = "Приказ 144 оригинал.docx"
    order_print = "Для печати_Приказ 144.docx"
    order_pdf = "144 от 15.01.2026 О назначении ответственных.pdf"
    scan_jpg = "Скан приложения.jpg"
    gateway.add(
        make_snapshot(
            a,
            number="144",
            status_id=CANCELLED_STATUS_ID,
            status_name="Отмененный",
            outgoing=[link(b), link(e, ref_type_name="приложение", ref_type_reverse_name="приложение к")],
            incoming=[link(b, ref_type_name=None, ref_type_reverse_name=None)],
            files=[
                make_file(a, order_docx, version_number=8),
                make_file(a, order_print),
                make_file(a, order_pdf),
                make_file(a, "Подпись.sig", category=None, size=64),
                make_file(a, "Лист согласования.html", category=None, is_virtual=True, size=-1),
                make_file(a, scan_jpg),
            ],
            extra_fields={"Comment": "Отменен приказом № 173."},
        ),
        {
            order_docx: minimal_docx_bytes(
                ["Приказ № 144", "1. Назначить ответственных."],
                with_table=True,
                terms_heading="Термины и определения",
            ),
            order_print: minimal_docx_bytes(["Приказ № 144 (печатная форма)"]),
            order_pdf: minimal_pdf_bytes(
                "Order 144 on appointment of persons responsible for gas hazardous works"
            ),
            "Подпись.sig": b"signature",
            scan_jpg: minimal_image_bytes(),
        },
    )
    gateway.add(
        make_snapshot(
            b,
            number="173",
            outgoing=[link(a), link(c, ref_type_name="изменяет", ref_type_reverse_name="изменён")],
            incoming=[link(a, ref_type_name=None, ref_type_reverse_name=None)],
            files=[make_file(b, "Приказ 173.pdf"), make_file(b, "Скан приказа 173.pdf")],
        ),
        {
            "Приказ 173.pdf": minimal_pdf_bytes(
                "Order 173 cancelling order 144 and appointing new responsible persons"
            ),
            "Скан приказа 173.pdf": minimal_pdf_bytes(None),
        },
    )
    gateway.add(
        make_snapshot(
            c,
            type_caption="Положение",
            number="П-7",
            doc_date=datetime(2023, 3, 1, tzinfo=UTC),
            outgoing=[link(d, ref_type_name="ссылается на", ref_type_reverse_name="упоминается в")],
            incoming=[link(b, ref_type_name=None, ref_type_reverse_name=None)],
            files=[make_file(c, "Положение.docx")],
        ),
        {"Положение.docx": minimal_docx_bytes(["Положение"], terms_heading="Сокращения")},
    )
    gateway.add(
        make_snapshot(
            d,
            type_caption="Инструкция",
            number="И-3",
            incoming=[link(c, ref_type_name=None, ref_type_reverse_name=None)],
            files=[make_file(d, "Инструкция.pdf")],
        ),
        {"Инструкция.pdf": minimal_pdf_bytes(None)},
    )
    gateway.add(
        make_snapshot(
            e,
            type_caption="Служебная записка",
            number="СЗ-9",
            incoming=[
                link(a, ref_type_name=None, ref_type_reverse_name=None),
                link(g, ref_type_name=None, ref_type_reverse_name=None),
            ],
        )
    )
    gateway.add(
        make_snapshot(
            f,
            type_caption="Договор",
            number="Д-1",
            files=[make_file(f, "Договор.pdf"), make_file(f, "Скан договора.pdf")],
        ),
        {
            "Договор.pdf": minimal_pdf_bytes(
                "Supply contract No. D-1 between the company and the contractor"
            ),
            "Скан договора.pdf": minimal_pdf_bytes(None),
        },
    )
    gateway.add(
        make_snapshot(
            g,
            type_caption="Акт",
            number="А-2",
            outgoing=[link(e, ref_type_name="ссылается на", ref_type_reverse_name="упоминается в")],
            files=[make_file(g, "Акт.xlsx")],
        ),
        {"Акт.xlsx": _minimal_xlsx_bytes()},
    )
    return gateway, [a, g]


def _minimal_xlsx_bytes() -> bytes:
    import io

    from openpyxl import Workbook

    workbook = Workbook()
    sheet = workbook.active
    assert sheet is not None
    sheet["A1"] = "Показатель"
    sheet["B1"] = "Значение"
    buffer = io.BytesIO()
    workbook.save(buffer)
    return buffer.getvalue()
