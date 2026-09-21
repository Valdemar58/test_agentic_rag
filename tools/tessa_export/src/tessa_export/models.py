"""Снимок карточки и протокол шлюза к Тессе.

Обход, сериализация и валидация работают только с этими структурами и не зависят от SDK:
реальный шлюз (gateway_sdk) и фейк для тестов (fake) отдают одинаковые снимки.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any, Protocol
from uuid import UUID

from tessa_export.config import ViewParameter

COMMON_SECTION = "DocumentCommonInfo"
OUTGOING_SECTION = "OutgoingRefDocs"
INCOMING_SECTION = "IncomingRefDocs"


class GatewayError(Exception):
    """Базовая ошибка обращения к Тессе."""


class CardNotFoundError(GatewayError):
    """Карточка не найдена или сервер вернул пустую карточку."""


class CardAccessError(GatewayError):
    """Нет прав на карточку или файл, либо сессия отклонена."""


class GatewayConnectionError(GatewayError):
    """Сетевая ошибка или тайм-аут."""


@dataclass(frozen=True)
class FileInfo:
    row_id: UUID
    version_row_id: UUID
    name: str
    size: int
    version_number: int = 1
    category: str | None = None
    is_virtual: bool = False
    type_name: str | None = None

    @property
    def extension(self) -> str:
        return PurePosixPath(self.name).suffix.lower().lstrip(".")


@dataclass(frozen=True)
class LinkInfo:
    doc_id: UUID
    description: str | None = None
    doc_type_name: str | None = None
    ref_type_id: UUID | None = None
    ref_type_name: str | None = None
    ref_type_reverse_name: str | None = None
    order: int | None = None


@dataclass(frozen=True)
class SectionSnapshot:
    fields: dict[str, Any] | None = None
    rows: list[dict[str, Any]] | None = None


@dataclass(frozen=True)
class CardSnapshot:
    """Карточка в том виде, в котором её видит экспорт."""

    card_id: UUID
    type_id: UUID | None
    type_name: str | None
    type_caption: str | None
    sections: dict[str, SectionSnapshot] = field(default_factory=dict)
    files: list[FileInfo] = field(default_factory=list)
    outgoing: list[LinkInfo] = field(default_factory=list)
    incoming: list[LinkInfo] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)
    card_data_json: dict[str, Any] = field(default_factory=dict)

    def section_field(self, section: str, name: str) -> Any:
        snapshot = self.sections.get(section)
        if snapshot is None or snapshot.fields is None:
            return None
        return snapshot.fields.get(name)

    def common_field(self, name: str) -> Any:
        return self.section_field(COMMON_SECTION, name)

    def common_text(self, name: str) -> str | None:
        value = self.common_field(name)
        if value is None:
            return None
        text = str(value).strip()
        return text or None


@dataclass(frozen=True)
class DownloadedContent:
    content: bytes
    file_name: str | None = None
    content_type: str | None = None


@dataclass(frozen=True)
class ViewMeta:
    """Представление Тессы в перечне доступных: алиас, подпись, колонки и параметры фильтра."""

    alias: str
    caption: str | None = None
    columns: list[str] = field(default_factory=list)
    parameters: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ViewPage:
    """Страница результата представления: позиционные строки уже разложены по именам колонок."""

    columns: list[str]
    rows: list[dict[str, Any]]
    row_count: int = 0
    """Всего строк в представлении; заполняется, только если запрошен подсчёт (with_count)."""


class TessaGateway(Protocol):
    def check_connection(self) -> None:
        """Проверяет доступ к серверу и учётные данные до начала обхода."""
        ...

    def get_card(self, card_id: UUID) -> CardSnapshot: ...

    def download_file(self, card_id: UUID, file: FileInfo) -> DownloadedContent: ...

    def close(self) -> None: ...


class ViewSource(Protocol):
    """Чтение представлений Тессы: перечень документов берётся только отсюда (§8 API Тессы)."""

    def list_views(self) -> list[ViewMeta]: ...

    def view_page(
        self,
        alias: str,
        parameters: Sequence[ViewParameter] = (),
        *,
        subset: str | None = None,
        sorting: tuple[str, bool] | None = None,
        page_offset: int | None = None,
        page_limit: int | None = None,
        with_count: bool = False,
    ) -> ViewPage:
        """`page_offset` — номер первой строки окна (Тесса считает смещение в строках, не в
        страницах). `with_count` запрашивает общее число строк представления в `row_count`."""
        ...


class TessaViewGateway(TessaGateway, ViewSource, Protocol):
    """Шлюз, который умеет и карточки, и представления (нужен режиму синхронизации приказов)."""


def rows_by_column(columns: Sequence[str], rows: Sequence[Sequence[Any]]) -> list[dict[str, Any]]:
    """Позиционные строки представления → словари по именам колонок; лишние значения отбрасываются."""
    names = list(columns)
    return [dict(zip(names, row, strict=False)) for row in rows]


def _optional_uuid(value: Any) -> UUID | None:
    if value is None or value == "":
        return None
    try:
        return UUID(str(value))
    except ValueError:
        return None


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def parse_links(rows: list[dict[str, Any]] | None) -> list[LinkInfo]:
    """Строки секций OutgoingRefDocs / IncomingRefDocs → связи. Строки без DocID пропускаются."""
    links: list[LinkInfo] = []
    for row in rows or []:
        doc_id = _optional_uuid(row.get("DocID"))
        if doc_id is None:
            continue
        order = row.get("Order")
        links.append(
            LinkInfo(
                doc_id=doc_id,
                description=_optional_text(row.get("DocDescription")),
                doc_type_name=_optional_text(row.get("DocTypeName")),
                ref_type_id=_optional_uuid(row.get("RefTypeID")),
                ref_type_name=_optional_text(row.get("RefTypeName")),
                ref_type_reverse_name=_optional_text(row.get("RefTypeReverseName")),
                order=int(order) if isinstance(order, int) else None,
            )
        )
    return links
