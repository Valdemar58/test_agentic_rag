"""Снимок карточки и протокол шлюза к Тессе.

Обход, сериализация и валидация работают только с этими структурами и не зависят от SDK:
реальный шлюз (gateway_sdk) и фейк для тестов (fake) отдают одинаковые снимки.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any, Protocol
from uuid import UUID

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


class TessaGateway(Protocol):
    def check_connection(self) -> None:
        """Проверяет доступ к серверу и учётные данные до начала обхода."""
        ...

    def get_card(self, card_id: UUID) -> CardSnapshot: ...

    def download_file(self, card_id: UUID, file: FileInfo) -> DownloadedContent: ...

    def close(self) -> None: ...


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
