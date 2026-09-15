"""Карточка документа из архива экспорта: `cards/<id>.json` по схеме CardData сервиса карточек.

Модель повторяет форму JSON контракта (`robot_skills.core.cards.schemas.CardData`) ровно настолько,
насколько нужно инжесту: секции с полями и строками, файлы, сведения о типе. Сам контракт
импортируется только в контрактных тестах (§8.0 ТЗ), чтобы инжест работал без внешних путей;
совпадение формы с реальной схемой проверяет `tests/contract/test_synthetic_cards.py`.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, ValidationError

COMMON_SECTION = "DocumentCommonInfo"
OUTGOING_SECTION = "OutgoingRefDocs"
INCOMING_SECTION = "IncomingRefDocs"
APPROVAL_SECTION = "KrApprovalCommonInfoVirtual"


class CardReadError(Exception):
    """Файл карточки не найден, не читается или не соответствует форме CardData."""


class _Loose(BaseModel):
    # Контракт может расширяться: лишние поля не ломают инжест, за формой следит контрактный тест
    model_config = ConfigDict(extra="ignore", frozen=True)


class CardSection(_Loose):
    name: str | None = None
    type: int = 0
    fields: dict[str, Any] | None = Field(default=None, description="Поля строковой секции")
    rows: list[dict[str, Any]] | None = Field(default=None, description="Строки табличной секции")


class CardFile(_Loose):
    row_id: UUID
    name: str
    category_id: UUID | None = None
    category_caption: str | None = None
    version_row_id: UUID | None = None
    version_number: int = 0
    is_virtual: bool = False
    size: int = 0


class CardRecord(_Loose):
    id: UUID
    type_id: UUID | None = None
    type_name: str | None = None
    type_caption: str | None = None
    created: datetime | None = None
    created_by_name: str | None = None
    modified: datetime | None = None
    modified_by_name: str | None = None
    sections: dict[str, CardSection] = Field(default_factory=dict)
    files: list[CardFile] = Field(default_factory=list)

    def fields(self, section: str) -> dict[str, Any]:
        item = self.sections.get(section)
        return dict(item.fields) if item is not None and item.fields else {}

    def rows(self, section: str) -> list[dict[str, Any]]:
        item = self.sections.get(section)
        return [dict(row) for row in item.rows] if item is not None and item.rows else []

    def common_field(self, name: str) -> Any:
        return self.fields(COMMON_SECTION).get(name)

    def common_text(self, name: str) -> str | None:
        value = self.common_field(name)
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    def file_by_row_id(self, row_id: UUID) -> CardFile | None:
        return next((file for file in self.files if file.row_id == row_id), None)


def load_card(path: Path) -> CardRecord:
    """Читает `cards/<id>.json`; любая проблема — `CardReadError` с понятным текстом."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise CardReadError(f"файл карточки не найден: {path.name}") from exc
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CardReadError(f"карточка {path.name} не читается как JSON: {exc}") from exc
    try:
        return CardRecord.model_validate(raw)
    except ValidationError as exc:
        first = exc.errors()[0]
        location = ".".join(str(part) for part in first["loc"]) or "корень"
        raise CardReadError(
            f"карточка {path.name} не соответствует форме CardData: {location}: {first['msg']}"
        ) from exc
