"""Скачивание файлов карточек: только последняя версия, только разрешённые форматы,
виртуальные файлы пропускаются, ошибка одного файла не останавливает экспорт."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

from tessa_export.models import FileInfo, GatewayError, TessaGateway
from tessa_export.storage import FILES_DIR, safe_file_name

logger = logging.getLogger(__name__)

SKIP_VIRTUAL = "virtual"
SKIP_FORMAT = "format"
SKIP_ERROR = "error"


@dataclass
class FileRecord:
    """Итог обработки одного файла карточки."""

    file: FileInfo
    relative_path: str | None = None
    sha256: str | None = None
    size: int | None = None
    content_type: str | None = None
    server_file_name: str | None = None
    skipped_reason: str | None = None
    skipped_detail: str = ""
    # Признаки, которые заполняет проверка файлов (validation): None = не проверялось
    smoke_ok: bool | None = None
    smoke_error: str | None = None
    has_text_layer: bool | None = None
    page_count: int | None = None
    has_tables: bool | None = None
    has_terms_section: bool | None = None

    @property
    def downloaded(self) -> bool:
        return self.relative_path is not None


def skip_reason(file: FileInfo, allowed_extensions: set[str]) -> tuple[str, str] | None:
    """Причина пропуска файла до скачивания или None."""
    if file.is_virtual:
        return SKIP_VIRTUAL, file.type_name or "виртуальный файл"
    extension = file.extension
    if extension not in allowed_extensions:
        return SKIP_FORMAT, extension or "без расширения"
    return None


def download_card_files(
    gateway: TessaGateway,
    card_id: UUID,
    files: list[FileInfo],
    allowed_extensions: set[str],
    export_root: Path,
) -> list[FileRecord]:
    records: list[FileRecord] = []
    used_names: set[str] = set()
    card_dir = export_root / FILES_DIR / str(card_id)
    for file in files:
        record = FileRecord(file=file)
        records.append(record)
        skip = skip_reason(file, allowed_extensions)
        if skip is not None:
            record.skipped_reason, record.skipped_detail = skip
            logger.info("Файл «%s» карточки %s пропущен: %s (%s)", file.name, card_id, skip[0], skip[1])
            continue
        try:
            downloaded = gateway.download_file(card_id, file)
        except GatewayError as exc:
            record.skipped_reason, record.skipped_detail = SKIP_ERROR, str(exc)
            logger.warning("Файл «%s» карточки %s не скачан: %s", file.name, card_id, exc)
            continue
        content = downloaded.content
        if not content:
            record.skipped_reason, record.skipped_detail = SKIP_ERROR, "получено 0 байт"
            logger.warning("Файл «%s» карточки %s пуст", file.name, card_id)
            continue
        name = safe_file_name(file.name or downloaded.file_name or "", used_names, str(file.row_id))
        target = card_dir / name
        try:
            card_dir.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
        except OSError as exc:
            record.skipped_reason, record.skipped_detail = SKIP_ERROR, f"не записан на диск: {exc}"
            logger.warning("Файл «%s» карточки %s не записан на диск: %s", name, card_id, exc)
            continue
        record.relative_path = f"{FILES_DIR}/{card_id}/{name}"
        record.sha256 = hashlib.sha256(content).hexdigest()
        record.size = len(content)
        record.content_type = downloaded.content_type
        record.server_file_name = downloaded.file_name
        logger.info("Файл «%s» карточки %s сохранён (%d байт)", name, card_id, len(content))
    return records
