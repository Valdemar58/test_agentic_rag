"""Офлайн-шлюз: карточки и файлы берутся из ранее полученного архива экспорта.

Позволяет применить правила исключения и лимиты обхода к уже скачанному полному замыканию
связей, не обращаясь к Тессе (команда `tessa-export filter`): обход, манифест, отчёт, таблица
отбора и архив строятся тем же кодом, что и при онлайн-запуске. Карточка восстанавливается из
`cards/<id>.json` (схема CardData сервиса карточек) и `cards_raw/<id>.json`, файлы читаются по
путям из `manifest.json` источника.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any
from uuid import UUID

from tessa_export.models import (
    INCOMING_SECTION,
    OUTGOING_SECTION,
    CardNotFoundError,
    CardSnapshot,
    DownloadedContent,
    FileInfo,
    GatewayError,
    SectionSnapshot,
    parse_links,
)
from tessa_export.storage import CARDS_DIR, CARDS_RAW_DIR, MANIFEST_NAME


class ArchiveSource:
    """Чтение файлов экспорта из распакованного каталога export/ или из zip-архива."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._zip: zipfile.ZipFile | None = None
        self._names: set[str] = set()
        if path.is_dir():
            return
        if path.is_file() and zipfile.is_zipfile(path):
            self._zip = zipfile.ZipFile(path)
            self._names = set(self._zip.namelist())
            return
        raise GatewayError(f"источник {path} не является ни каталогом экспорта, ни zip-архивом")

    def exists(self, name: str) -> bool:
        if self._zip is not None:
            return name in self._names
        return (self.path / name).is_file()

    def read_bytes(self, name: str) -> bytes:
        if self._zip is not None:
            return self._zip.read(name)
        return (self.path / name).read_bytes()

    def read_json(self, name: str) -> Any:
        return json.loads(self.read_bytes(name).decode("utf-8"))

    def close(self) -> None:
        if self._zip is not None:
            self._zip.close()


def _optional_uuid(value: Any) -> UUID | None:
    return UUID(str(value)) if value else None


def snapshot_from_card_data(card_data: dict[str, Any], raw: dict[str, Any]) -> CardSnapshot:
    """Снимок карточки из JSON по схеме CardData (cards/<id>.json) и сырого ответа Тессы."""
    sections = {
        name: SectionSnapshot(
            fields=dict(section["fields"]) if section.get("fields") is not None else None,
            rows=[dict(row) for row in section["rows"]] if section.get("rows") is not None else None,
        )
        for name, section in (card_data.get("sections") or {}).items()
    }
    files = [
        FileInfo(
            row_id=UUID(item["row_id"]),
            version_row_id=UUID(item["version_row_id"]),
            name=item.get("name") or "",
            size=int(item.get("size") or 0),
            version_number=int(item.get("version_number") or 1),
            category=item.get("category_caption"),
            is_virtual=bool(item.get("is_virtual")),
            type_name=item.get("type_name"),
        )
        for item in card_data.get("files") or []
    ]
    outgoing_rows = sections[OUTGOING_SECTION].rows if OUTGOING_SECTION in sections else None
    incoming_rows = sections[INCOMING_SECTION].rows if INCOMING_SECTION in sections else None
    return CardSnapshot(
        card_id=UUID(card_data["id"]),
        type_id=_optional_uuid(card_data.get("type_id")),
        type_name=card_data.get("type_name"),
        type_caption=card_data.get("type_caption"),
        sections=sections,
        files=files,
        outgoing=parse_links(outgoing_rows),
        incoming=parse_links(incoming_rows),
        raw=raw,
        card_data_json=card_data,
    )


class ArchiveGateway:
    """Реализация TessaGateway поверх ранее полученного архива экспорта."""

    def __init__(self, source: ArchiveSource) -> None:
        self._source = source
        try:
            manifest = source.read_json(MANIFEST_NAME)
        except (KeyError, FileNotFoundError, ValueError) as exc:
            raise GatewayError(f"в источнике {source.path} нет читаемого {MANIFEST_NAME}: {exc}") from exc
        # (card_id, row_id) → путь и content-type скачанного файла в источнике
        self._files: dict[tuple[UUID, UUID], tuple[str, str | None]] = {}
        documents = manifest.get("documents") or []
        for document in documents:
            card_id = UUID(document["card_id"])
            for file in document.get("files") or []:
                if file.get("downloaded") and file.get("path"):
                    self._files[(card_id, UUID(file["row_id"]))] = (file["path"], file.get("content_type"))
        self.documents_in_source = len(documents)

    def check_connection(self) -> None:
        """Источник уже проверен при создании: manifest.json прочитан."""

    def get_card(self, card_id: UUID) -> CardSnapshot:
        name = f"{CARDS_DIR}/{card_id}.json"
        if not self._source.exists(name):
            raise CardNotFoundError(
                f"карточка {card_id}: нет в исходном архиве (была за пределами обхода или исключена)"
            )
        card_data = self._source.read_json(name)
        raw_name = f"{CARDS_RAW_DIR}/{card_id}.json"
        raw = self._source.read_json(raw_name) if self._source.exists(raw_name) else {}
        return snapshot_from_card_data(card_data, raw)

    def download_file(self, card_id: UUID, file: FileInfo) -> DownloadedContent:
        entry = self._files.get((card_id, file.row_id))
        if entry is None:
            raise GatewayError(f"файл «{file.name}» карточки {card_id}: не был скачан в исходном архиве")
        path, content_type = entry
        return DownloadedContent(
            content=self._source.read_bytes(path), file_name=file.name, content_type=content_type
        )

    def close(self) -> None:
        self._source.close()
