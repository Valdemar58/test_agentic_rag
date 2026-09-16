"""Данные мока из архива экспорта: карточки по схеме `CardData`, файлы, типы карточек, типы связей.

Состав берётся из `manifest.json` (тот же, что читает инжест), карточки — из `cards/<id>.json`
и валидируются реальной схемой `CardData` при загрузке: мок отдаёт ровно то, что отдал бы сервис
заказчика. Карточка, не прошедшая схему, в лог и пропускается, остальные обслуживаются.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ValidationError

from contracts.card_service import CardServiceContract
from ingest.corpus import CorpusError, resolve_export_dir
from tessa_export.manifest import LinksGraph, Manifest
from tessa_export.storage import LINKS_GRAPH_NAME, MANIFEST_NAME

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StoredFile:
    """Файл карточки; `path` пуст, если экспорт содержимое не скачал (виртуальный, формат, ошибка)."""

    card_id: UUID
    row_id: UUID
    version_row_id: UUID | None
    name: str
    path: Path | None
    content_type: str | None
    skipped_reason: str | None


@dataclass(frozen=True)
class CardTypeInfo:
    id: UUID
    name: str | None
    caption: str | None


@dataclass(frozen=True)
class RefTypeInfo:
    id: UUID | None
    name: str
    reverse_name: str | None


@dataclass
class CardStore:
    export_dir: Path
    synthetic: bool
    cards: dict[UUID, Any]
    files: dict[tuple[UUID, UUID], StoredFile]
    card_types: list[CardTypeInfo]
    ref_types: list[RefTypeInfo]
    invalid_cards: int

    def card(self, card_id: UUID) -> Any | None:
        return self.cards.get(card_id)

    def file(self, card_id: UUID, file_id: UUID) -> StoredFile | None:
        return self.files.get((card_id, file_id))

    def card_type(self, type_id: UUID) -> CardTypeInfo | None:
        return next((item for item in self.card_types if item.id == type_id), None)

    def summary_lines(self) -> list[str]:
        downloaded = sum(1 for item in self.files.values() if item.path is not None)
        lines = [
            f"Архив: {self.export_dir}",
            f"Карточек: {len(self.cards)} (не прошли схему CardData: {self.invalid_cards})",
            f"Файлов: {len(self.files)}, с содержимым в архиве: {downloaded}",
            f"Типов карточек: {len(self.card_types)}, типов связей: {len(self.ref_types)}",
        ]
        if self.synthetic:
            lines.insert(0, "ДАННЫЕ СИНТЕТИЧЕСКИЕ: мок отдаёт синтетический корпус")
        return lines

    @classmethod
    def load(cls, root: Path, contract: CardServiceContract) -> CardStore:
        """Читает архив; карточки валидируются схемой `CardData` реального сервиса."""
        export_dir = resolve_export_dir(root)
        manifest = _read(export_dir / MANIFEST_NAME, Manifest)
        graph = _read(export_dir / LINKS_GRAPH_NAME, LinksGraph)

        cards: dict[UUID, Any] = {}
        files: dict[tuple[UUID, UUID], StoredFile] = {}
        types: dict[UUID, CardTypeInfo] = {}
        invalid = 0
        for entry in manifest.documents:
            card = _load_card(export_dir / entry.card_path, contract)
            if card is None or card.id != entry.card_id:
                invalid += 1
                continue
            cards[card.id] = card
            types.setdefault(card.type_id, CardTypeInfo(card.type_id, card.type_name, card.type_caption))
            versions = {item.row_id: item.version_row_id for item in card.files or []}
            for file in entry.files:
                path = export_dir / file.path if file.downloaded and file.path else None
                files[(card.id, file.row_id)] = StoredFile(
                    card_id=card.id,
                    row_id=file.row_id,
                    version_row_id=versions.get(file.row_id),
                    name=file.name,
                    path=path if path is not None and path.is_file() else None,
                    content_type=file.content_type,
                    skipped_reason=file.skipped_reason if path is None else None,
                )

        ref_ids = {
            edge.relation_type: edge.relation_type_id
            for edge in (*graph.edges, *graph.dangling_edges)
            if edge.relation_type and edge.relation_type_id
        }
        ref_types = [
            RefTypeInfo(id=ref_ids.get(name), name=name, reverse_name=reverse)
            for name, reverse in sorted(manifest.stats.relation_types.items())
        ]
        store = cls(
            export_dir=export_dir,
            synthetic=manifest.synthetic,
            cards=cards,
            files=files,
            card_types=sorted(types.values(), key=lambda item: item.name or ""),
            ref_types=ref_types,
            invalid_cards=invalid,
        )
        for line in store.summary_lines():
            logger.info(line)
        return store


def _read[T: BaseModel](path: Path, model: type[T]) -> T:
    try:
        return model.model_validate_json(path.read_bytes())
    except FileNotFoundError as exc:
        raise CorpusError(f"в архиве нет файла {path.name}: {path.parent}") from exc
    except (ValidationError, ValueError) as exc:
        raise CorpusError(f"{path.name} не соответствует формату экспорта: {exc}") from exc


def _load_card(path: Path, contract: CardServiceContract) -> Any | None:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        logger.error("Карточка %s не читается: %s — пропущена", path.name, exc)
        return None
    try:
        return contract.card_data.model_validate(raw)
    except ValidationError as exc:
        first = exc.errors()[0]
        location = ".".join(str(part) for part in first["loc"]) or "корень"
        logger.error(
            "Карточка %s не проходит схему CardData: %s: %s — пропущена", path.name, location, first["msg"]
        )
        return None
