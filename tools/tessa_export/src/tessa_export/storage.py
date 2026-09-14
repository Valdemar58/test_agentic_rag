"""Раскладка результата экспорта на диске и запись JSON.

output_dir/
  export/                    рабочий каталог, упаковывается в архив
    cards/<card_id>.json     карточка по схеме CardData сервиса карточек
    cards_raw/<card_id>.json сырой ответ Тессы cards/get
    files/<card_id>/<имя>    скачанные файлы
    manifest.json, links_graph.json, validation_report.md
  tessa_export.log
  <archive_name>
"""

from __future__ import annotations

import json
import re
from datetime import date, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

from tessa_export.models import CardSnapshot

EXPORT_DIR_NAME = "export"
CARDS_DIR = "cards"
CARDS_RAW_DIR = "cards_raw"
FILES_DIR = "files"
MANIFEST_NAME = "manifest.json"
LINKS_GRAPH_NAME = "links_graph.json"
REPORT_NAME = "validation_report.md"
LOG_NAME = "tessa_export.log"

_UNSAFE_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_MAX_NAME_LENGTH = 150


def _json_default(value: Any) -> Any:
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, datetime | date):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"объект типа {type(value).__name__} не сериализуется в JSON")


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(data, ensure_ascii=False, indent=2, default=_json_default)
    path.write_text(text + "\n", encoding="utf-8", newline="\n")


def export_dir(output_dir: Path) -> Path:
    return output_dir / EXPORT_DIR_NAME


def save_card(export_root: Path, snapshot: CardSnapshot) -> tuple[Path, Path]:
    """Пишет cards/<id>.json (CardData) и cards_raw/<id>.json (сырой ответ)."""
    card_path = export_root / CARDS_DIR / f"{snapshot.card_id}.json"
    raw_path = export_root / CARDS_RAW_DIR / f"{snapshot.card_id}.json"
    write_json(card_path, snapshot.card_data_json)
    write_json(raw_path, snapshot.raw)
    return card_path, raw_path


def safe_file_name(name: str, used: set[str], fallback: str) -> str:
    """Имя файла без разделителей путей и запрещённых символов; при коллизии добавляется fallback."""
    base = Path(name.replace("\\", "/")).name if name else ""
    base = _UNSAFE_CHARS.sub("_", base).strip(" .")
    if not base or base in {".", ".."}:
        base = fallback
    if len(base) > _MAX_NAME_LENGTH:
        suffix = Path(base).suffix
        base = base[: _MAX_NAME_LENGTH - len(suffix)] + suffix
    candidate = base
    if candidate.casefold() in used:
        stem, suffix = Path(base).stem, Path(base).suffix
        candidate = f"{stem}__{fallback}{suffix}"
    used.add(candidate.casefold())
    return candidate
