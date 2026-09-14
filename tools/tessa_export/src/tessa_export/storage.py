"""Раскладка результата экспорта на диске и запись JSON.

output_dir/
  export/                    рабочий каталог, упаковывается в архив
    cards/<card_id>.json     карточка по схеме CardData сервиса карточек
    cards_raw/<card_id>.json сырой ответ Тессы cards/get
    files/<card_id>/<имя>    скачанные файлы
    manifest.json, links_graph.json, validation_report.md
    documents_review.csv     таблица для ручного отбора состава сета
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
REVIEW_NAME = "documents_review.csv"
LOG_NAME = "tessa_export.log"

_UNSAFE_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
# Лимит имени файла в ext4/NTFS — 255 байт/символов; кириллица в UTF-8 занимает 2 байта на символ.
# Запас нужен под суффикс коллизии «__<uuid>» (38 байт): 200 + 38 < 255.
_MAX_NAME_BYTES = 200


def _truncate_utf8(text: str, max_bytes: int) -> str:
    """Обрезает строку так, чтобы её UTF-8 представление не превышало max_bytes байт."""
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text
    return encoded[:max_bytes].decode("utf-8", errors="ignore").rstrip(" .")


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
    """Имя файла без разделителей путей и запрещённых символов, не длиннее лимита файловой
    системы (в байтах UTF-8, расширение сохраняется); при коллизии добавляется fallback."""
    base = Path(name.replace("\\", "/")).name if name else ""
    base = _UNSAFE_CHARS.sub("_", base).strip(" .")
    if not base or base in {".", ".."}:
        base = fallback
    stem, suffix = Path(base).stem, Path(base).suffix
    stem = _truncate_utf8(stem, _MAX_NAME_BYTES - len(suffix.encode("utf-8"))) or fallback
    candidate = f"{stem}{suffix}"
    if candidate.casefold() in used:
        candidate = f"{stem}__{fallback}{suffix}"
    used.add(candidate.casefold())
    return candidate
