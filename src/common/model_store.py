"""Локальное хранилище весов моделей: загрузка по pinned-ревизиям и проверка наличия (NFR-3, NFR-5).

Каждая модель лежит в `models/<local_name>` как обычный каталог (без symlink'ов кэша HF); после
успешной загрузки рядом пишется `.revision` с commit'ом Hugging Face. Рантайм читает веса только
отсюда и работает офлайн (`HF_HUB_OFFLINE=1`), сеть нужна один раз — на шаге загрузки.
"""

from __future__ import annotations

import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from common.config import ModelSource

REVISION_FILE = ".revision"
HF_METADATA_DIR = ".cache"


class Downloader(Protocol):
    def __call__(
        self, *, repo_id: str, revision: str, local_dir: Path, ignore_patterns: Sequence[str]
    ) -> None: ...


def hf_downloader(*, repo_id: str, revision: str, local_dir: Path, ignore_patterns: Sequence[str]) -> None:
    """Загрузка снимка репозитория HF в обычный каталог; повторный вызов докачивает недостающее."""
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=repo_id,
        revision=revision,
        local_dir=str(local_dir),
        ignore_patterns=list(ignore_patterns) or None,
    )


@dataclass(frozen=True)
class ModelStatus:
    name: str
    repo_id: str
    revision: str
    path: Path
    present: bool
    size_bytes: int

    @property
    def size_gib(self) -> float:
        return self.size_bytes / 2**30


def stored_revision(path: Path) -> str | None:
    marker = path / REVISION_FILE
    if not marker.is_file():
        return None
    return marker.read_text(encoding="utf-8").strip() or None


def directory_size(path: Path) -> int:
    """Размер весов без служебных метаданных загрузчика."""
    if not path.is_dir():
        return 0
    total = 0
    for item in path.rglob("*"):
        relative = item.relative_to(path)
        if HF_METADATA_DIR in relative.parts or relative.name == REVISION_FILE:
            continue
        if item.is_file():
            total += item.stat().st_size
    return total


def model_status(source: ModelSource, models_dir: Path) -> ModelStatus:
    path = models_dir / source.local_name
    present = stored_revision(path) == source.revision
    return ModelStatus(
        name=source.local_name,
        repo_id=source.repo_id,
        revision=source.revision,
        path=path,
        present=present,
        size_bytes=directory_size(path),
    )


def ensure_model(
    source: ModelSource, models_dir: Path, *, downloader: Downloader = hf_downloader
) -> tuple[ModelStatus, bool]:
    """Гарантирует наличие модели нужной ревизии; возвращает (статус, была ли загрузка).

    Каталог с другой ревизией очищается целиком, чтобы не смешивать файлы двух снимков;
    незавершённая загрузка (нет маркера) докачивается на месте.
    """
    before = model_status(source, models_dir)
    if before.present:
        return before, False
    stored = stored_revision(before.path)
    if stored is not None and stored != source.revision:
        shutil.rmtree(before.path)
    before.path.mkdir(parents=True, exist_ok=True)
    downloader(
        repo_id=source.repo_id,
        revision=source.revision,
        local_dir=before.path,
        ignore_patterns=source.ignore_patterns,
    )
    (before.path / REVISION_FILE).write_text(source.revision + "\n", encoding="utf-8")
    return model_status(source, models_dir), True
