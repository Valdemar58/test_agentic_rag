"""Упаковка результата экспорта в один zip-архив (§8.1.6)."""

from __future__ import annotations

import zipfile
from pathlib import Path


def make_archive(export_root: Path, archive_path: Path) -> Path:
    """Пакует содержимое export_root в архив; пути внутри архива относительные, с прямыми слэшами."""
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    if archive_path.exists():
        archive_path.unlink()
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(export_root.rglob("*")):
            if path.is_file():
                archive.write(path, arcname=path.relative_to(export_root).as_posix())
    return archive_path
