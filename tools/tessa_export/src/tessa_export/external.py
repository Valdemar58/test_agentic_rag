"""Подключение SDK Тессы и схем сервиса карточек по путям из конфига (§8.0 ТЗ).

Код не копируется в пакет: каталоги src внешних репозиториев добавляются в sys.path.
В Docker-образе они лежат в /opt/external, пути заполнены в config.example.yaml.
"""

from __future__ import annotations

import sys
from pathlib import Path

from tessa_export.config import ExternalCodeSettings

SDK_PACKAGE = "tessa_client"
CARD_SERVICE_PACKAGE = "robot_skills"


class ExternalCodeError(RuntimeError):
    """Внешний код не найден по указанному пути."""


def _source_dir(label: str, root: Path, package: str, hint: str) -> Path:
    source_dir = root / "src"
    if not (source_dir / package / "__init__.py").is_file():
        raise ExternalCodeError(
            f"{hint} не найден: ожидался пакет {package} в {source_dir}. "
            f"Проверьте параметр {label} в конфиге."
        )
    return source_dir


def attach_external_code(settings: ExternalCodeSettings) -> None:
    """Проверяет пути и добавляет каталоги src в sys.path. Идемпотентно."""
    sources = [
        _source_dir("external.tessa_sdk_path", settings.tessa_sdk_path, SDK_PACKAGE, "SDK Тессы"),
        _source_dir(
            "external.card_service_path",
            settings.card_service_path,
            CARD_SERVICE_PACKAGE,
            "Сервис карточек (схемы)",
        ),
    ]
    for source in sources:
        entry = str(source)
        if entry not in sys.path:
            sys.path.append(entry)
