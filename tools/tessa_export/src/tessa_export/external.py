"""Подключение SDK Тессы и схем сервиса карточек (§8.0 ТЗ: код не копируется в пакет).

Каждый из двух пакетов ищется по трём источникам подряд, первый подходящий выигрывает:

1. каталог репозитория из конфига (`external.tessa_sdk_path`, `external.card_service_path`) —
   так работает Docker-образ, где они лежат в /opt/external;
2. каталог из переменной окружения `TESSA_SDK_PATH` / `CARD_SERVICE_PATH` (в том числе из `.env`) —
   те же переменные, что читает остальной проект;
3. пакет, установленный в окружение (например, колесом из внутреннего индекса).

Если не нашлось нигде, ошибка называет все три источника и их состояние.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path

from tessa_export.config import ExternalCodeSettings

SDK_PACKAGE = "tessa_client"
CARD_SERVICE_PACKAGE = "robot_skills"
SDK_PATH_ENV = "TESSA_SDK_PATH"
CARD_SERVICE_PATH_ENV = "CARD_SERVICE_PATH"


class ExternalCodeError(RuntimeError):
    """Внешний код не найден: ни по пути, ни в переменной окружения, ни среди установленных."""


# Каталоги, которые этот модуль сам добавил в sys.path: пакет оттуда — не «установленный»
_attached: set[Path] = set()


def package_installed(package: str) -> bool:
    """True, если пакет стоит в окружении (колесом), а не подхвачен из каталога по пути."""
    try:
        spec = find_spec(package)
    except (ImportError, ValueError):
        return False
    if spec is None:
        return False
    origin = Path(spec.origin) if spec.origin else None
    return origin is None or not any(origin.is_relative_to(entry) for entry in _attached)


@dataclass(frozen=True)
class _Source:
    """Один источник каталога с внешним кодом и его состояние для сообщения об ошибке."""

    label: str
    root: Path | None

    def source_dir(self, package: str) -> Path | None:
        if self.root is None:
            return None
        source_dir = self.root / "src"
        return source_dir if (source_dir / package / "__init__.py").is_file() else None

    def describe(self, package: str) -> str:
        if self.root is None:
            return f"{self.label} не задан"
        return f"{self.label} = {self.root}, но пакета {package} там нет"


def _env_path(name: str) -> Path | None:
    value = os.environ.get(name, "").strip()
    return Path(value) if value else None


def _resolve(package: str, hint: str, sources: list[_Source]) -> Path | None:
    """Каталог src для sys.path или None, если пакет уже установлен в окружение."""
    for source in sources:
        found = source.source_dir(package)
        if found is not None:
            return found
    if package_installed(package):
        return None
    where = "; ".join(source.describe(package) for source in sources)
    raise ExternalCodeError(
        f"{hint} не найден: {where}; и пакет {package} не установлен в окружение "
        "(например, из внутреннего индекса пакетов)."
    )


def attach_external_code(settings: ExternalCodeSettings) -> None:
    """Проверяет источники внешнего кода и добавляет каталоги src в sys.path. Идемпотентно."""
    sources = [
        _resolve(
            SDK_PACKAGE,
            "SDK Тессы",
            [
                _Source("параметр external.tessa_sdk_path в конфиге", settings.tessa_sdk_path),
                _Source(f"переменная {SDK_PATH_ENV}", _env_path(SDK_PATH_ENV)),
            ],
        ),
        _resolve(
            CARD_SERVICE_PACKAGE,
            "Сервис карточек (схемы)",
            [
                _Source("параметр external.card_service_path в конфиге", settings.card_service_path),
                _Source(f"переменная {CARD_SERVICE_PATH_ENV}", _env_path(CARD_SERVICE_PATH_ENV)),
            ],
        ),
    ]
    for source in sources:
        if source is None:
            continue
        entry = str(source)
        _attached.add(source)
        if entry not in sys.path:
            sys.path.append(entry)
