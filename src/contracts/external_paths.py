"""Подключение внешнего кода SDK Тессы и сервиса карточек (§8.0 ТЗ: в репозиторий не копируется).

Два способа, оба не тащат чужой код к нам:

1. Каталоги репозиториев в переменных TESSA_SDK_PATH и CARD_SERVICE_PATH (или в .env) — их
   подкаталоги src добавляются в sys.path.
2. Пакеты `tessa_client` и `robot_skills`, уже установленные в окружение (например, колёсами из
   внутреннего devpi) — тогда переменные не нужны.

Путь имеет приоритет; если он задан, но пакета там нет, а в окружении пакет установлен, берётся
установленный. Когда нет ни того, ни другого, код сообщает понятную причину, а зависящие тесты
скипаются.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

SDK_PACKAGE = "tessa_client"
CARD_SERVICE_PACKAGE = "robot_skills"


# Каталоги, которые этот модуль сам добавил в sys.path: пакет оттуда — не «установленный»
_attached: set[Path] = set()


def package_installed(package: str) -> bool:
    """True, если пакет ставится в окружение (колесом), а не подхвачен из каталога по пути."""
    try:
        spec = find_spec(package)
    except (ImportError, ValueError):
        return False
    if spec is None:
        return False
    origin = Path(spec.origin) if spec.origin else None
    return origin is None or not any(origin.is_relative_to(entry) for entry in _attached)


class ExternalPaths(BaseSettings):
    """Пути к внешним репозиториям SDK Тессы и сервиса карточек."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    tessa_sdk_path: Path | None = Field(
        default=None, description="Корень репозитория SDK Тессы (пакет tessa_client лежит в src/)"
    )
    card_service_path: Path | None = Field(
        default=None, description="Корень репозитория сервиса карточек (пакет robot_skills лежит в src/)"
    )


@dataclass(frozen=True)
class ExternalPathsStatus:
    """Результат проверки внешнего кода: откуда берётся каждый пакет."""

    available: bool
    reason: str | None
    source_dirs: tuple[Path, ...]
    installed_packages: tuple[str, ...] = ()


def resolve_external_paths(paths: ExternalPaths | None = None) -> ExternalPathsStatus:
    """Ищет каждый пакет по пути из окружения, иначе среди установленных в окружение."""
    settings = paths if paths is not None else ExternalPaths()
    expected = (
        ("TESSA_SDK_PATH", settings.tessa_sdk_path, SDK_PACKAGE),
        ("CARD_SERVICE_PATH", settings.card_service_path, CARD_SERVICE_PACKAGE),
    )
    problems: list[str] = []
    source_dirs: list[Path] = []
    installed: list[str] = []
    for env_name, root, package in expected:
        source_dir = None if root is None else root / "src"
        if source_dir is not None and (source_dir / package / "__init__.py").is_file():
            source_dirs.append(source_dir)
            continue
        if package_installed(package):
            installed.append(package)
            continue
        where = (
            f"{env_name} не задан" if source_dir is None else f"{env_name}={root}: нет {source_dir / package}"
        )
        problems.append(f"{where}, и пакет {package} не установлен в окружение")
    if problems:
        reason = "внешний код недоступен: " + "; ".join(problems)
        return ExternalPathsStatus(available=False, reason=reason, source_dirs=())
    return ExternalPathsStatus(
        available=True,
        reason=None,
        source_dirs=tuple(source_dirs),
        installed_packages=tuple(installed),
    )


def ensure_external_paths(paths: ExternalPaths | None = None) -> ExternalPathsStatus:
    """Добавляет каталоги src внешнего кода в sys.path. Идемпотентно."""
    status = resolve_external_paths(paths)
    if status.available:
        for source_dir in status.source_dirs:
            entry = str(source_dir)
            _attached.add(source_dir)
            if entry not in sys.path:
                sys.path.append(entry)
    return status


def contracts_available() -> bool:
    """True, если внешние контракты можно импортировать."""
    return resolve_external_paths().available


def missing_reason() -> str | None:
    """Причина недоступности внешних контрактов или None."""
    return resolve_external_paths().reason
