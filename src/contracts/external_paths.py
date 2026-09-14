"""Подключение внешнего кода SDK Тессы и сервиса карточек по путям из окружения.

Пути задаются переменными TESSA_SDK_PATH и CARD_SERVICE_PATH (или в .env). Каталоги src
этих репозиториев добавляются в sys.path, после чего пакеты tessa_client и robot_skills
импортируются как обычные внешние пакеты. Если пути не заданы или неверны, код сообщает
понятную причину, а зависящие тесты скипаются.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

SDK_PACKAGE = "tessa_client"
CARD_SERVICE_PACKAGE = "robot_skills"


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
    """Результат проверки внешних путей."""

    available: bool
    reason: str | None
    source_dirs: tuple[Path, ...]


def resolve_external_paths(paths: ExternalPaths | None = None) -> ExternalPathsStatus:
    """Проверяет, что оба пути заданы и содержат ожидаемые пакеты в подкаталоге src."""
    settings = paths if paths is not None else ExternalPaths()
    expected = (
        ("TESSA_SDK_PATH", settings.tessa_sdk_path, SDK_PACKAGE),
        ("CARD_SERVICE_PATH", settings.card_service_path, CARD_SERVICE_PACKAGE),
    )
    problems: list[str] = []
    source_dirs: list[Path] = []
    for env_name, root, package in expected:
        if root is None:
            problems.append(f"{env_name} не задан")
            continue
        source_dir = root / "src"
        if not (source_dir / package / "__init__.py").is_file():
            problems.append(f"{env_name}={root}: пакет {package} не найден в {source_dir}")
            continue
        source_dirs.append(source_dir)
    if problems:
        reason = "внешний код недоступен: " + "; ".join(problems)
        return ExternalPathsStatus(available=False, reason=reason, source_dirs=())
    return ExternalPathsStatus(available=True, reason=None, source_dirs=tuple(source_dirs))


def ensure_external_paths(paths: ExternalPaths | None = None) -> ExternalPathsStatus:
    """Добавляет каталоги src внешнего кода в sys.path. Идемпотентно."""
    status = resolve_external_paths(paths)
    if status.available:
        for source_dir in status.source_dirs:
            entry = str(source_dir)
            if entry not in sys.path:
                sys.path.append(entry)
    return status


def contracts_available() -> bool:
    """True, если внешние контракты можно импортировать."""
    return resolve_external_paths().available


def missing_reason() -> str | None:
    """Причина недоступности внешних контрактов или None."""
    return resolve_external_paths().reason
