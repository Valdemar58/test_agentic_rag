"""Конфигурация экспорт-скрипта: один YAML-файл + учётные данные из переменных окружения.

Все параметры обхода, форматов, правил исключения и ориентиров покрытия задаются здесь;
значения по умолчанию соответствуют разделу 8 ТЗ и решениям заказчика.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Literal
from uuid import UUID

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

Direction = Literal["outgoing", "incoming"]

# Статус «Отмененный» в справочнике статусов заказчика (ответ заказчика, 2026-09-14)
CANCELLED_STATUS_ID = UUID("de9d3b6d-532b-4cb8-aa7b-e055e8986e48")


class ConfigError(ValueError):
    """Ошибка конфигурации с понятным для заказчика сообщением."""


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TessaSettings(StrictModel):
    base_url: str = Field(description="Адрес сервера Тессы, например https://tessa.company.local")
    username_env: str = Field(default="TESSA_USERNAME", description="Имя переменной окружения с логином")
    password_env: str = Field(default="TESSA_PASSWORD", description="Имя переменной окружения с паролем")
    verify_tls: bool = Field(default=True, description="Проверять TLS-сертификат сервера")
    ca_bundle: Path | None = Field(default=None, description="Путь к корпоративному CA, если нужен")
    tessa_version: str = Field(default="4.2", description="Значение заголовка tessa-version")
    timeout_seconds: float = Field(default=60.0, gt=0, description="Тайм-аут HTTP-запроса")
    max_retries: int = Field(default=2, ge=0, description="Повторы при обрыве соединения")

    @field_validator("base_url")
    @classmethod
    def _check_url(cls, value: str) -> str:
        if not re.match(r"^https?://", value):
            raise ValueError("base_url должен начинаться с http:// или https://")
        return value.rstrip("/")


class ExternalCodeSettings(StrictModel):
    tessa_sdk_path: Path = Field(description="Корень репозитория SDK Тессы (пакет tessa_client в src/)")
    card_service_path: Path = Field(
        description="Корень репозитория сервиса карточек (пакет robot_skills в src/)"
    )


class TraversalSettings(StrictModel):
    max_depth: int = Field(default=2, ge=0, description="Максимальная глубина обхода связей от seed")
    max_docs: int = Field(default=200, ge=1, description="Предохранительный лимит числа документов")
    directions: list[Direction] = Field(default=["outgoing", "incoming"], description="Какие связи обходить")

    @field_validator("directions")
    @classmethod
    def _check_directions(cls, value: list[Direction]) -> list[Direction]:
        if not value:
            raise ValueError("directions не может быть пустым")
        if len(set(value)) != len(value):
            raise ValueError("directions содержит повторы")
        return value


class ExcludeRule(StrictModel):
    """Правило исключения документа из экспорта. Срабатывает, если совпал любой критерий."""

    reason: str = Field(default="", description="Подпись правила для отчёта")
    card_type_names: list[str] = Field(default_factory=list, description="TypeName карточки")
    doc_type_titles: list[str] = Field(default_factory=list, description="DocTypeTitle документа")
    field: str | None = Field(default=None, description="Поле секции вида Секция.Поле")
    values: list[str] = Field(default_factory=list, description="Значения поля, при которых исключать")
    card_ids: list[UUID] = Field(default_factory=list, description="Явный список ID карточек")

    @model_validator(mode="after")
    def _check_criteria(self) -> ExcludeRule:
        if not (self.card_type_names or self.doc_type_titles or self.field or self.card_ids):
            raise ValueError("правило исключения не содержит ни одного критерия")
        if (self.field is None) != (not self.values):
            raise ValueError("field и values задаются вместе")
        if self.field is not None and self.field.count(".") != 1:
            raise ValueError("field должен иметь вид Секция.Поле")
        return self


class FilesSettings(StrictModel):
    allowed_extensions: list[str] = Field(
        default=["pdf", "docx", "xlsx", "pptx", "png", "jpg", "jpeg", "tif", "tiff", "bmp", "gif"],
        description="Расширения файлов, которые скачиваются и идут в инжест; остальные пропускаются",
    )
    image_extensions: list[str] = Field(
        default=["png", "jpg", "jpeg", "tif", "tiff", "bmp", "gif"],
        description="Расширения, считающиеся сканами (без текстового слоя)",
    )

    @field_validator("allowed_extensions", "image_extensions")
    @classmethod
    def _normalize(cls, value: list[str]) -> list[str]:
        normalized = [item.lower().lstrip(".") for item in value]
        if not normalized:
            raise ValueError("список расширений не может быть пустым")
        return normalized


class CoverageSettings(StrictModel):
    """Ориентиры состава корпуса из раздела 8.2 ТЗ. Дефицит даёт предупреждение, не ошибку."""

    doc_kind_map: dict[str, list[str]] = Field(
        default={
            "Приказы": ["приказ"],
            "Положения / ЛНА": ["положени", "регламент", "политик", "стандарт", "лна"],
            "Инструкции": ["инструкц"],
            "Договоры": ["договор", "соглашени", "контракт"],
            "Акты": [r"\bакт"],
            "Служебные записки": ["служебн", "записк"],
        },
        description="Категория чек-листа → регэкспы (без учёта регистра) над DocTypeTitle/TypeCaption",
    )
    other_kind_label: str = Field(default="Прочие виды", description="Категория для несопоставленных")
    targets: dict[str, int] = Field(
        default={
            "Приказы": 20,
            "Положения / ЛНА": 10,
            "Инструкции": 10,
            "Договоры": 10,
            "Акты": 5,
            "Служебные записки": 10,
            "Прочие виды": 5,
        },
        description="Желательное число документов по категориям",
    )
    min_cancelled_orders: int = Field(default=5, description="Приказов со статусом «отменён»")
    min_linked_orders: int = Field(default=5, description="Приказов в связке «отменяет/изменяет»")
    min_orders_with_terms: int = Field(default=3, description="Приказов с разделом терминов")
    min_regulations_with_terms: int = Field(default=2, description="Положений с разделом терминов")
    min_instructions_with_tables: int = Field(default=3, description="Инструкций со списками/таблицами")
    min_contracts_with_attachments: int = Field(default=3, description="Договоров с приложениями")
    min_memos_with_links: int = Field(default=3, description="Служебных записок со ссылками")
    min_digital_share: float = Field(default=0.60, description="Доля файлов с текстовым слоем")
    min_scan_share: float = Field(default=0.15, description="Доля сканов")
    min_docs_with_tables: int = Field(default=10, description="Документов с таблицами")
    min_date_span_years: int = Field(default=3, description="Разброс дат документов, лет")
    cancelled_status_ids: list[UUID] = Field(
        default=[CANCELLED_STATUS_ID], description="Значения StatusID, означающие «отменён»"
    )
    terms_section_patterns: list[str] = Field(
        default=["термины и определения", "термины, определения", "сокращения", "используемые термины"],
        description="Регэкспы заголовков раздела терминов",
    )


class ExportConfig(StrictModel):
    tessa: TessaSettings
    external: ExternalCodeSettings
    seed_file: Path = Field(default=Path("seed_cards.yaml"), description="Seed-список карточек")
    output_dir: Path = Field(default=Path("output"), description="Каталог результата и лога")
    archive_name: str = Field(default="tessa_export.zip", description="Имя итогового архива")
    traversal: TraversalSettings = Field(default_factory=TraversalSettings)
    exclude_rules: list[ExcludeRule] = Field(default_factory=list)
    files: FilesSettings = Field(default_factory=FilesSettings)
    coverage: CoverageSettings = Field(default_factory=CoverageSettings)
    log_level: str = Field(default="INFO", description="Уровень логирования")

    def resolve_credentials(self) -> tuple[str, str]:
        """Читает логин и пароль из переменных окружения, названных в конфиге."""
        username = os.environ.get(self.tessa.username_env, "")
        password = os.environ.get(self.tessa.password_env, "")
        if not username or not password:
            raise ConfigError(
                "не заданы учётные данные Тессы: установите переменные окружения "
                f"{self.tessa.username_env} и {self.tessa.password_env}"
            )
        return username, password


class SeedCard(StrictModel):
    id: UUID
    comment: str = ""


def _format_validation_error(exc: ValidationError) -> str:
    lines = []
    for error in exc.errors():
        location = ".".join(str(part) for part in error["loc"]) or "<корень>"
        lines.append(f"  - {location}: {error['msg']}")
    return "\n".join(lines)


def _read_yaml(path: Path, what: str) -> Any:
    if not path.is_file():
        raise ConfigError(f"{what} не найден: {path}")
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"{what} {path}: некорректный YAML: {exc}") from exc


def load_config(path: Path) -> ExportConfig:
    """Загружает и валидирует конфиг; относительные пути считаются от каталога конфига."""
    data = _read_yaml(path, "конфиг")
    if not isinstance(data, dict):
        raise ConfigError(f"конфиг {path}: ожидался YAML-объект с ключами tessa, external, …")
    try:
        config = ExportConfig.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(f"конфиг {path} содержит ошибки:\n{_format_validation_error(exc)}") from exc
    base = path.resolve().parent
    if not config.seed_file.is_absolute():
        config.seed_file = base / config.seed_file
    if not config.output_dir.is_absolute():
        config.output_dir = base / config.output_dir
    return config


def load_seed(path: Path) -> list[SeedCard]:
    """Читает seed-список: либо {cards: [{id, comment}]}, либо просто список ID."""
    data = _read_yaml(path, "seed-файл")
    raw_items: Any
    if isinstance(data, dict) and "cards" in data:
        raw_items = data["cards"]
    else:
        raw_items = data
    if not isinstance(raw_items, list) or not raw_items:
        raise ConfigError(f"seed-файл {path}: ожидался непустой список карточек")
    cards: list[SeedCard] = []
    for index, item in enumerate(raw_items, start=1):
        payload = {"id": item} if isinstance(item, str | UUID) else item
        try:
            cards.append(SeedCard.model_validate(payload))
        except ValidationError as exc:
            raise ConfigError(f"seed-файл {path}, элемент {index}: {_format_validation_error(exc)}") from exc
    seen: set[UUID] = set()
    duplicates: list[str] = []
    for card in cards:
        if card.id in seen:
            duplicates.append(str(card.id))
        seen.add(card.id)
    if duplicates:
        raise ConfigError(f"seed-файл {path}: повторяющиеся ID: {', '.join(duplicates)}")
    return cards
