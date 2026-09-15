"""Единый конфиг стенда: `configs/app.yaml` (NFR-4).

Здесь структура конфига и его загрузка. Секреты и пути к внешнему коду в файл не попадают:
они читаются из переменных окружения отдельными настройками (задача 3.3). Путь к файлу можно
переопределить переменной окружения `APP_CONFIG_PATH`.
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = ROOT / "configs" / "app.yaml"
CONFIG_PATH_ENV = "APP_CONFIG_PATH"


class ConfigError(Exception):
    """Конфиг не найден, не читается или не проходит проверку."""


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class GpuSettings(StrictModel):
    vram_budget_gib: float = Field(
        gt=0, description="Бюджет VRAM целевой карты (RTX 3080 — 12 ГиБ); действует на любой карте"
    )
    vllm_memory_gib: float = Field(
        gt=0, description="Сколько из бюджета отдаётся серверу vLLM: веса, активации и KV-кэш"
    )
    max_utilization: float = Field(
        gt=0, le=1, description="Предохранитель: доля памяти карты, выше которой vLLM не поднимается"
    )

    @model_validator(mode="after")
    def _memory_within_budget(self) -> GpuSettings:
        if self.vllm_memory_gib > self.vram_budget_gib:
            raise ValueError(
                f"gpu.vllm_memory_gib={self.vllm_memory_gib} больше бюджета "
                f"gpu.vram_budget_gib={self.vram_budget_gib}"
            )
        return self


class VllmServerSettings(StrictModel):
    served_model_name: str = Field(description="Имя модели в OpenAI-совместимом API")
    port: int = Field(ge=1, le=65535, description="Порт на хосте")
    max_model_len: int = Field(gt=0, description="Максимальная длина контекста (токены)")
    max_num_seqs: int = Field(gt=0, description="Максимум одновременных последовательностей")


class QwenServerSettings(VllmServerSettings):
    tool_call_parser: str = Field(description="Парсер вызовов инструментов vLLM (для Qwen3 — hermes)")
    reasoning_parser: str = Field(description="Парсер блока рассуждений vLLM (для Qwen3 — qwen3)")


class VllmSettings(StrictModel):
    qwen: QwenServerSettings
    dots: VllmServerSettings


class ModelSource(StrictModel):
    repo_id: str = Field(pattern=r"^[\w.-]+/[\w.-]+$", description="Репозиторий на Hugging Face")
    revision: str = Field(pattern=r"^[0-9a-f]{40}$", description="Commit на Hugging Face (NFR-3)")
    local_name: str = Field(
        pattern=r"^[A-Za-z0-9_-]+$",
        description="Имя каталога внутри models.dir; без точек — требование dots.mocr",
    )
    ignore_patterns: list[str] = Field(default_factory=list, description="Файлы, которые не скачиваются")


class ModelsSettings(StrictModel):
    dir: Path = Field(description="Каталог весов; относительный путь считается от корня репозитория")
    qwen: ModelSource
    dots: ModelSource
    embedding: ModelSource
    reranker: ModelSource

    @property
    def dir_absolute(self) -> Path:
        return self.dir if self.dir.is_absolute() else ROOT / self.dir

    def local_path(self, source: ModelSource) -> Path:
        return self.dir_absolute / source.local_name

    def all_sources(self) -> list[ModelSource]:
        return [self.qwen, self.dots, self.embedding, self.reranker]


class AppConfig(StrictModel):
    gpu: GpuSettings
    vllm: VllmSettings
    models: ModelsSettings


def config_path() -> Path:
    override = os.environ.get(CONFIG_PATH_ENV)
    return Path(override) if override else DEFAULT_CONFIG_PATH


def load_app_config(path: Path | None = None) -> AppConfig:
    """Читает YAML и валидирует его строго: неизвестные и пропущенные поля — ошибка."""
    target = path or config_path()
    try:
        raw = yaml.safe_load(target.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"конфиг не найден: {target}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"конфиг {target} не разбирается как YAML: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"конфиг {target}: ожидался словарь верхнего уровня")
    try:
        return AppConfig.model_validate(raw)
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(part) for part in error['loc']) or '<корень>'}: {error['msg']}"
            for error in exc.errors()
        )
        raise ConfigError(f"конфиг {target} не прошёл проверку: {problems}") from exc
