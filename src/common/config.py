"""Единый конфиг стенда и приложения: `configs/app.yaml` (NFR-4).

Здесь структура конфига и его загрузка. Секреты, URL сервисов и пути к внешнему коду в файл не
попадают: они читаются из переменных окружения (`common.settings`). Путь к файлу можно
переопределить переменной окружения `APP_CONFIG_PATH`.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = ROOT / "configs" / "app.yaml"
CONFIG_PATH_ENV = "APP_CONFIG_PATH"

Device = Literal["cpu", "cuda"]
DocStatus = Literal["active", "cancelled", "draft"]


class ConfigError(Exception):
    """Конфиг не найден, не читается или не проходит проверку."""


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def _absolute(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


# ---------- стенд ----------


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


MODEL_SOURCE_KEYS = ("qwen", "dots", "embedding", "reranker", "docling_layout", "docling_tables")
DOCLING_SOURCE_KEYS = ("docling_layout", "docling_tables")


class ModelsSettings(StrictModel):
    dir: Path = Field(description="Каталог весов; относительный путь считается от корня репозитория")
    qwen: ModelSource
    dots: ModelSource
    embedding: ModelSource
    reranker: ModelSource
    docling_layout: ModelSource = Field(description="Разметка страницы для нативного PDF-конвейера Docling")
    docling_tables: ModelSource = Field(description="TableFormer: структура таблиц в нативном конвейере")

    @model_validator(mode="after")
    def _docling_names_follow_docling_layout(self) -> ModelsSettings:
        # Docling ищет модель в artifacts_path/<repo_id с «/» → «--»>; иначе полезет в сеть
        for key in DOCLING_SOURCE_KEYS:
            source: ModelSource = getattr(self, key)
            expected = source.repo_id.replace("/", "--")
            if source.local_name != expected:
                raise ValueError(f"models.{key}.local_name должен быть {expected!r} (так ищет Docling)")
        return self

    @property
    def dir_absolute(self) -> Path:
        return _absolute(self.dir)

    @property
    def docling_artifacts_dir(self) -> Path:
        """Каталог, который передаётся Docling как artifacts_path (модели лежат в нём по repo_id)."""
        return self.dir_absolute

    def local_path(self, source: ModelSource) -> Path:
        return self.dir_absolute / source.local_name

    def all_sources(self) -> list[ModelSource]:
        return [getattr(self, key) for key in MODEL_SOURCE_KEYS]


# ---------- приложение ----------


class PathsSettings(StrictModel):
    corpus_dir: Path = Field(description="Распакованный архив экспорта (cards, files, manifest.json)")
    work_dir: Path = Field(description="Рабочие данные инжеста: кэш разбора, отчёты")

    @property
    def corpus_dir_absolute(self) -> Path:
        return _absolute(self.corpus_dir)

    @property
    def work_dir_absolute(self) -> Path:
        return _absolute(self.work_dir)


class EmbeddingSettings(StrictModel):
    dense_dim: int = Field(gt=0, description="Размерность dense-вектора bge-m3")
    max_length: int = Field(gt=0, description="Максимум токенов на вход модели")
    batch_size: int = Field(gt=0)
    normalize: bool = Field(description="Нормировать dense-векторы (косинусная близость)")
    runtime_device: Device = Field(description="Устройство в рантайме (§2: CPU)")
    ingest_device: Device = Field(description="Устройство при инжесте (§2: GPU после выгрузки VLM)")


class RerankerSettings(StrictModel):
    device: Device = Field(description="Устройство reranker'а (§2: CPU в рантайме)")
    max_length: int = Field(gt=0, description="Максимум токенов пары запрос+чанк")
    batch_size: int = Field(gt=0)
    quantize_int8: bool = Field(
        description="Динамическое int8-квантование Linear-слоёв на CPU (N17: ~2× быстрее, оценки почти те же)"
    )


class QdrantSettings(StrictModel):
    collection: str = Field(description="Коллекция child-чанков документов (с векторами)")
    parents_collection: str = Field(description="Коллекция parent-разделов (только payload)")
    glossary_collection: str = Field(description="Коллекция глоссария (FR-5)")
    dense_vector: str = Field(description="Имя named vector для dense")
    sparse_vector: str = Field(description="Имя named vector для sparse")
    distance: Literal["Cosine", "Dot", "Euclid"]
    upsert_batch_size: int = Field(gt=0)
    timeout_s: float = Field(gt=0)


class VlmSettings(StrictModel):
    max_tokens: int = Field(gt=0, description="Лимит генерации dots.mocr на страницу")
    image_scale: float = Field(gt=0, description="Масштаб растеризации страницы pdf для VLM")
    raster_image_scale: float = Field(gt=0, description="Масштаб для готовых изображений (jpg/png/tiff)")
    timeout_s: float = Field(gt=0)


class ChunkingSettings(StrictModel):
    max_tokens: int = Field(gt=0, description="Размер чанка фоллбэка и предел структурного чанка")
    overlap_tokens: int = Field(ge=0, description="Перекрытие чанков фоллбэка")
    min_tokens: int = Field(ge=0, description="Короче этого — чанк склеивается с соседом")
    breadcrumb_separator: str = Field(description="Разделитель хлебных крошек в тексте чанка")
    section_title_max_words: int = Field(gt=0, description="Короткий нумерованный абзац = заголовок раздела")
    breadcrumb_max_words: int = Field(gt=0, description="Крошка длиннее стольких слов обрезается с «…»")
    parent_max_tokens: int = Field(
        gt=0, description="Предел parent-чанка (раздела); длиннее — несколько окон"
    )

    @model_validator(mode="after")
    def _overlap_below_size(self) -> ChunkingSettings:
        if self.overlap_tokens >= self.max_tokens:
            raise ValueError("ingest.chunking.overlap_tokens должен быть меньше max_tokens")
        if self.parent_max_tokens < self.max_tokens:
            raise ValueError("ingest.chunking.parent_max_tokens не может быть меньше max_tokens")
        return self


class DoclingSettings(StrictModel):
    device: Device = Field(description="Устройство моделей разметки и таблиц при инжесте (§2: GPU занят VLM)")
    num_threads: int = Field(gt=0, description="Потоки CPU для моделей Docling")
    document_timeout_s: float = Field(gt=0, description="Лимит времени на разбор одного файла")
    table_mode: Literal["accurate", "fast"] = Field(description="Режим TableFormer")
    images_scale: float = Field(gt=0, description="Масштаб растеризации страниц для разметки")


class MainFileCandidate(StrictModel):
    extension: str = Field(description="Расширение без точки")
    categories: list[str] = Field(default_factory=list, description="Допустимые категории файла Тессы")
    any_category: bool = Field(default=False, description="Любая категория, в том числе без категории")

    @model_validator(mode="after")
    def _categories_or_any(self) -> MainFileCandidate:
        if bool(self.categories) == self.any_category:
            raise ValueError("у кандидата основного файла либо categories, либо any_category: true")
        return self


class FileRulesSettings(StrictModel):
    """Правило файлов карточки (О5, docs/chunk_metadata_mapping.md п. 3.1) [ТРЕБУЕТ ПОДТВЕРЖДЕНИЯ]."""

    never_index_name_prefixes: list[str] = Field(description="Имена с таким началом не индексируются")
    never_index_categories: list[str] = Field(description="Категории файлов, которые не индексируются")
    main_candidates: list[MainFileCandidate] = Field(
        min_length=1, description="Порядок выбора основного файла"
    )
    main_text_categories: list[str] = Field(description="Категории с копиями основного текста")
    skip_pdf_duplicate_of_docx_main: bool = Field(
        description="pdf в main_text_categories при docx-оригинале — дубль"
    )
    appendix_categories: list[str] = Field(description="Категории приложений (role=appendix)")
    supplement_categories: list[str] = Field(
        description="Категории дополнительных сведений (role=supplement)"
    )
    supplement_card_types: list[str] = Field(description="Типы карточек, у которых индексируются дополнения")
    skip_cross_card_duplicates: bool = Field(description="Одинаковый sha256 в разных карточках — один раз")


class StatusRuleSettings(StrictModel):
    """Правило статуса документа (О1): то же, что `status` в конфиге экспорт-скрипта."""

    cancelled_status_ids: list[str] = Field(description="StatusID, означающие «отменён»")
    cancelled_state_ids: list[int] = Field(description="StateID маршрута, означающие отмену")
    active_state_ids: list[int] = Field(description="StateID действующего документа")
    state_names: dict[int, str] = Field(default_factory=dict, description="StateID → имя состояния")

    def resolve(self, status_id: str | None, state_id: int | None) -> DocStatus:
        cancelled_ids = {value.lower() for value in self.cancelled_status_ids}
        if status_id and status_id.lower() in cancelled_ids:
            return "cancelled"
        if state_id in self.cancelled_state_ids:
            return "cancelled"
        if state_id in self.active_state_ids:
            return "active"
        return "draft"


class IngestSettings(StrictModel):
    extensions: list[str] = Field(min_length=1, description="Обрабатываемые расширения без точки (N2)")
    status: StatusRuleSettings
    text_layer_min_chars_per_page: int = Field(ge=0, description="Порог маршрутизатора «скан/текст»")
    text_layer_min_page_share: float = Field(
        ge=0, le=1, description="Минимальная доля страниц с текстовым слоем для нативного разбора pdf"
    )
    text_layer_max_mixed_script_share: float = Field(
        ge=0,
        le=1,
        description="Доля слов со смесью кириллицы и латиницы, выше которой слой считается мусорным",
    )
    parse_cache: bool = Field(description="Кэшировать разобранные DoclingDocument в work_dir (JSON)")
    vlm: VlmSettings
    docling: DoclingSettings
    files: FileRulesSettings
    chunking: ChunkingSettings
    parent_level: Literal["section", "document"] = Field(description="Уровень parent-чанка")
    tables_as_separate_chunks: bool
    glossary_headings: list[str] = Field(min_length=1, description="Заголовки разделов глоссария")


class RetrievalSettings(StrictModel):
    prefetch_limit: int = Field(gt=0, description="Кандидатов по каждому вектору до RRF")
    rerank_candidates: int = Field(gt=0, description="Сколько кандидатов после RRF идёт в reranker")
    top_k: int = Field(gt=0, description="Результатов агенту по умолчанию")
    max_top_k: int = Field(gt=0, description="Верхняя граница top_k в запросе инструмента")
    default_statuses: list[DocStatus] = Field(min_length=1, description="Фильтр статуса по умолчанию")
    return_parent: bool = Field(description="Возвращать родительский раздел вместе с child-чанком")
    known_values_limit: int = Field(
        gt=0, description="Сколько различных значений вида документа и подразделения читать из индекса"
    )
    content_max_tokens: int = Field(
        gt=0, description="get_document_content: бюджет одного ответа в токенах (токены разделов из инжеста)"
    )
    content_max_sections: int = Field(
        gt=0, description="get_document_content: не больше разделов за один ответ"
    )

    @model_validator(mode="after")
    def _limits_are_nested(self) -> RetrievalSettings:
        if not (self.top_k <= self.max_top_k <= self.rerank_candidates <= self.prefetch_limit):
            raise ValueError(
                "retrieval: должно выполняться top_k ≤ max_top_k ≤ rerank_candidates ≤ prefetch_limit"
            )
        return self


LlmRole = Literal["rewrite", "tool_loop", "answer", "summary"]


class SamplingSettings(StrictModel):
    temperature: float = Field(ge=0)
    top_p: float = Field(gt=0, le=1)
    top_k: int = Field(ge=0, description="top_k vLLM (в OpenAI API его нет, уходит в тело запроса)")


class LlmSettings(StrictModel):
    max_tokens: int = Field(gt=0, description="Лимит генерации без размышлений")
    thinking_max_tokens: int = Field(gt=0, description="Лимит генерации с размышлениями (они входят в лимит)")
    timeout_s: float = Field(gt=0)
    max_retries: int = Field(ge=0, description="Повторы запроса к vLLM при сетевой ошибке")
    sampling: SamplingSettings = Field(description="Сэмплинг без размышлений (рекомендация Qwen)")
    thinking_sampling: SamplingSettings = Field(description="Сэмплинг с размышлениями (рекомендация Qwen)")


class ThinkingSettings(StrictModel):
    """Режим размышлений Qwen3 по ролям LLM агента (N9, решение заказчика 2026-09-16)."""

    rewrite: bool = Field(description="Разбор и переписывание запроса с учётом истории")
    tool_loop: bool = Field(description="Цикл выбора и вызова инструментов")
    answer: bool = Field(description="Итоговый ответ с самопроверкой и цитатами")
    summary: bool = Field(description="Суммаризация старых сообщений диалога")

    def enabled(self, role: LlmRole) -> bool:
        return bool(getattr(self, role))


class LlmRequestOptions(StrictModel):
    """Параметры одного запроса к vLLM для роли агента."""

    temperature: float
    top_p: float
    top_k: int
    max_tokens: int
    enable_thinking: bool

    def chat_template_kwargs(self) -> dict[str, bool]:
        return {"enable_thinking": self.enable_thinking}


class MemorySettings(StrictModel):
    buffer_messages: int = Field(gt=0, description="Сколько последних сообщений передаётся агенту")
    summary_trigger_tokens: int = Field(
        gt=0, description="Порог, после которого старые сообщения суммаризируются"
    )


class AgentSettings(StrictModel):
    llm: LlmSettings
    thinking: ThinkingSettings
    max_tool_calls: int = Field(gt=0, description="Бюджет вызовов инструментов на запрос (FR-1)")
    memory: MemorySettings
    session_document_cache: int = Field(ge=0, description="Кэш найденных документов в сессии (FR-6)")
    rewrite_query: bool = Field(description="Переписывать запрос с учётом истории и глоссария")

    def llm_options(self, role: LlmRole) -> LlmRequestOptions:
        """Сэмплинг, лимит и режим размышлений для роли — из `llm` и `thinking`."""
        thinking = self.thinking.enabled(role)
        sampling = self.llm.thinking_sampling if thinking else self.llm.sampling
        return LlmRequestOptions(
            temperature=sampling.temperature,
            top_p=sampling.top_p,
            top_k=sampling.top_k,
            max_tokens=self.llm.thinking_max_tokens if thinking else self.llm.max_tokens,
            enable_thinking=thinking,
        )


class McpSettings(StrictModel):
    host: str
    port: int = Field(ge=1, le=65535)
    path: str = Field(pattern=r"^/", description="Путь streamable-http endpoint")


class CardServiceSettings(StrictModel):
    """Клиент сервиса карточек (FR-2.2, FR-2.4): один URL из окружения, параметры здесь."""

    timeout_s: float = Field(gt=0, description="Тайм-аут запроса к сервису карточек")
    cache_size: int = Field(ge=0, description="Кэш карточек в памяти (для типа входящих связей, N7)")
    default_sections: list[str] = Field(
        min_length=1,
        description="get_document_card: секции карточки в ответе по умолчанию (N21); full=true отдаёт все",
    )


class MockCardServiceSettings(StrictModel):
    """Мок сервиса карточек в dev (§9 ТЗ): те же маршруты и схемы, данные из архива экспорта."""

    host: str
    port: int = Field(ge=1, le=65535, description="Порт мока; тот же, что у реального сервиса по умолчанию")
    ref_type_view: str = Field(description="Алиас представления справочника типов связей")
    ref_type_columns: list[str] = Field(
        min_length=3, description="Колонки представления типов связей: id, прямое и обратное имя"
    )


class UiSettings(StrictModel):
    host: str
    port: int = Field(ge=1, le=65535)
    title: str


class LangfuseSettings(StrictModel):
    flush_interval_s: float = Field(gt=0)
    environment: str


class JudgeSettings(StrictModel):
    temperature: float = Field(ge=0)
    max_tokens: int = Field(gt=0)


class EvalSettings(StrictModel):
    golden_set: Path
    reports_dir: Path
    first_signal_budget_s: float = Field(gt=0, description="NFR-2 / M6: первый сигнал, секунды")
    judge: JudgeSettings

    @property
    def golden_set_absolute(self) -> Path:
        return _absolute(self.golden_set)

    @property
    def reports_dir_absolute(self) -> Path:
        return _absolute(self.reports_dir)


class LoggingSettings(StrictModel):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"]


class AppConfig(StrictModel):
    gpu: GpuSettings
    vllm: VllmSettings
    models: ModelsSettings
    paths: PathsSettings
    embedding: EmbeddingSettings
    reranker: RerankerSettings
    qdrant: QdrantSettings
    ingest: IngestSettings
    retrieval: RetrievalSettings
    agent: AgentSettings
    mcp: McpSettings
    card_service: CardServiceSettings
    mock_card_service: MockCardServiceSettings
    ui: UiSettings
    langfuse: LangfuseSettings
    eval: EvalSettings
    logging: LoggingSettings


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
