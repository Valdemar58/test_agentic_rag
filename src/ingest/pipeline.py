"""Конвейер одного файла (FR-3), разделённый на две фазы под бюджет VRAM (§2 ТЗ).

`prepare_file`: маршрут → разбор (нативный Docling или dots.mocr, пока поднят профиль ingest) →
чанки → parent → payload; эмбеддингов ещё нет. `index_prepared`: эмбеддинги bge-m3 (на GPU уже
после выгрузки VLM) → удаление старых точек файла → upsert (AC-3.3). `process_file` = обе фазы
подряд (тесты и одиночные прогоны). Ошибка любого шага фиксируется в `FileOutcome` и не прерывает
инжест остальных файлов (AC-3.1).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from common.config import AppConfig
from ingest.chunking import StructuralChunker
from ingest.corpus import Corpus, CorpusDocument
from ingest.embeddings import Embedder
from ingest.files import FilePlan
from ingest.index import ChunkIndex
from ingest.metadata import (
    ChunkPayload,
    DocumentMetadata,
    child_payload,
    document_metadata,
    file_metadata,
    parent_payload,
    root_crumbs,
)
from ingest.parents import build_chunk_set
from ingest.parsing import Parser
from ingest.router import RouteDecision, choose_route
from ingest.tokens import TokenCounter

logger = logging.getLogger(__name__)

PDF_EXTENSION = "pdf"


@dataclass(frozen=True)
class FileOutcome:
    plan: FilePlan
    status: str
    reason: str | None = None
    route: str | None = None
    chunks: int = 0
    parents: int = 0
    pages: int = 0
    seconds: float = 0.0
    parse_errors: list[str] = field(default_factory=list)

    @property
    def indexed(self) -> bool:
        return self.status == "indexed"


@dataclass(frozen=True)
class PreparedFile:
    """Результат первой фазы: чанки с payload, готовые к эмбеддингу и записи."""

    plan: FilePlan
    route: str
    children: list[ChunkPayload]
    parents: list[ChunkPayload]
    pages: int
    seconds: float
    parse_errors: list[str] = field(default_factory=list)


class IngestPipeline:
    def __init__(
        self,
        config: AppConfig,
        *,
        parser: Parser,
        embedder: Embedder,
        index: ChunkIndex,
        counter: TokenCounter,
    ) -> None:
        self._config = config
        self._parser = parser
        self._embedder = embedder
        self._index = index
        self._chunker = StructuralChunker(config.ingest.chunking, counter)
        self._counter = counter

    def document_metadata(self, corpus: Corpus, document: CorpusDocument) -> DocumentMetadata:
        return document_metadata(document, corpus.links_graph, self._config.ingest.status)

    def prepare_file(
        self, document: DocumentMetadata, plan: FilePlan, decision: RouteDecision | None = None
    ) -> PreparedFile | FileOutcome:
        """Фаза 1: разбор и чанкинг; при ошибке — `FileOutcome`, иначе `PreparedFile`."""
        started = time.perf_counter()
        if not plan.indexed:
            return FileOutcome(plan=plan, status="skipped", reason=plan.skip_reason)
        try:
            route = decision or choose_route(plan.file, self._config.ingest)
            result = self._parser.parse(plan.file.path, route.route)
            chunks = (
                self._chunker.chunk(
                    result.document, root_crumbs(document, plan), file_sha256=plan.file.sha256
                )
                if result.ok
                else []
            )
            if not chunks and route.route == "native" and plan.file.extension == PDF_EXTENSION:
                # текстовый слой был, а текста нет (пустой или мусорный слой): пробуем распознать страницы
                logger.warning("Нативный разбор %s не дал текста — пробую dots.mocr", plan.file.relative_path)
                route = RouteDecision(
                    "vlm", f"{route.reason}; нативный разбор без текста → dots.mocr", route.text_layer
                )
                result = self._parser.parse(plan.file.path, route.route)
                chunks = (
                    self._chunker.chunk(
                        result.document, root_crumbs(document, plan), file_sha256=plan.file.sha256
                    )
                    if result.ok
                    else []
                )
            if not result.ok:
                details = "; ".join(result.errors) or "без деталей"
                return FileOutcome(
                    plan=plan,
                    status="error",
                    reason=f"разбор ({route.route}): {result.status}; {details}",
                    route=route.route,
                    seconds=time.perf_counter() - started,
                    parse_errors=result.errors,
                )
            if not chunks:
                return FileOutcome(
                    plan=plan,
                    status="error",
                    reason=f"разбор ({route.route}) не дал текста",
                    route=route.route,
                    pages=result.pages,
                    seconds=time.perf_counter() - started,
                )
            chunk_set = build_chunk_set(
                chunks,
                self._config.ingest.chunking,
                self._counter,
                file_sha256=plan.file.sha256,
                level=self._config.ingest.parent_level,
            )
            file = file_metadata(plan, route.route)
            separator = self._config.ingest.chunking.breadcrumb_separator
            children = [child_payload(document, file, chunk, separator) for chunk in chunk_set.children]
            parents = [parent_payload(document, file, parent, separator) for parent in chunk_set.parents]
        except Exception as exc:  # noqa: BLE001 — один файл не должен ронять инжест (AC-3.1)
            logger.exception("Файл %s не разобран", plan.file.relative_path)
            return FileOutcome(
                plan=plan,
                status="error",
                reason=f"{type(exc).__name__}: {exc}",
                seconds=time.perf_counter() - started,
            )
        seconds = time.perf_counter() - started
        logger.info(
            "Разобран %s (%s): чанков %d, разделов %d, страниц %d, %.1f с",
            plan.file.relative_path,
            route.route,
            len(children),
            len(parents),
            result.pages,
            seconds,
        )
        return PreparedFile(
            plan=plan,
            route=route.route,
            children=children,
            parents=parents,
            pages=result.pages,
            seconds=seconds,
            parse_errors=result.errors,
        )

    def index_prepared(self, prepared: PreparedFile) -> FileOutcome:
        """Фаза 2: эмбеддинги child-чанков и идемпотентная запись в Qdrant."""
        started = time.perf_counter()
        try:
            embeddings = self._embedder.encode([child.text for child in prepared.children])
            self._index.delete_file(prepared.plan.file.row_id)
            self._index.upsert(prepared.children, embeddings, prepared.parents)
        except Exception as exc:  # noqa: BLE001 — один файл не должен ронять инжест (AC-3.1)
            logger.exception("Файл %s не записан в индекс", prepared.plan.file.relative_path)
            return FileOutcome(
                plan=prepared.plan,
                status="error",
                reason=f"{type(exc).__name__}: {exc}",
                route=prepared.route,
                pages=prepared.pages,
                seconds=prepared.seconds + time.perf_counter() - started,
                parse_errors=prepared.parse_errors,
            )
        seconds = prepared.seconds + time.perf_counter() - started
        logger.info(
            "Проиндексирован %s: чанков %d, разделов %d, %.1f с",
            prepared.plan.file.relative_path,
            len(prepared.children),
            len(prepared.parents),
            seconds,
        )
        return FileOutcome(
            plan=prepared.plan,
            status="indexed",
            route=prepared.route,
            chunks=len(prepared.children),
            parents=len(prepared.parents),
            pages=prepared.pages,
            seconds=seconds,
            parse_errors=prepared.parse_errors,
        )

    def process_file(self, document: DocumentMetadata, plan: FilePlan) -> FileOutcome:
        """Обе фазы подряд для одного файла."""
        prepared = self.prepare_file(document, plan)
        if isinstance(prepared, FileOutcome):
            return prepared
        return self.index_prepared(prepared)
