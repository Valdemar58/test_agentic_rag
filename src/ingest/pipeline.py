"""Конвейер инжеста одного файла (FR-3): маршрут → разбор → чанки → parent → метаданные → эмбеддинги → Qdrant.

Ошибка любого шага фиксируется в `FileOutcome` и не прерывает инжест остальных файлов (AC-3.1).
Запись в Qdrant идемпотентна: точки файла удаляются перед upsert (AC-3.3). Инкрементальность по
хэшу и удаление исчезнувших файлов — в `ingest.run` (задача 4.7), оркестрация GPU-профилей —
в `scripts/run_ingest.py` (задача 4.8).
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
    DocumentMetadata,
    child_payload,
    document_metadata,
    file_metadata,
    parent_payload,
    root_crumbs,
)
from ingest.parents import build_chunk_set
from ingest.parsing import Parser
from ingest.router import choose_route
from ingest.tokens import TokenCounter

logger = logging.getLogger(__name__)


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

    def process_file(self, document: DocumentMetadata, plan: FilePlan) -> FileOutcome:
        """Полный путь одного файла; любое исключение превращается в `FileOutcome(status="error")`."""
        started = time.perf_counter()
        if not plan.indexed:
            return FileOutcome(plan=plan, status="skipped", reason=plan.skip_reason)
        try:
            decision = choose_route(plan.file, self._config.ingest)
            result = self._parser.parse(plan.file.path, decision.route)
            if not result.ok:
                return FileOutcome(
                    plan=plan,
                    status="error",
                    reason=f"разбор ({decision.route}): {result.status}; "
                    + ("; ".join(result.errors) or "без деталей"),
                    route=decision.route,
                    seconds=time.perf_counter() - started,
                    parse_errors=result.errors,
                )
            chunks = self._chunker.chunk(
                result.document, root_crumbs(document, plan), file_sha256=plan.file.sha256
            )
            if not chunks:
                return FileOutcome(
                    plan=plan,
                    status="error",
                    reason=f"разбор ({decision.route}) не дал текста",
                    route=decision.route,
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
            file = file_metadata(plan, decision.route)
            separator = self._config.ingest.chunking.breadcrumb_separator
            children = [child_payload(document, file, chunk, separator) for chunk in chunk_set.children]
            parents = [parent_payload(document, file, parent, separator) for parent in chunk_set.parents]
            embeddings = self._embedder.encode([child.text for child in children])
            self._index.delete_file(plan.file.row_id)
            self._index.upsert(children, embeddings, parents)
        except Exception as exc:  # noqa: BLE001 — один файл не должен ронять инжест (AC-3.1)
            logger.exception("Файл %s не проиндексирован", plan.file.relative_path)
            return FileOutcome(
                plan=plan,
                status="error",
                reason=f"{type(exc).__name__}: {exc}",
                seconds=time.perf_counter() - started,
            )
        seconds = time.perf_counter() - started
        logger.info(
            "Проиндексирован %s (%s): чанков %d, разделов %d, %.1f с",
            plan.file.relative_path,
            decision.route,
            len(children),
            len(parents),
            seconds,
        )
        return FileOutcome(
            plan=plan,
            status="indexed",
            route=decision.route,
            chunks=len(children),
            parents=len(parents),
            pages=result.pages,
            seconds=seconds,
            parse_errors=result.errors,
        )
