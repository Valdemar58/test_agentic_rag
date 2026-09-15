"""Прогон инжеста по корпусу (FR-3, §4 ТЗ): инкрементально по хэшу, с реестром и уборкой.

Для каждого файла из плана (правило файлов О5):
- пропущенный правилом — записывается как `skipped`, его точки (если были) удаляются;
- sha256 и отпечаток метаданных совпадают с реестром и прошлый статус `indexed` — `unchanged`,
  файл не разбирается заново;
- sha256 тот же, но карточка изменилась (например, приказ отменён) — обновляется только payload
  точек в Qdrant, без повторного разбора и эмбеддингов;
- иначе — полный конвейер (`IngestPipeline.process_file`), ошибка одного файла не прерывает прогон.
После обхода удаляются точки и записи реестра файлов, исчезнувших из корпуса, и точки Qdrant,
о которых реестр ничего не знает. Итоги прогона — в `ingest_run`.
"""

from __future__ import annotations

import hashlib
import logging
import time
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field

from common.config import AppConfig
from ingest.corpus import Corpus, CorpusIssue
from ingest.files import FilePlan, plan_corpus_files
from ingest.index import ChunkIndex
from ingest.metadata import DocumentMetadata, file_metadata
from ingest.pipeline import FileOutcome, IngestPipeline
from ingest.registry import FileKey, FileRecordInput, IndexedFileLike, Registry, RunCounters

logger = logging.getLogger(__name__)

UNCHANGED = "unchanged"
METADATA_ONLY = "обновлены метаданные карточки без повторного разбора"


@dataclass
class RunReport:
    run_id: int
    outcome: str = "running"
    files_total: int = 0
    indexed: int = 0
    unchanged: int = 0
    skipped: int = 0
    failed: int = 0
    removed: int = 0
    chunks_total: int = 0
    seconds: float = 0.0
    outcomes: list[FileOutcome] = field(default_factory=list)
    removed_files: list[str] = field(default_factory=list)
    issues: list[CorpusIssue] = field(default_factory=list)
    error_text: str | None = None

    @property
    def counters(self) -> RunCounters:
        return RunCounters(
            files_total=self.files_total,
            files_indexed=self.indexed,
            files_skipped=self.skipped,
            files_failed=self.failed,
            files_unchanged=self.unchanged,
            chunks_total=self.chunks_total,
        )

    @property
    def denominator(self) -> int:
        """Файлы, которые инжест пытался обработать (AC-3.1/M8): без пропущенных правилом."""
        return self.indexed + self.unchanged + self.failed

    @property
    def success_share(self) -> float:
        return (self.indexed + self.unchanged) / self.denominator if self.denominator else 1.0

    def summary_lines(self) -> list[str]:
        by_reason = Counter(
            (outcome.reason or "").split(":")[0] for outcome in self.outcomes if outcome.status == "error"
        )
        lines = [
            f"Прогон #{self.run_id}: {self.outcome}, {self.seconds:.0f} с",
            f"Файлов в плане: {self.files_total}; проиндексировано {self.indexed}, "
            f"без изменений {self.unchanged}, пропущено правилом {self.skipped}, "
            f"с ошибкой {self.failed}, удалено исчезнувших {self.removed}",
            f"Чанков в индексе по реестру: {self.chunks_total}",
            f"Доля успешно обработанных (AC-3.1): {self.success_share:.1%} из {self.denominator}",
        ]
        if self.issues:
            lines.append(f"Не допущено загрузчиком архива: {len(self.issues)}")
        for reason, count in by_reason.most_common(5):
            lines.append(f"  ошибка ×{count}: {reason}")
        return lines


def metadata_fingerprint(document: DocumentMetadata, plan: FilePlan) -> str:
    """Отпечаток всего, что попадает в payload помимо текста: метаданные карточки и роль файла."""
    file = file_metadata(plan, "-")
    payload = document.model_dump_json() + file.model_dump_json(exclude={"parse_route"})
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class IngestRunner:
    def __init__(
        self,
        config: AppConfig,
        *,
        corpus: Corpus,
        pipeline: IngestPipeline,
        index: ChunkIndex,
        registry: Registry,
        force: bool = False,
    ) -> None:
        self._config = config
        self._corpus = corpus
        self._pipeline = pipeline
        self._index = index
        self._registry = registry
        self._force = force

    async def run(self) -> RunReport:
        started = time.perf_counter()
        corpus = self._corpus
        run_id = await self._registry.start_run(str(corpus.export_dir), synthetic=corpus.synthetic)
        report = RunReport(run_id=run_id, issues=list(corpus.issues))
        try:
            await self._process(report)
            report.outcome = "success" if report.failed == 0 else "partial"
        except Exception as exc:  # noqa: BLE001 — итог прогона фиксируется в реестре, потом исключение наверх
            report.outcome = "failed"
            report.error_text = f"{type(exc).__name__}: {exc}"
            report.seconds = time.perf_counter() - started
            await self._registry.finish_run(
                run_id, report.outcome, report.counters, error_text=report.error_text
            )
            raise
        report.seconds = time.perf_counter() - started
        await self._registry.finish_run(run_id, report.outcome, report.counters, error_text=report.error_text)
        return report

    async def _process(self, report: RunReport) -> None:
        corpus = self._corpus
        plans = plan_corpus_files(corpus, self._config.ingest)
        known = await self._registry.load()
        seen: set[FileKey] = set()
        self._index.ensure_collections()
        total = sum(len(card_plans) for card_plans in plans.values())
        report.files_total = total
        position = 0
        for document in corpus.documents:
            metadata = self._pipeline.document_metadata(corpus, document)
            for plan in plans[document.card_id]:
                position += 1
                key = (plan.file.card_id, plan.file.row_id)
                seen.add(key)
                previous = known.get(key)
                fingerprint = metadata_fingerprint(metadata, plan)
                logger.info("[%d/%d] %s", position, total, plan.file.relative_path)
                if not plan.indexed:
                    if previous is not None and previous.status == "indexed":
                        self._index.delete_file(plan.file.row_id)
                    report.skipped += 1
                    await self._record(
                        report.run_id, plan, "skipped", plan.skip_reason, metadata, fingerprint
                    )
                    continue
                if (
                    not self._force
                    and previous is not None
                    and previous.status == "indexed"
                    and previous.sha256 == plan.file.sha256
                ):
                    if previous.metadata_sha256 == fingerprint:
                        report.unchanged += 1
                        report.chunks_total += previous.chunk_count
                        await self._record(
                            report.run_id,
                            plan,
                            "indexed",
                            None,
                            metadata,
                            fingerprint,
                            chunk_count=previous.chunk_count,
                            file_role=previous.file_role,
                            parse_route=previous.parse_route,
                        )
                        continue
                    self._index.set_payload(
                        plan.file.row_id, self._payload_fields(metadata, plan, previous.parse_route)
                    )
                    report.indexed += 1
                    report.chunks_total += previous.chunk_count
                    await self._record(
                        report.run_id,
                        plan,
                        "indexed",
                        METADATA_ONLY,
                        metadata,
                        fingerprint,
                        chunk_count=previous.chunk_count,
                        file_role=previous.file_role,
                        parse_route=previous.parse_route,
                    )
                    continue
                outcome = self._pipeline.process_file(metadata, plan)
                report.outcomes.append(outcome)
                if outcome.indexed:
                    report.indexed += 1
                    report.chunks_total += outcome.chunks
                else:
                    report.failed += 1
                await self._record(
                    report.run_id,
                    plan,
                    outcome.status,
                    outcome.reason,
                    metadata,
                    fingerprint,
                    chunk_count=outcome.chunks,
                    file_role=plan.role,
                    parse_route=outcome.route,
                )
        await self._remove_vanished(report, known, seen)

    def _payload_fields(
        self, metadata: DocumentMetadata, plan: FilePlan, parse_route: str | None
    ) -> dict[str, object]:
        file = file_metadata(plan, parse_route or "-")
        fields: dict[str, object] = metadata.model_dump(mode="json")
        fields.update(file.model_dump(mode="json"))
        return fields

    async def _record(
        self,
        run_id: int,
        plan: FilePlan,
        status: str,
        reason: str | None,
        metadata: DocumentMetadata,
        fingerprint: str,
        *,
        chunk_count: int = 0,
        file_role: str | None = None,
        parse_route: str | None = None,
    ) -> None:
        await self._registry.record(
            run_id,
            [
                FileRecordInput(
                    card_id=plan.file.card_id,
                    file_row_id=plan.file.row_id,
                    relative_path=plan.file.relative_path,
                    sha256=plan.file.sha256,
                    size_bytes=plan.file.size,
                    extension=plan.file.extension,
                    status=status,
                    reason=reason,
                    file_role=file_role or plan.role,
                    parse_route=parse_route,
                    chunk_count=chunk_count,
                    doc_status=metadata.doc_status,
                    metadata_sha256=fingerprint,
                )
            ],
        )

    async def _remove_vanished(
        self, report: RunReport, known: Mapping[FileKey, IndexedFileLike], seen: set[FileKey]
    ) -> None:
        vanished = [key for key in known if key not in seen]
        for card_id, row_id in vanished:
            self._index.delete_file(row_id)
            report.removed_files.append(f"{card_id}/{row_id}")
        if vanished:
            report.removed += await self._registry.remove(vanished)
        # точки, о которых реестр не знает (например, реестр очищен) — тоже лишние
        known_rows = {str(row_id) for _, row_id in seen}
        for orphan in self._index.file_row_ids() - known_rows:
            self._index.delete_file(orphan)
            report.removed += 1
            report.removed_files.append(f"?/{orphan}")
