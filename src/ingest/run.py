"""Прогон инжеста по корпусу (FR-3, §2, §4 ТЗ): две фазы под бюджет VRAM, инкрементально, с реестром.

Фаза 1 «разбор» (профиль ingest поднят, dots.mocr занимает GPU): для каждого файла плана —
пропуск правилом (`skipped`, старые точки удаляются), `unchanged` (sha256 и отпечаток метаданных
совпадают с реестром), обновление только payload (файл тот же, карточка изменилась) или разбор
и чанкинг (`PreparedFile`). Между фазами вызывается `before_index` — там оркестратор выгружает
VLM. Фаза 2 «эмбеддинги и запись»: bge-m3 на освободившемся GPU, upsert в Qdrant, записи реестра.
В конце удаляются точки и записи файлов, исчезнувших из корпуса, и точки без записи в реестре.
Ошибка одного файла не прерывает прогон; итоги — в `ingest_run`.
"""

from __future__ import annotations

import hashlib
import logging
import time
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from common.config import AppConfig
from ingest.corpus import Corpus, CorpusIssue
from ingest.files import FilePlan, plan_corpus_files
from ingest.index import ChunkIndex
from ingest.metadata import DocumentMetadata, file_metadata
from ingest.pipeline import FileOutcome, IngestPipeline, PreparedFile
from ingest.registry import FileKey, FileRecordInput, IndexedFileLike, Registry, RunCounters
from ingest.router import RouteDecision, choose_route

logger = logging.getLogger(__name__)

METADATA_ONLY = "обновлены метаданные карточки без повторного разбора"
BeforeIndexHook = Callable[[], None]


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
    parse_seconds: float = 0.0
    index_seconds: float = 0.0
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
            f"Прогон #{self.run_id}: {self.outcome}, {self.seconds:.0f} с "
            f"(разбор {self.parse_seconds:.0f} с, эмбеддинги и запись {self.index_seconds:.0f} с)",
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


@dataclass(frozen=True)
class CorpusChange:
    """Прогон указывает на другой каталог корпуса, чем предыдущий: индекс будет частично очищен."""

    previous: str
    current: str
    removed_files: int

    def message(self) -> str:
        return (
            f"СМЕНА КОРПУСА: прошлый прогон индексировал {self.previous}, сейчас указан {self.current}.\n"
            f"Файлов в индексе, которых нет в новом корпусе: {self.removed_files} — они будут удалены.\n"
            "Если это намеренно, повторите команду с --switch-corpus; "
            f"если нет — укажите --corpus {self.previous}."
        )


@dataclass(frozen=True)
class WorkPlan:
    """Оценка объёма до запуска: нужна ли VLM и сколько файлов ждёт разбора."""

    files_total: int
    to_parse: int
    vlm_files: int
    unchanged: int
    skipped: int

    @property
    def needs_vlm(self) -> bool:
        return self.vlm_files > 0

    def summary(self) -> str:
        return (
            f"файлов в плане {self.files_total}: разобрать {self.to_parse} "
            f"(из них через dots.mocr {self.vlm_files}), без изменений {self.unchanged}, "
            f"пропущено правилом {self.skipped}"
        )


def _same_path(first: str, second: str) -> bool:
    """Сравнение каталогов с учётом относительных путей; несуществующий путь сравнивается как текст."""
    try:
        return Path(first).resolve() == Path(second).resolve()
    except OSError:
        return first == second


def metadata_fingerprint(document: DocumentMetadata, plan: FilePlan) -> str:
    """Отпечаток всего, что попадает в payload помимо текста: метаданные карточки и роль файла."""
    file = file_metadata(plan, "-")
    payload = document.model_dump_json() + file.model_dump_json(exclude={"parse_route"})
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class _Work:
    metadata: DocumentMetadata
    plan: FilePlan
    fingerprint: str
    previous: IndexedFileLike | None
    decision: RouteDecision | None

    @property
    def key(self) -> FileKey:
        return (self.plan.file.card_id, self.plan.file.row_id)

    @property
    def unchanged(self) -> bool:
        return (
            self.previous is not None
            and self.previous.status == "indexed"
            and self.previous.sha256 == self.plan.file.sha256
        )


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
        self._work: list[_Work] | None = None
        self._known: Mapping[FileKey, IndexedFileLike] = {}

    async def _collect(self) -> list[_Work]:
        if self._work is None:
            corpus = self._corpus
            plans = plan_corpus_files(corpus, self._config.ingest)
            self._known = await self._registry.load()
            work: list[_Work] = []
            for document in corpus.documents:
                metadata = self._pipeline.document_metadata(corpus, document)
                for plan in plans[document.card_id]:
                    key = (plan.file.card_id, plan.file.row_id)
                    item = _Work(
                        metadata, plan, metadata_fingerprint(metadata, plan), self._known.get(key), None
                    )
                    needs_parse = plan.indexed and (self._force or not item.unchanged)
                    decision = choose_route(plan.file, self._config.ingest) if needs_parse else None
                    work.append(_Work(metadata, plan, item.fingerprint, item.previous, decision))
            self._work = work
        return self._work

    async def corpus_change(self) -> CorpusChange | None:
        """Сверяет каталог корпуса с прошлым прогоном: None — тот же корпус или индекс пуст."""
        previous = await self._registry.last_corpus_path()
        current = str(self._corpus.export_dir)
        if previous is None or _same_path(previous, current):
            return None
        work = await self._collect()
        seen = {item.key for item in work}
        return CorpusChange(previous, current, sum(key not in seen for key in self._known))

    async def preflight(self) -> WorkPlan:
        work = await self._collect()
        to_parse = [item for item in work if item.decision is not None]
        return WorkPlan(
            files_total=len(work),
            to_parse=len(to_parse),
            vlm_files=sum(item.decision is not None and item.decision.route == "vlm" for item in to_parse),
            unchanged=sum(item.plan.indexed and item.decision is None for item in work),
            skipped=sum(not item.plan.indexed for item in work),
        )

    async def run(self, *, before_index: BeforeIndexHook | None = None) -> RunReport:
        started = time.perf_counter()
        corpus = self._corpus
        run_id = await self._registry.start_run(str(corpus.export_dir), synthetic=corpus.synthetic)
        report = RunReport(run_id=run_id, issues=list(corpus.issues))
        try:
            await self._process(report, before_index)
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

    async def _process(self, report: RunReport, before_index: BeforeIndexHook | None) -> None:
        work = await self._collect()
        report.files_total = len(work)
        self._index.ensure_collections()

        # ---- фаза 1: разбор (VLM поднята) ----
        phase_started = time.perf_counter()
        prepared: list[tuple[_Work, PreparedFile]] = []
        to_parse = sum(item.decision is not None for item in work)
        logger.info("Фаза 1/2: разбор — файлов к разбору %d из %d", to_parse, len(work))
        position = 0
        for item in work:
            plan, previous = item.plan, item.previous
            if not plan.indexed:
                if previous is not None and previous.status == "indexed":
                    self._index.delete_file(plan.file.row_id)
                report.skipped += 1
                await self._record(report.run_id, item, "skipped", plan.skip_reason)
                continue
            if item.decision is None:
                assert previous is not None  # noqa: S101 — unchanged означает запись в реестре
                if previous.metadata_sha256 == item.fingerprint:
                    report.unchanged += 1
                else:
                    self._index.set_payload(
                        plan.file.row_id, self._payload_fields(item, previous.parse_route)
                    )
                    report.indexed += 1
                report.chunks_total += previous.chunk_count
                await self._record(
                    report.run_id,
                    item,
                    "indexed",
                    None if previous.metadata_sha256 == item.fingerprint else METADATA_ONLY,
                    chunk_count=previous.chunk_count,
                    file_role=previous.file_role,
                    parse_route=previous.parse_route,
                )
                continue
            position += 1
            logger.info("[%d/%d] %s", position, to_parse, plan.file.relative_path)
            result = self._pipeline.prepare_file(item.metadata, plan, item.decision)
            if isinstance(result, FileOutcome):
                report.failed += 1
                report.outcomes.append(result)
                await self._record(
                    report.run_id, item, result.status, result.reason, parse_route=result.route
                )
                continue
            prepared.append((item, result))
        report.parse_seconds = time.perf_counter() - phase_started

        # ---- между фазами: выгрузка VLM, GPU освобождается под эмбеддинги ----
        if before_index is not None:
            before_index()

        # ---- фаза 2: эмбеддинги и запись ----
        phase_started = time.perf_counter()
        logger.info("Фаза 2/2: эмбеддинги и запись — файлов %d", len(prepared))
        for position, (item, ready) in enumerate(prepared, start=1):
            logger.info("[%d/%d] %s", position, len(prepared), item.plan.file.relative_path)
            outcome = self._pipeline.index_prepared(ready)
            report.outcomes.append(outcome)
            if outcome.indexed:
                report.indexed += 1
                report.chunks_total += outcome.chunks
            else:
                report.failed += 1
            await self._record(
                report.run_id,
                item,
                outcome.status,
                outcome.reason,
                chunk_count=outcome.chunks,
                file_role=item.plan.role,
                parse_route=outcome.route,
            )
        report.index_seconds = time.perf_counter() - phase_started
        await self._remove_vanished(report, {item.key for item in work})

    def _payload_fields(self, item: _Work, parse_route: str | None) -> dict[str, object]:
        file = file_metadata(item.plan, parse_route or "-")
        fields: dict[str, object] = item.metadata.model_dump(mode="json")
        fields.update(file.model_dump(mode="json"))
        return fields

    async def _record(
        self,
        run_id: int,
        item: _Work,
        status: str,
        reason: str | None,
        *,
        chunk_count: int = 0,
        file_role: str | None = None,
        parse_route: str | None = None,
    ) -> None:
        plan = item.plan
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
                    doc_status=item.metadata.doc_status,
                    metadata_sha256=item.fingerprint,
                )
            ],
        )

    async def _remove_vanished(self, report: RunReport, seen: set[FileKey]) -> None:
        vanished = [key for key in self._known if key not in seen]
        for card_id, row_id in vanished:
            self._index.delete_file(row_id)
            report.removed_files.append(f"{card_id}/{row_id}")
        if vanished:
            report.removed += await self._registry.remove(vanished)
        known_rows = {str(row_id) for _, row_id in seen}
        for orphan in self._index.file_row_ids() - known_rows:
            self._index.delete_file(orphan)
            report.removed += 1
            report.removed_files.append(f"?/{orphan}")
