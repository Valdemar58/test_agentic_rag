"""Инкрементальный прогон (4.7): без изменений — не разбирает; изменился файл — переразбор; изменилась
карточка — только payload; исчез файл — точки и запись удалены; пропуски и ошибки — в отчёте."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest
import structlog
from docling_core.types.doc.document import DoclingDocument
from docling_core.types.doc.labels import DocItemLabel
from qdrant_client import QdrantClient, models

from common.config import DEFAULT_CONFIG_PATH, load_app_config
from ingest.corpus import load_corpus
from ingest.embeddings import FakeEmbedder
from ingest.index import ChunkIndex
from ingest.parsing import ParseResult
from ingest.pipeline import IngestPipeline
from ingest.registry import FileKey, FileRecordInput, RunCounters
from ingest.router import ParseRoute
from ingest.run import METADATA_ONLY, IngestRunner, RunReport
from ingest.tokens import WordTokenCounter
from synthetic.corpus import export_config
from tessa_export.config import CANCELLED_STATUS_ID
from tessa_export.fake import FakeGateway, link, make_file, make_snapshot, stable_uuid
from tessa_export.runner import run_export
from tessa_export.sample_files import minimal_docx_bytes, minimal_pdf_bytes

CONFIG = load_app_config(DEFAULT_CONFIG_PATH)
ORDER = stable_uuid("run", "order")
MEMO = stable_uuid("run", "memo")
# фиксированная дата: у фейка modified = doc_date, а «сейчас» меняло бы отпечаток метаданных между экспортами
DOC_DATE = datetime(2026, 1, 15, tzinfo=UTC)


@pytest.fixture(autouse=True)
def _reset_logging() -> Iterator[None]:
    yield
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()
    structlog.reset_defaults()


@dataclass
class Entry:
    sha256: str
    status: str
    metadata_sha256: str | None
    chunk_count: int
    file_role: str | None
    parse_route: str | None
    doc_status: str | None
    reason: str | None
    run_id: int


class MemoryRegistry:
    def __init__(self) -> None:
        self.rows: dict[FileKey, Entry] = {}
        self.runs: dict[int, tuple[str, RunCounters, str | None]] = {}

    async def start_run(self, corpus_path: str, *, synthetic: bool) -> int:
        run_id = len(self.runs) + 1
        self.runs[run_id] = ("running", RunCounters(), None)
        return run_id

    async def finish_run(
        self, run_id: int, outcome: str, counters: RunCounters, *, error_text: str | None = None
    ) -> None:
        self.runs[run_id] = (outcome, counters, error_text)

    async def load(self, card_ids: Iterable[UUID] | None = None) -> Mapping[FileKey, Entry]:
        return dict(self.rows)

    async def record(self, run_id: int, items: Iterable[FileRecordInput]) -> None:
        for item in items:
            self.rows[(item.card_id, item.file_row_id)] = Entry(
                item.sha256,
                item.status,
                item.metadata_sha256,
                item.chunk_count,
                item.file_role,
                item.parse_route,
                item.doc_status,
                item.reason,
                run_id,
            )

    async def remove(self, keys: Iterable[FileKey]) -> int:
        removed = 0
        for key in keys:
            removed += self.rows.pop(key, None) is not None
        return removed


class CountingParser:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def parse(self, path: Path, route: ParseRoute) -> ParseResult:
        self.calls.append(path.name)
        doc = DoclingDocument(name=path.stem)
        doc.add_heading(path.stem, level=1)
        doc.add_heading("1. Раздел", level=2)
        doc.add_text(label=DocItemLabel.TEXT, text=f"1.1. Первый пункт файла {path.stem}.")
        doc.add_text(label=DocItemLabel.TEXT, text=f"1.2. Второй пункт файла {path.stem}.")
        return ParseResult(route=route, status="success", seconds=0.01, document=doc, pages=1)


def _export(
    output: Path, *, order_text: str = "Приказ", cancelled: bool = False, with_memo: bool = True
) -> None:
    gateway = FakeGateway()
    order_files = [
        make_file(ORDER, "ДокШаблон Приказ №144.docx"),
        make_file(ORDER, "Для печати_Приказ №144.docx"),
        make_file(ORDER, "144 от 15.01.2026.pdf"),
    ]
    gateway.add(
        make_snapshot(
            ORDER,
            number="144",
            doc_date=DOC_DATE,
            status_id=CANCELLED_STATUS_ID if cancelled else None,
            status_name="Отмененный" if cancelled else None,
            state_id=6,
            outgoing=[link(MEMO, ref_type_name="Документ-основание", ref_type_reverse_name="Приказ")]
            if with_memo
            else [],
            files=order_files,
        ),
        {
            "ДокШаблон Приказ №144.docx": minimal_docx_bytes([order_text]),
            "Для печати_Приказ №144.docx": minimal_docx_bytes(["копия"]),
            "144 от 15.01.2026.pdf": minimal_pdf_bytes("Order 144"),
        },
    )
    if with_memo:
        gateway.add(
            make_snapshot(
                MEMO,
                type_name="InhouseDocumentMKC",
                type_caption="Служебная записка",
                number="СЗ-9",
                doc_date=DOC_DATE,
                status_id=None,
                status_name=None,
                files=[make_file(MEMO, "СЗ-9.docx")],
            ),
            {"СЗ-9.docx": minimal_docx_bytes(["Прошу назначить."])},
        )
    run_export(export_config(output), [ORDER], gateway, synthetic=True, source="synthetic")


def _runner(
    output: Path, parser: CountingParser, registry: MemoryRegistry, index: ChunkIndex, **kw: Any
) -> IngestRunner:
    corpus = load_corpus(output)
    pipeline = IngestPipeline(
        CONFIG, parser=parser, embedder=FakeEmbedder(), index=index, counter=WordTokenCounter()
    )
    return IngestRunner(CONFIG, corpus=corpus, pipeline=pipeline, index=index, registry=registry, **kw)


def _run(runner: IngestRunner) -> RunReport:
    return asyncio.run(runner.run())


def test_incremental_runs_reindex_only_changed_files(tmp_path: Path) -> None:
    output = tmp_path / "corpus"
    _export(output)
    parser, registry = CountingParser(), MemoryRegistry()
    index = ChunkIndex(QdrantClient(":memory:"), CONFIG.qdrant, 8)

    first = _run(_runner(output, parser, registry, index))
    # 4 файла в плане: docx приказа и записки индексируются, «Для печати» и pdf-копия — пропуск правилом
    assert (first.files_total, first.indexed, first.skipped, first.failed, first.unchanged) == (4, 2, 2, 0, 0)
    assert first.outcome == "success" and first.chunks_total == 4 and first.success_share == 1.0
    assert sorted(parser.calls) == ["ДокШаблон Приказ №144.docx", "СЗ-9.docx"]
    assert index.count() == 4 and index.count(parents=True) == 2
    assert registry.runs[1][0] == "success" and registry.runs[1][1].files_indexed == 2
    skipped = [entry for entry in registry.rows.values() if entry.status == "skipped"]
    assert len(skipped) == 2 and all(entry.reason for entry in skipped)

    # тот же корпус: ничего не разбирается, число точек не меняется (AC-3.3)
    second = _run(_runner(output, parser, registry, index))
    assert (second.indexed, second.unchanged, second.failed) == (0, 2, 0) and second.chunks_total == 4
    assert len(parser.calls) == 2 and index.count() == 4 and index.count(parents=True) == 2

    # изменился текст приказа (новый sha256) — переразбирается только он
    _export(output, order_text="Приказ в новой редакции")
    third = _run(_runner(output, parser, registry, index))
    assert (third.indexed, third.unchanged) == (1, 1) and parser.calls[-1] == "ДокШаблон Приказ №144.docx"
    assert len(parser.calls) == 3 and index.count() == 4

    # --force переразбирает всё
    forced = _run(_runner(output, parser, registry, index, force=True))
    assert forced.indexed == 2 and forced.unchanged == 0 and len(parser.calls) == 5 and index.count() == 4


def test_card_change_updates_payload_without_reparse_and_vanished_file_is_removed(tmp_path: Path) -> None:
    output = tmp_path / "corpus"
    _export(output)
    parser, registry = CountingParser(), MemoryRegistry()
    index = ChunkIndex(QdrantClient(":memory:"), CONFIG.qdrant, 8)
    _run(_runner(output, parser, registry, index))
    assert len(parser.calls) == 2

    def statuses() -> set[str]:
        points, _ = index.client.scroll(
            CONFIG.qdrant.collection, limit=100, with_payload=["doc_status", "doc_id"]
        )
        return {
            str(point.payload["doc_status"])
            for point in points
            if point.payload and point.payload["doc_id"] == str(ORDER)
        }

    assert statuses() == {"active"}  # StateID 6 (Registered) без StatusID — действующий по правилу О1
    # приказ отменён, файлы те же: payload обновлён, разбора нет
    _export(output, cancelled=True)
    report = _run(_runner(output, parser, registry, index))
    assert len(parser.calls) == 2 and report.indexed == 1 and report.unchanged == 1
    assert statuses() == {"cancelled"}
    order_key = next(
        key for key in registry.rows if key[0] == ORDER and registry.rows[key].status == "indexed"
    )
    assert registry.rows[order_key].reason == METADATA_ONLY and registry.rows[order_key].chunk_count == 2
    parents, _ = index.client.scroll(CONFIG.qdrant.parents_collection, limit=10, with_payload=["doc_status"])
    assert {str(p.payload["doc_status"]) for p in parents if p.payload} <= {"cancelled", "active"}

    # записка исчезла из корпуса: её точки и запись реестра удалены
    _export(output, cancelled=True, with_memo=False)
    report = _run(_runner(output, parser, registry, index))
    assert report.removed == 1 and any(str(MEMO) in item for item in report.removed_files)
    assert index.count() == 2 and index.count(parents=True) == 1
    assert all(key[0] != MEMO for key in registry.rows)
    assert len(parser.calls) == 2 and "Прогон #" in report.summary_lines()[0]

    # точки без записи в реестре (реестр очищен) — тоже удаляются как лишние
    index.client.upsert(
        CONFIG.qdrant.collection,
        points=[
            models.PointStruct(
                id="6f1c4a4e-2d5b-4d7e-9a3c-7b8e1f2a3c4d",
                vector={"dense": [1.0] * 8},
                payload={"file_row_id": "чужой"},
            )
        ],
    )
    report = _run(_runner(output, parser, registry, index))
    assert report.removed >= 1 and index.count() == 2
