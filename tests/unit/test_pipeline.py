"""Конвейер файла (4.6): успех, пропуск, ошибка разбора, пустой разбор, сбой шага — без исключений наружу."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import uuid4

from docling_core.types.doc.document import DoclingDocument
from docling_core.types.doc.labels import DocItemLabel
from qdrant_client import QdrantClient

from common.config import DEFAULT_CONFIG_PATH, load_app_config
from ingest.corpus import CorpusFile
from ingest.embeddings import Embedding, FakeEmbedder
from ingest.files import FilePlan
from ingest.index import ChunkIndex
from ingest.metadata import DocumentMetadata
from ingest.parsing import ParseResult
from ingest.pipeline import IngestPipeline, PreparedFile
from ingest.tokens import WordTokenCounter

CONFIG = load_app_config(DEFAULT_CONFIG_PATH)


class FakeParser:
    """Отдаёт заранее заданный результат вместо Docling."""

    def __init__(self, result: ParseResult) -> None:
        self.result = result
        self.calls: list[tuple[Path, str]] = []

    def parse(self, path: Path, route: str) -> ParseResult:
        self.calls.append((path, route))
        return self.result


class FailingEmbedder(FakeEmbedder):
    def encode(self, texts: list[str]) -> list[Embedding]:
        raise RuntimeError("нет памяти на GPU")


def _document() -> DoclingDocument:
    doc = DoclingDocument(name="приказ")
    doc.add_heading("Об утверждении", level=1)
    doc.add_heading("1. Утверждение", level=2)
    doc.add_text(label=DocItemLabel.TEXT, text="1.1. Утвердить Положение об охране труда.")
    doc.add_text(label=DocItemLabel.TEXT, text="1.2. Ввести Положение в действие с 1 февраля.")
    return doc


def _meta() -> DocumentMetadata:
    card = str(uuid4())
    return DocumentMetadata(
        doc_id=card,
        tessa_card_id=card,
        card_type_name="OrderMKC",
        card_type_caption="Приказ",
        doc_kind="Приказ",
        doc_number="144",
        doc_date=None,
        doc_date_ts=None,
        doc_status="active",
        doc_status_name=None,
        state_id=6,
        state_name=None,
        approval_state=None,
        approval_state_name=None,
        department=None,
        department_id=None,
        author=None,
        relations=[],
        subject=None,
        comment=None,
        signed_by=None,
        direction_activity=[],
        approvers=[],
        responsible=[],
        validity_period=None,
        card_version=None,
        card_modified=None,
    )


def _plan(tmp_path: Path, role: str | None = "main", reason: str | None = None) -> FilePlan:
    name = "ДокШаблон Приказ №144.docx"
    path = tmp_path / name
    path.write_bytes(b"PK")
    file = CorpusFile(
        card_id=uuid4(),
        row_id=uuid4(),
        name=name,
        extension="docx",
        category="Документ",
        relative_path=f"files/x/{name}",
        path=path,
        sha256="f" * 64,
        size=2,
        has_text_layer=None,
        page_count=None,
        duplicate_of=None,
        smoke_note=None,
    )
    return FilePlan(file=file, role=role, skip_reason=reason)  # type: ignore[arg-type]


def _pipeline(parser: Any, embedder: Any = None) -> tuple[IngestPipeline, ChunkIndex]:
    embedder = embedder or FakeEmbedder()
    index = ChunkIndex(QdrantClient(":memory:"), CONFIG.qdrant, embedder.dense_dim)
    index.ensure_collections()
    pipeline = IngestPipeline(
        CONFIG, parser=parser, embedder=embedder, index=index, counter=WordTokenCounter()
    )
    return pipeline, index


def test_successful_file_is_chunked_embedded_and_written(tmp_path: Path) -> None:
    parser = FakeParser(
        ParseResult(route="native", status="success", seconds=0.1, document=_document(), pages=1)
    )
    pipeline, index = _pipeline(parser)
    plan = _plan(tmp_path)
    prepared = pipeline.prepare_file(_meta(), plan)
    assert isinstance(prepared, PreparedFile) and len(prepared.children) == 2 and len(prepared.parents) == 1
    assert index.count() == 0  # фаза 1 в индекс не пишет
    outcome = pipeline.index_prepared(prepared)
    assert outcome.indexed and outcome.route == "native" and outcome.chunks == 2 and outcome.parents == 1
    assert parser.calls == [(plan.file.path, "native")]
    assert index.count_file(plan.file.row_id) == 2 and index.count_file(plan.file.row_id, parents=True) == 1
    # повторный прогон того же файла не добавляет точек (AC-3.3)
    pipeline.process_file(_meta(), plan)
    assert index.count() == 2 and index.count(parents=True) == 1


def test_skipped_plan_and_parse_failure_and_empty_document(tmp_path: Path) -> None:
    pipeline, index = _pipeline(
        FakeParser(ParseResult(route="native", status="failure", seconds=0.1, errors=["backend: битый zip"]))
    )
    skipped = pipeline.process_file(_meta(), _plan(tmp_path, role=None, reason="копия для печати"))
    assert skipped.status == "skipped" and skipped.reason == "копия для печати"
    failed = pipeline.process_file(_meta(), _plan(tmp_path))
    assert failed.status == "error" and "битый zip" in (failed.reason or "") and failed.parse_errors
    assert index.count() == 0

    empty = DoclingDocument(name="пустой")
    pipeline, _ = _pipeline(
        FakeParser(ParseResult(route="native", status="success", seconds=0.1, document=empty))
    )
    outcome = pipeline.process_file(_meta(), _plan(tmp_path))
    assert outcome.status == "error" and "не дал текста" in (outcome.reason or "")


def test_step_exception_becomes_error_outcome(tmp_path: Path) -> None:
    parser = FakeParser(
        ParseResult(route="native", status="success", seconds=0.1, document=_document(), pages=1)
    )
    pipeline, index = _pipeline(parser, FailingEmbedder())
    outcome = pipeline.process_file(_meta(), _plan(tmp_path))
    assert outcome.status == "error" and "RuntimeError: нет памяти на GPU" == outcome.reason
    assert index.count() == 0
