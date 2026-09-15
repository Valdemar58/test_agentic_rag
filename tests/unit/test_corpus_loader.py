"""Загрузка архива экспорта (4.1): манифест, карточки, хэши; файл без карточки — ошибка и пропуск."""

from __future__ import annotations

import hashlib
import json
import logging
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from uuid import UUID, uuid4

import pytest
import structlog

from ingest.cards import CardRecord, load_card
from ingest.corpus import CorpusError, load_corpus
from synthetic.corpus import export_config
from tessa_export.fake import FakeGateway, link, make_file, make_snapshot, stable_uuid
from tessa_export.runner import run_export
from tessa_export.sample_files import minimal_docx_bytes, minimal_pdf_bytes
from tessa_export.storage import CARDS_DIR, FILES_DIR, export_dir

ORDER = stable_uuid("loader", "order")
MEMO = stable_uuid("loader", "memo")
ORDER_DOCX = "ДокШаблон Приказ №144.docx"
ORDER_PDF = "144 от 15.01.2026 О назначении ответственных.pdf"
MEMO_DOCX = "СЗ-9.docx"


@pytest.fixture(autouse=True)
def _reset_logging() -> Iterator[None]:
    yield
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()
    structlog.reset_defaults()


def build_mini_scenario() -> tuple[FakeGateway, list[UUID]]:
    """Приказ с оригиналом, регистрационным pdf, виртуальным листом и подписью; служебная записка."""
    gateway = FakeGateway()
    order_files = [
        make_file(ORDER, ORDER_DOCX),
        make_file(ORDER, ORDER_PDF),
        make_file(ORDER, "Лист согласования.html", category=None, is_virtual=True, size=-1),
        make_file(ORDER, "Подпись.sig", category="Подписи ЭП"),
    ]
    gateway.add(
        make_snapshot(
            ORDER,
            number="144",
            outgoing=[link(MEMO, ref_type_name="Документ-основание", ref_type_reverse_name="Приказ")],
            files=order_files,
        ),
        {
            ORDER_DOCX: minimal_docx_bytes(["ПРИКАЗЫВАЮ:", "1. Назначить ответственных."]),
            ORDER_PDF: minimal_pdf_bytes("Order 144"),
            "Подпись.sig": b"signature",
        },
    )
    gateway.add(
        make_snapshot(
            MEMO,
            type_name="InhouseDocumentMKC",
            type_caption="Служебная записка",
            number="СЗ-9",
            status_id=None,
            status_name=None,
            incoming=[link(ORDER, ref_type_name=None, ref_type_reverse_name=None)],
            files=[make_file(MEMO, MEMO_DOCX)],
        ),
        {MEMO_DOCX: minimal_docx_bytes(["Прошу назначить ответственных."])},
    )
    return gateway, [ORDER]


@pytest.fixture
def corpus_dir(tmp_path: Path) -> Path:
    gateway, seed = build_mini_scenario()
    output = tmp_path / "corpus"
    run_export(export_config(output), seed, gateway, synthetic=True, source="synthetic")
    return output


def test_load_corpus_reads_manifest_cards_and_hashes(corpus_dir: Path) -> None:
    corpus = load_corpus(corpus_dir)
    # принимается и родитель с export/, и сам каталог export/
    assert load_corpus(export_dir(corpus_dir)).export_dir == corpus.export_dir == export_dir(corpus_dir)
    assert corpus.synthetic and corpus.issues == []
    assert {document.card_id for document in corpus.documents} == {ORDER, MEMO}

    files = corpus.files
    assert sorted(file.name for file in files) == sorted([ORDER_DOCX, ORDER_PDF, MEMO_DOCX])
    manifest_hashes = {
        file.row_id: file.sha256 for document in corpus.manifest.documents for file in document.files
    }
    for file in files:
        assert (
            file.sha256 == hashlib.sha256(file.path.read_bytes()).hexdigest() == manifest_hashes[file.row_id]
        )
        assert file.size == file.path.stat().st_size > 0
        assert file.relative_path.startswith(f"{FILES_DIR}/{file.card_id}/")
    # виртуальный html и .sig экспорт не скачал — они не вход инжеста и не в знаменателе
    assert corpus.not_downloaded == 2
    assert len(corpus.links_graph.edges) == 1

    order = next(document for document in corpus.documents if document.card_id == ORDER)
    assert order.card.type_name == "OrderMKC" and order.card.common_text("FullNumber") == "144"
    assert order.entry.doc_status == "active" and order.entry.number == "144"
    assert [row["DocID"] for row in order.card.rows("OutgoingRefDocs")] == [str(MEMO)]
    assert order.card.file_by_row_id(order.files[0].row_id) is not None
    assert "СИНТЕТИЧЕСКИЕ" in corpus.summary_lines()[0]


def test_file_without_card_is_logged_and_skipped(corpus_dir: Path, caplog: pytest.LogCaptureFixture) -> None:
    export = export_dir(corpus_dir)
    (export / CARDS_DIR / f"{MEMO}.json").unlink()
    orphan_dir = export / FILES_DIR / str(uuid4())
    orphan_dir.mkdir()
    (orphan_dir / "чужой.docx").write_bytes(b"orphan")
    (export / FILES_DIR / str(ORDER) / "лишний.pdf").write_bytes(b"extra")

    with caplog.at_level(logging.ERROR, logger="ingest.corpus"):
        corpus = load_corpus(corpus_dir)

    assert {document.card_id for document in corpus.documents} == {ORDER}
    assert all(file.card_id != MEMO for file in corpus.files)
    assert Counter(issue.kind for issue in corpus.issues) == {
        "card_missing": 1,
        "file_without_card": 1,
        "file_not_in_manifest": 1,
    }
    memo_issue = next(issue for issue in corpus.issues if issue.kind == "card_missing")
    assert memo_issue.card_id == MEMO and memo_issue.path.endswith(MEMO_DOCX)
    assert "нет карточки" in caplog.text and "файл без карточки в сете" in caplog.text
    assert len(corpus.files) == 2


def test_invalid_card_is_reported_with_reason(corpus_dir: Path) -> None:
    export = export_dir(corpus_dir)
    memo_card = export / CARDS_DIR / f"{MEMO}.json"
    memo_card.write_text("{not json", encoding="utf-8")
    corpus = load_corpus(corpus_dir)
    invalid = [issue for issue in corpus.issues if issue.kind == "card_invalid"]
    assert len(invalid) == 1 and "JSON" in invalid[0].detail

    # валидный JSON, но чужая карточка под именем записки
    order_card = json.loads((export / CARDS_DIR / f"{ORDER}.json").read_text(encoding="utf-8"))
    memo_card.write_text(json.dumps(order_card), encoding="utf-8")
    corpus = load_corpus(corpus_dir)
    invalid = [issue for issue in corpus.issues if issue.kind == "card_invalid"]
    assert len(invalid) == 1 and "не совпадает с манифестом" in invalid[0].detail

    memo_card.write_text(json.dumps({"id": str(MEMO), "sections": "не словарь"}), encoding="utf-8")
    corpus = load_corpus(corpus_dir)
    assert "sections" in corpus.issues[0].detail


def test_missing_and_modified_files_are_reported(corpus_dir: Path) -> None:
    export = export_dir(corpus_dir)
    order_dir = export / FILES_DIR / str(ORDER)
    (order_dir / ORDER_PDF).unlink()
    with (order_dir / ORDER_DOCX).open("ab") as stream:
        stream.write(b"tail")

    corpus = load_corpus(corpus_dir)
    assert Counter(issue.kind for issue in corpus.issues) == {"file_missing": 1, "hash_mismatch": 1}
    assert [file.name for file in corpus.files] == [MEMO_DOCX]
    # документ остаётся в корпусе, просто без недопущенных файлов
    order = next(document for document in corpus.documents if document.card_id == ORDER)
    assert order.files == []
    mismatch = next(issue for issue in corpus.issues if issue.kind == "hash_mismatch")
    assert "на диске" in mismatch.detail and "в манифесте" in mismatch.detail


def test_load_corpus_errors_are_readable(tmp_path: Path) -> None:
    with pytest.raises(CorpusError, match="manifest.json"):
        load_corpus(tmp_path / "nothing")
    (tmp_path / "manifest.json").write_text("{}", encoding="utf-8")
    with pytest.raises(CorpusError, match="manifest.json не соответствует формату экспорта"):
        load_corpus(tmp_path)


def test_card_record_mirrors_card_data_shape(corpus_dir: Path) -> None:
    card = load_card(export_dir(corpus_dir) / CARDS_DIR / f"{ORDER}.json")
    assert isinstance(card, CardRecord) and card.id == ORDER
    assert card.fields("DocumentCommonInfo")["Subject"] and card.rows("DocumentCommonInfo") == []
    assert card.rows("IncomingRefDocs") == [] and card.fields("НетТакойСекции") == {}
    assert card.common_text("Comment") is None
    names = {file.name for file in card.files}
    assert names == {ORDER_DOCX, ORDER_PDF, "Лист согласования.html", "Подпись.sig"}
    virtual = next(file for file in card.files if file.is_virtual)
    assert virtual.size == -1 and virtual.category_caption is None
