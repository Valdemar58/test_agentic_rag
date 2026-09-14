"""Проверка файлов, валидация 8.3, покрытие 8.2 и отчёт (AC-EXP.2)."""

from __future__ import annotations

from pathlib import Path
from uuid import UUID, uuid4

from tessa_export.config import CoverageSettings, ExportConfig
from tessa_export.fake import FakeGateway, build_demo_scenario, make_file, make_snapshot, stable_uuid
from tessa_export.files import FileRecord, download_card_files
from tessa_export.inspect_files import inspect_file, inspect_records
from tessa_export.manifest import Manifest, build_links_graph, build_manifest
from tessa_export.report import render_report
from tessa_export.sample_files import (
    docx_with_broken_part_bytes,
    docx_with_misdeclared_altchunk_bytes,
    minimal_docx_bytes,
    minimal_image_bytes,
    minimal_pdf_bytes,
)
from tessa_export.validation import validate
from tessa_export.walker import Walker

A, B = stable_uuid("card", "A"), stable_uuid("card", "B")
IMAGES = {"png", "jpg", "jpeg", "tif", "tiff", "bmp", "gif"}
TERMS = CoverageSettings().terms_section_patterns


def _config() -> ExportConfig:
    return ExportConfig.model_validate(
        {"tessa": {"base_url": "http://t"}, "external": {"tessa_sdk_path": "/a", "card_service_path": "/b"}}
    )


def _pipeline(gateway: FakeGateway, seed: list[UUID], root: Path, config: ExportConfig) -> Manifest:
    result = Walker(gateway, config.traversal, config.exclude_rules).walk(seed)
    allowed = set(config.files.allowed_extensions)
    records: dict[UUID, list[FileRecord]] = {
        card_id: download_card_files(gateway, card_id, visited.snapshot.files, allowed, root)
        for card_id, visited in result.cards.items()
    }
    inspect_records(root, records, config.files, config.coverage)
    manifest = build_manifest(result, records, config, synthetic=True)
    manifest_graph = build_links_graph(result)
    manifest.__dict__["_graph"] = manifest_graph  # для теста; в рантайме граф передаётся отдельно
    return manifest


def test_inspect_file_kinds() -> None:
    text_pdf = inspect_file(minimal_pdf_bytes("Термины и определения"), "pdf", IMAGES, TERMS)
    assert text_pdf.smoke_ok and text_pdf.has_text_layer is True and text_pdf.page_count == 1
    scan_pdf = inspect_file(minimal_pdf_bytes(None), "pdf", IMAGES, TERMS)
    assert scan_pdf.smoke_ok and scan_pdf.has_text_layer is False and scan_pdf.has_terms_section is None
    docx = inspect_file(
        minimal_docx_bytes(["x"], with_table=True, terms_heading="Сокращения"), "docx", IMAGES, TERMS
    )
    assert docx.smoke_ok and docx.has_tables is True and docx.has_terms_section is True
    plain_docx = inspect_file(minimal_docx_bytes(["Обычный текст"]), "docx", IMAGES, TERMS)
    assert plain_docx.has_tables is False and plain_docx.has_terms_section is False
    image = inspect_file(minimal_image_bytes(), "jpg", IMAGES, TERMS)
    assert image.smoke_ok and image.has_text_layer is False
    broken = inspect_file(b"not a pdf at all", "pdf", IMAGES, TERMS)
    assert broken.smoke_ok is False and broken.smoke_error
    unknown = inspect_file(b"whatever", "bin", IMAGES, TERMS)
    assert unknown.smoke_ok and unknown.has_text_layer is None


def test_docx_unreadable_by_python_docx_but_valid_ooxml_opens_with_note() -> None:
    import io

    import pytest
    from docx import Document

    sample = docx_with_misdeclared_altchunk_bytes(
        ["Термины и определения", "СИЗ — средства"], with_table=True
    )
    # фикстура воспроизводит реальный случай: python-docx падает на такой файл
    with pytest.raises(Exception, match="Start tag expected"):
        Document(io.BytesIO(sample))
    inspection = inspect_file(sample, "docx", IMAGES, TERMS)
    assert inspection.smoke_ok and inspection.smoke_error is None
    assert inspection.smoke_note and "python-docx" in inspection.smoke_note
    assert inspection.has_text_layer is True and inspection.has_tables is True
    assert inspection.has_terms_section is True

    plain = inspect_file(docx_with_misdeclared_altchunk_bytes(["Обычный текст"]), "docx", IMAGES, TERMS)
    assert plain.smoke_ok and plain.has_tables is False and plain.has_terms_section is False

    # по-настоящему битый пакет остаётся ошибкой с исходным текстом python-docx
    broken = inspect_file(docx_with_broken_part_bytes(), "docx", IMAGES, TERMS)
    assert broken.smoke_ok is False and broken.smoke_note is None
    assert broken.smoke_error and "XMLSyntaxError" in broken.smoke_error


def test_misdeclared_altchunk_docx_is_warn_not_fail(tmp_path: Path) -> None:
    gateway = FakeGateway()
    card_id = uuid4()
    file = make_file(card_id, "Для печати_ДокШаблон приказа с ЭЦП.docx")
    gateway.add(
        make_snapshot(card_id, files=[file]),
        {file.name: docx_with_misdeclared_altchunk_bytes(["Приказ"])},
    )
    config = _config()
    manifest = _pipeline(gateway, [card_id], tmp_path, config)
    report = validate(manifest, manifest.__dict__["_graph"], config.coverage)
    check = next(item for item in report.checks if item.name == "Файлы открываются")
    assert check.status == "WARN" and report.overall == "PASS"
    assert len(check.details) == 1 and "python-docx" in check.details[0]
    entry = manifest.documents[0].files[0]
    assert entry.smoke_ok is True and entry.smoke_note
    text = render_report(manifest, report)
    assert "ИТОГ: ПРИГОДЕН" in text and "прямым разбором OOXML" in text


def test_demo_scenario_passes_checks_and_reports_deficits(tmp_path: Path) -> None:
    gateway, seed = build_demo_scenario()
    config = _config()
    manifest = _pipeline(gateway, seed, tmp_path, config)
    graph = manifest.__dict__["_graph"]
    report = validate(manifest, graph, config.coverage)

    statuses = {check.name: check.status for check in report.checks}
    assert statuses["Файлы скачаны"] == "PASS"
    assert statuses["Файлы открываются"] == "PASS"
    assert statuses["Дедупликация"] == "PASS"
    assert statuses["Исключения"] == "PASS"
    assert statuses["Получение карточек"] == "PASS"
    assert statuses["Полнота обхода связей"] == "WARN"  # есть связь глубже max_depth
    assert report.overall == "PASS"
    deficits = {row.name for row in report.coverage_deficits}
    assert "Приказы" in deficits  # в демо всего 2 приказа против ориентира 20
    order_docs = [document for document in manifest.documents if "Приказы" in document.coverage_kinds]
    assert any(any(file.has_terms_section for file in document.files) for document in order_docs)
    scan_row = next(row for row in report.coverage if row.name == "Сканы, доля")
    assert scan_row.ok  # jpg + pdf без текстового слоя

    text = render_report(manifest, report)
    assert "ИТОГ: ПРИГОДЕН" in text
    assert "Данные синтетические" in text
    assert "в отмену → отменено" in text
    assert "Отмененный" in text
    assert "## Покрытие 8.2" in text
    (tmp_path / "validation_report.md").write_text(text, encoding="utf-8")


def test_broken_file_and_seed_error_make_report_fail(tmp_path: Path) -> None:
    gateway = FakeGateway()
    good, missing_seed = uuid4(), uuid4()
    broken_pdf = make_file(good, "битый.pdf")
    gateway.add(make_snapshot(good, files=[broken_pdf]), {"битый.pdf": b"not a pdf"})
    config = _config()
    manifest = _pipeline(gateway, [good, missing_seed], tmp_path, config)
    graph = manifest.__dict__["_graph"]
    report = validate(manifest, graph, config.coverage)
    statuses = {check.name: check.status for check in report.checks}
    assert statuses["Файлы открываются"] == "FAIL"
    assert statuses["Получение карточек"] == "FAIL"
    assert report.overall == "FAIL"
    text = render_report(manifest, report)
    assert "ИТОГ: НЕ ПРИГОДЕН" in text
    assert "битый.pdf" in text
    assert str(missing_seed) in text


def test_download_error_and_nonseed_error(tmp_path: Path) -> None:
    from tessa_export.models import GatewayConnectionError

    gateway, seed = build_demo_scenario()
    gateway.card_errors[B] = GatewayConnectionError("сеть")
    order = gateway.cards[A]
    pdf = next(file for file in order.files if file.extension == "pdf")
    gateway.file_errors[(A, pdf.row_id)] = GatewayConnectionError("тайм-аут")
    config = _config()
    manifest = _pipeline(gateway, seed, tmp_path, config)
    report = validate(manifest, manifest.__dict__["_graph"], config.coverage)
    statuses = {check.name: check.status for check in report.checks}
    assert statuses["Файлы скачаны"] == "FAIL"
    assert statuses["Получение карточек"] == "WARN"
    assert report.overall == "FAIL"
