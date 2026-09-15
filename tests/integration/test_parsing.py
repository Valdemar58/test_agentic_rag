"""Конвейеры Docling (4.2): нативный разбор docx/xlsx/pdf и VLM-ветка с dots.mocr (AC-3.4, основа).

Нативные тесты docx/xlsx работают без моделей; pdf требует скачанных моделей Docling (`models/`),
VLM-тест — поднятого профиля `ingest` (vllm-dots). Без них — скип с причиной.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
from docling_core.types.doc.labels import DocItemLabel

from common.config import DEFAULT_CONFIG_PATH, load_app_config
from common.model_store import model_status
from common.settings import load_settings
from ingest.parsing import DocumentParser
from synthetic import texts
from synthetic.documents import docx_bytes, pdf_bytes, scan_image_bytes, scan_pdf_bytes, xlsx_bytes

pytestmark = pytest.mark.integration

CONFIG = load_app_config(DEFAULT_CONFIG_PATH)


@pytest.fixture(scope="module")
def parser(tmp_path_factory: pytest.TempPathFactory) -> Iterator[DocumentParser]:
    settings = load_settings()
    yield DocumentParser(
        CONFIG, vlm_base_url=settings.resolve_vlm_base_url(CONFIG), work_dir=tmp_path_factory.mktemp("work")
    )


def _require_docling_models() -> None:
    for source in (CONFIG.models.docling_layout, CONFIG.models.docling_tables):
        if not model_status(source, CONFIG.models.dir_absolute).present:
            pytest.skip(f"модель Docling {source.repo_id} не скачана (scripts/download_models.py)")


def _require_vlm(parser: DocumentParser) -> None:
    try:
        httpx.get(parser.vlm_endpoint.replace("/chat/completions", "/models"), timeout=3).raise_for_status()
    except (httpx.HTTPError, OSError) as exc:
        pytest.skip(f"vllm-dots недоступен ({exc}); поднимите профиль ingest")


def _items(document: Any, label: DocItemLabel) -> list[Any]:
    return [item for item, _ in document.iterate_items() if item.label == label]


def _headers(document: Any) -> list[str]:
    return [str(item.text) for item in _items(document, DocItemLabel.SECTION_HEADER)]


def _tables(document: Any) -> list[Any]:
    return _items(document, DocItemLabel.TABLE)


def test_docx_native_keeps_headings_and_tables(parser: DocumentParser, tmp_path: Path) -> None:
    path = tmp_path / "положение.docx"
    path.write_bytes(docx_bytes(texts.REGULATION_OT))
    result = parser.parse(path, "native")
    assert result.ok and result.status == "success", result.errors
    headers = _headers(result.document)
    assert any("Термины и определения" in header for header in headers), headers
    assert len(headers) >= 3 and _tables(result.document)
    assert "СИЗ" in result.document.export_to_markdown()


def test_xlsx_native_gives_table(parser: DocumentParser, tmp_path: Path) -> None:
    path = tmp_path / "спецификация.xlsx"
    path.write_bytes(xlsx_bytes("Спецификация", texts.CONTRACT_SPECIFICATION))
    result = parser.parse(path, "native")
    assert result.ok, result.errors
    tables = _tables(result.document)
    assert tables and tables[0].export_to_dataframe(result.document).shape[0] >= 1


def test_pdf_native_uses_layout_models_offline(parser: DocumentParser, tmp_path: Path) -> None:
    _require_docling_models()
    path = tmp_path / "приказ.pdf"
    path.write_bytes(pdf_bytes(texts.ORDER_144))
    result = parser.parse(path, "native")
    assert result.ok and result.pages >= 1, result.errors
    assert _items(result.document, DocItemLabel.TEXT) or _items(result.document, DocItemLabel.LIST_ITEM)
    markdown = result.document.export_to_markdown()
    assert "ПРИКАЗЫВАЮ" in markdown.upper()


@pytest.mark.gpu
def test_scan_image_and_scan_pdf_via_dots_mocr(parser: DocumentParser, tmp_path: Path) -> None:
    _require_vlm(parser)
    image = tmp_path / "скан.jpg"
    image.write_bytes(scan_image_bytes(texts.INSTRUCTION_FIRE))
    result = parser.parse(image, "vlm")
    assert result.ok and result.pages == 1, result.errors
    markdown = result.document.export_to_markdown()
    assert texts.INSTRUCTION_FIRE.title.split()[0].lower() in markdown.lower()
    labels = {item.label for item, _ in result.document.iterate_items()}
    assert labels & {DocItemLabel.SECTION_HEADER, DocItemLabel.TITLE, DocItemLabel.TEXT}

    scan = tmp_path / "скан.pdf"
    scan.write_bytes(scan_pdf_bytes(texts.ORDER_109))
    result = parser.parse(scan, "vlm")
    assert result.ok and result.pages >= 1, result.errors
    markdown = result.document.export_to_markdown().lower()
    assert "приказываю" in markdown and "отчёт" in markdown
    assert _headers(result.document)
