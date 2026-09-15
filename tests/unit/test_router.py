"""Маршрутизатор форматов (4.2): текстовый слой pdf, офисные форматы, изображения, битые файлы."""

from __future__ import annotations

import io
from pathlib import Path
from uuid import uuid4

import pytest
from pypdf import PdfReader, PdfWriter

from common.config import DEFAULT_CONFIG_PATH, load_app_config
from ingest.corpus import CorpusFile
from ingest.router import choose_route, pdf_text_layer
from synthetic import texts
from synthetic.documents import pdf_bytes, scan_pdf_bytes

SETTINGS = load_app_config(DEFAULT_CONFIG_PATH).ingest


def _file(tmp_path: Path, name: str, content: bytes) -> CorpusFile:
    path = tmp_path / name
    path.write_bytes(content)
    return CorpusFile(
        card_id=uuid4(),
        row_id=uuid4(),
        name=name,
        extension=name.rsplit(".", 1)[-1].lower(),
        category="Документ",
        relative_path=f"files/{name}",
        path=path,
        sha256="0" * 64,
        size=len(content),
        has_text_layer=None,
        page_count=None,
        duplicate_of=None,
        smoke_note=None,
    )


def _mixed_pdf(text_pdf: bytes, scan_pdf: bytes, scan_pages: int) -> bytes:
    writer = PdfWriter()
    for page in PdfReader(io.BytesIO(text_pdf)).pages:
        writer.add_page(page)
    scan_page = PdfReader(io.BytesIO(scan_pdf)).pages[0]
    for _ in range(scan_pages):
        writer.add_page(scan_page)
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


def test_text_pdf_goes_native_and_scan_pdf_goes_vlm(tmp_path: Path) -> None:
    text = choose_route(_file(tmp_path, "приказ.pdf", pdf_bytes(texts.ORDER_144)), SETTINGS)
    assert text.route == "native" and text.text_layer is not None
    assert text.text_layer.pages_with_text == text.text_layer.pages >= 1
    scan = choose_route(_file(tmp_path, "скан.pdf", scan_pdf_bytes(texts.ORDER_144)), SETTINGS)
    assert scan.route == "vlm" and scan.text_layer is not None
    assert scan.text_layer.pages_with_text == 0 and "скан" in scan.reason


def test_mixed_pdf_follows_page_share_threshold(tmp_path: Path) -> None:
    text_pdf = pdf_bytes(texts.ORDER_144)
    text_pages = len(PdfReader(io.BytesIO(text_pdf)).pages)
    # столько же сканов, сколько текстовых страниц: доля ровно 0.5 → нативно при пороге 0.5
    half = _file(tmp_path, "половина.pdf", _mixed_pdf(text_pdf, scan_pdf_bytes(texts.ORDER_109), text_pages))
    layer = pdf_text_layer(half.path, SETTINGS.text_layer_min_chars_per_page)
    assert layer.pages == 2 * text_pages and layer.share == 0.5
    assert choose_route(half, SETTINGS).route == "native"
    stricter = SETTINGS.model_copy(update={"text_layer_min_page_share": 0.6})
    assert choose_route(half, stricter).route == "vlm"
    mostly_scan = _file(
        tmp_path, "сканы.pdf", _mixed_pdf(text_pdf, scan_pdf_bytes(texts.ORDER_109), 3 * text_pages)
    )
    assert choose_route(mostly_scan, SETTINGS).route == "vlm"


def test_office_formats_native_images_vlm_broken_pdf_vlm(tmp_path: Path) -> None:
    assert choose_route(_file(tmp_path, "текст.docx", b"PK"), SETTINGS).route == "native"
    assert choose_route(_file(tmp_path, "таблица.xlsx", b"PK"), SETTINGS).route == "native"
    assert choose_route(_file(tmp_path, "слайды.pptx", b"PK"), SETTINGS).route == "native"
    for name in ("скан.jpg", "скан.tiff", "скан.png", "скан.gif"):
        assert choose_route(_file(tmp_path, name, b"\xff\xd8"), SETTINGS).route == "vlm"
    broken = choose_route(_file(tmp_path, "битый.pdf", b"%PDF-1.4 garbage"), SETTINGS)
    assert broken.route == "vlm" and "не определён" in broken.reason and broken.text_layer is None
    with pytest.raises(ValueError, match="не поддерживается"):
        choose_route(_file(tmp_path, "файл.doc", b""), SETTINGS)
