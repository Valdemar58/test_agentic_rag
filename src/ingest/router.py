"""Маршрутизатор форматов (FR-3): нативный конвейер Docling или VLM-конвейер с dots.mocr.

- docx / xlsx / pptx — всегда нативно (структура берётся из самого файла);
- изображения — всегда VLM (распознавание dots.mocr);
- pdf — по текстовому слою: страница считается текстовой, если pypdf извлекает из неё не меньше
  `ingest.text_layer_min_chars_per_page` символов и слой не выглядит мусором чужого OCR (доля
  слов со смесью кириллицы и латиницы не выше `text_layer_max_mixed_script_share`); документ идёт
  нативно, когда доля таких страниц не меньше `ingest.text_layer_min_page_share`, иначе целиком
  в VLM. Нечитаемый pdf тоже отправляется в VLM: растеризация может сработать там, где не
  работает извлечение текста.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pypdf import PdfReader

from common.config import IngestSettings
from ingest.corpus import CorpusFile

ParseRoute = Literal["native", "vlm"]
OFFICE_EXTENSIONS = frozenset({"docx", "xlsx", "pptx"})
IMAGE_EXTENSIONS = frozenset({"png", "jpg", "jpeg", "tif", "tiff", "bmp", "gif"})

_WORD_RE = re.compile(r"[^\W\d_]+")
_CYRILLIC_RE = re.compile(r"[А-Яа-яЁё]")
_LATIN_RE = re.compile(r"[A-Za-z]")


@dataclass(frozen=True)
class TextLayer:
    pages: int
    pages_with_text: int
    chars: int
    garbage_pages: int = 0

    @property
    def share(self) -> float:
        return self.pages_with_text / self.pages if self.pages else 0.0


@dataclass(frozen=True)
class RouteDecision:
    route: ParseRoute
    reason: str
    text_layer: TextLayer | None = None


def mixed_script_share(text: str) -> float:
    """Доля слов, в которых есть и кириллица, и латиница («прихOд» с латинской O)."""
    words = _WORD_RE.findall(text)
    if not words:
        return 0.0
    mixed = sum(1 for word in words if _CYRILLIC_RE.search(word) and _LATIN_RE.search(word))
    return mixed / len(words)


def pdf_text_layer(path: Path, min_chars_per_page: int, max_mixed_share: float = 1.0) -> TextLayer:
    """Считает страницы с текстовым слоем; ошибки чтения pdf поднимаются наверх."""
    reader = PdfReader(path)
    if reader.is_encrypted:
        reader.decrypt("")
    pages = 0
    pages_with_text = 0
    garbage_pages = 0
    chars = 0
    for page in reader.pages:
        pages += 1
        raw = page.extract_text() or ""
        text = "".join(raw.split())
        chars += len(text)
        if len(text) < min_chars_per_page:
            continue
        if mixed_script_share(raw) > max_mixed_share:
            garbage_pages += 1
            continue
        pages_with_text += 1
    return TextLayer(pages=pages, pages_with_text=pages_with_text, chars=chars, garbage_pages=garbage_pages)


def choose_route(file: CorpusFile, settings: IngestSettings) -> RouteDecision:
    extension = file.extension
    if extension in OFFICE_EXTENSIONS:
        return RouteDecision("native", "офисный формат: структура из файла, нативный конвейер Docling")
    if extension in IMAGE_EXTENSIONS:
        return RouteDecision("vlm", "изображение: распознавание dots.mocr")
    if extension != "pdf":
        raise ValueError(f"расширение {extension!r} не поддерживается маршрутизатором")
    try:
        layer = pdf_text_layer(
            file.path, settings.text_layer_min_chars_per_page, settings.text_layer_max_mixed_script_share
        )
    except Exception as exc:  # noqa: BLE001 — любой сбой pypdf: файл пробует VLM, ошибка сохраняется
        return RouteDecision("vlm", f"текстовый слой не определён ({type(exc).__name__}: {exc}); dots.mocr")
    detail = f"текстовый слой на {layer.pages_with_text} из {layer.pages} страниц"
    if layer.garbage_pages:
        detail += f", мусорный слой на {layer.garbage_pages}"
    if layer.pages and layer.share >= settings.text_layer_min_page_share:
        return RouteDecision("native", f"{detail}: нативный конвейер Docling", layer)
    return RouteDecision("vlm", f"скан: {detail}, распознавание dots.mocr", layer)
