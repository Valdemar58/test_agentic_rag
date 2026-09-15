"""Кэш разбора (4.9): второй разбор того же файла берётся из JSON, конвертер не вызывается."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from docling_core.types.doc.document import DoclingDocument
from docling_core.types.doc.labels import DocItemLabel

from common.config import DEFAULT_CONFIG_PATH, load_app_config
from ingest.parsing import CACHE_DIR, CACHED_STATUS, DocumentParser

CONFIG = load_app_config(DEFAULT_CONFIG_PATH)


class FakeConverter:
    def __init__(self) -> None:
        self.calls = 0

    def convert(self, source: Path, raises_on_error: bool = False) -> Any:
        from docling.datamodel.base_models import ConversionStatus

        self.calls += 1
        document = DoclingDocument(name=source.stem)
        document.add_heading("Приказ", level=1)
        document.add_text(label=DocItemLabel.TEXT, text="1. Утвердить положение.")

        class Result:
            status = ConversionStatus.SUCCESS
            errors: list[Any] = []

        result = Result()
        result.document = document  # type: ignore[attr-defined]
        return result


def test_second_parse_is_served_from_cache(tmp_path: Path) -> None:
    parser = DocumentParser(CONFIG, vlm_base_url="http://127.0.0.1:1/v1", work_dir=tmp_path)
    converter = FakeConverter()
    parser._native = converter  # подменяем ленивый конвертер, Docling не нужен
    path = tmp_path / "приказ.docx"
    path.write_bytes(b"PK first")

    first = parser.parse(path, "native")
    assert first.ok and first.status == "success" and converter.calls == 1
    cached_files = list((tmp_path / CACHE_DIR).glob("*-native-v*.json"))
    assert len(cached_files) == 1

    second = parser.parse(path, "native")
    assert second.ok and second.status == CACHED_STATUS and converter.calls == 1
    assert second.document.export_to_markdown() == first.document.export_to_markdown()

    # другой маршрут и другое содержимое — свой ключ кэша
    path.write_bytes(b"PK second")
    third = parser.parse(path, "native")
    assert third.status == "success" and converter.calls == 2
    assert len(list((tmp_path / CACHE_DIR).glob("*.json"))) == 2

    disabled = DocumentParser(
        CONFIG.model_copy(update={"ingest": CONFIG.ingest.model_copy(update={"parse_cache": False})}),
        vlm_base_url="http://127.0.0.1:1/v1",
        work_dir=tmp_path / "other",
    )
    disabled._native = converter
    disabled.parse(path, "native")
    assert converter.calls == 3 and not (tmp_path / "other" / CACHE_DIR).exists()
