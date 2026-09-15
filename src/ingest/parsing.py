"""Разбор файлов в DoclingDocument (FR-3, §3 ТЗ): два конвейера Docling по решению маршрутизатора.

- native: `StandardPdfPipeline` для pdf с текстовым слоем (разметка layout-heron и TableFormer из
  локального `models/`, OCR Docling выключен) и встроенные бэкенды для docx/xlsx/pptx;
- vlm: `VlmPipeline` с пресетом `dots_mocr` через OpenAI-совместимый endpoint профиля `ingest`
  (`vllm-dots`): страница растеризуется, dots.mocr возвращает JSON разметки (заголовки, абзацы,
  списки, таблицы HTML), Docling собирает из него DoclingDocument. Изображения идут тем же путём;
  gif Docling не принимает — конвертируется в png в рабочем каталоге.

Ошибка разбора одного файла не поднимается наверх, а возвращается в `ParseResult` (AC-3.1).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from PIL import Image

from common.config import AppConfig
from ingest.router import ParseRoute

logger = logging.getLogger(__name__)

DOTS_PRESET = "dots_mocr"
CHAT_COMPLETIONS_PATH = "/chat/completions"
GIF_EXTENSION = ".gif"


@dataclass(frozen=True)
class ParseResult:
    route: ParseRoute
    status: str
    seconds: float
    document: Any = None
    pages: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.document is not None


class Parser(Protocol):
    """Разборщик файла в DoclingDocument; в тестах — фейк без Docling."""

    def parse(self, path: Path, route: ParseRoute) -> ParseResult: ...


class DocumentParser:
    """Ленивая обёртка над двумя `DocumentConverter`: конвертеры создаются при первом обращении."""

    def __init__(self, config: AppConfig, *, vlm_base_url: str, work_dir: Path | None = None) -> None:
        self._config = config
        self._vlm_base_url = vlm_base_url.rstrip("/")
        self._work_dir = work_dir or config.paths.work_dir_absolute
        self._native: Any = None
        self._vlm: Any = None

    @property
    def vlm_endpoint(self) -> str:
        return f"{self._vlm_base_url}{CHAT_COMPLETIONS_PATH}"

    def _native_converter(self) -> Any:
        if self._native is None:
            from docling.datamodel.accelerator_options import AcceleratorOptions
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import (
                PdfPipelineOptions,
                TableFormerMode,
                TableStructureOptions,
            )
            from docling.document_converter import DocumentConverter, PdfFormatOption

            docling = self._config.ingest.docling
            pdf_options = PdfPipelineOptions(
                do_ocr=False,
                do_table_structure=True,
                table_structure_options=TableStructureOptions(
                    mode=TableFormerMode(docling.table_mode), do_cell_matching=True
                ),
                artifacts_path=self._config.models.docling_artifacts_dir,
                accelerator_options=AcceleratorOptions(
                    device=docling.device, num_threads=docling.num_threads
                ),
                document_timeout=docling.document_timeout_s,
                images_scale=docling.images_scale,
                generate_page_images=False,
            )
            self._native = DocumentConverter(
                allowed_formats=[InputFormat.PDF, InputFormat.DOCX, InputFormat.XLSX, InputFormat.PPTX],
                format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_options)},
            )
        return self._native

    def _vlm_converter(self) -> Any:
        if self._vlm is None:
            from docling.datamodel.accelerator_options import AcceleratorOptions
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import VlmConvertOptions, VlmPipelineOptions
            from docling.datamodel.vlm_engine_options import ApiVlmEngineOptions
            from docling.document_converter import DocumentConverter, ImageFormatOption, PdfFormatOption
            from docling.models.inference_engines.vlm.base import VlmEngineType
            from docling.pipeline.vlm_pipeline import VlmPipeline

            vlm = self._config.ingest.vlm
            docling = self._config.ingest.docling
            engine = ApiVlmEngineOptions(
                engine_type=VlmEngineType.API,
                url=self.vlm_endpoint,
                params={
                    "model": self._config.vllm.dots.served_model_name,
                    "max_tokens": vlm.max_tokens,
                    "temperature": 0.0,
                },
                timeout=vlm.timeout_s,
                concurrency=1,
            )
            preset = VlmConvertOptions.from_preset(DOTS_PRESET, engine_options=engine)

            def pipeline_for(scale: float) -> Any:
                return VlmPipelineOptions(
                    vlm_options=preset.model_copy(update={"scale": scale}),
                    enable_remote_services=True,
                    document_timeout=docling.document_timeout_s,
                    accelerator_options=AcceleratorOptions(device="cpu", num_threads=docling.num_threads),
                )

            # pdf растеризуется из точек (72 dpi) с масштабом, готовые изображения идут в своём разрешении
            self._vlm = DocumentConverter(
                allowed_formats=[InputFormat.PDF, InputFormat.IMAGE],
                format_options={
                    InputFormat.PDF: PdfFormatOption(
                        pipeline_cls=VlmPipeline, pipeline_options=pipeline_for(vlm.image_scale)
                    ),
                    InputFormat.IMAGE: ImageFormatOption(
                        pipeline_cls=VlmPipeline, pipeline_options=pipeline_for(vlm.raster_image_scale)
                    ),
                },
            )
        return self._vlm

    def _prepare_source(self, path: Path, route: ParseRoute) -> Path:
        if route == "vlm" and path.suffix.lower() == GIF_EXTENSION:
            target = self._work_dir / "converted" / f"{path.stem}.png"
            target.parent.mkdir(parents=True, exist_ok=True)
            with Image.open(path) as image:
                image.convert("RGB").save(target, format="PNG")
            return target
        return path

    def parse(self, path: Path, route: ParseRoute) -> ParseResult:
        """Разбирает файл выбранным конвейером; сбой — в `errors`, без исключения."""
        from docling.datamodel.base_models import ConversionStatus

        started = time.perf_counter()
        try:
            source = self._prepare_source(path, route)
            converter = self._vlm_converter() if route == "vlm" else self._native_converter()
            result = converter.convert(source, raises_on_error=False)
        except Exception as exc:  # noqa: BLE001 — любой сбой конвейера фиксируется как ошибка файла
            logger.exception("Разбор %s (%s) не удался", path.name, route)
            return ParseResult(
                route=route,
                status="failure",
                seconds=time.perf_counter() - started,
                errors=[f"{type(exc).__name__}: {exc}"],
            )
        errors = [f"{item.module_name}: {item.error_message}" for item in result.errors]
        status = str(result.status.value)
        document = (
            result.document
            if result.status
            in (
                ConversionStatus.SUCCESS,
                ConversionStatus.PARTIAL_SUCCESS,
            )
            else None
        )
        pages = int(document.num_pages()) if document is not None else 0
        seconds = time.perf_counter() - started
        if document is None:
            logger.error(
                "Разбор %s (%s): %s — %s", path.name, route, status, "; ".join(errors) or "без деталей"
            )
        else:
            logger.info("Разобран %s (%s): %s, страниц %d, %.1f с", path.name, route, status, pages, seconds)
        return ParseResult(
            route=route, status=status, seconds=seconds, document=document, pages=pages, errors=errors
        )
