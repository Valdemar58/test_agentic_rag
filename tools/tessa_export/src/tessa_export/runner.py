"""Оркестрация экспорта: обход → карточки → файлы → проверка файлов → манифест и граф →
валидация и отчёт → архив."""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from uuid import UUID

from tessa_export.archive import make_archive
from tessa_export.config import ExportConfig
from tessa_export.files import FileRecord, download_card_files
from tessa_export.inspect_files import inspect_records
from tessa_export.manifest import build_links_graph, build_manifest
from tessa_export.models import GatewayError, TessaGateway
from tessa_export.report import render_report
from tessa_export.review import render_review_csv
from tessa_export.storage import (
    LINKS_GRAPH_NAME,
    MANIFEST_NAME,
    REPORT_NAME,
    REVIEW_NAME,
    export_dir,
    save_card,
    write_json,
)
from tessa_export.validation import validate
from tessa_export.walker import Walker

logger = logging.getLogger(__name__)


class ExportError(RuntimeError):
    """Экспорт не дал результата: ни одна карточка не получена."""


@dataclass(frozen=True)
class RunSummary:
    export_root: Path
    manifest_path: Path
    links_graph_path: Path
    report_path: Path
    review_path: Path
    archive_path: Path
    documents: int
    files_downloaded: int
    overall: Literal["PASS", "FAIL"]


def run_export(
    config: ExportConfig, seed_ids: list[UUID], gateway: TessaGateway, *, synthetic: bool = False
) -> RunSummary:
    export_root = export_dir(config.output_dir)
    if export_root.exists():
        logger.info("Очищаю предыдущий результат: %s", export_root)
        shutil.rmtree(export_root)
    export_root.mkdir(parents=True, exist_ok=True)

    logger.info("Проверяю подключение к Тессе")
    gateway.check_connection()

    logger.info(
        "Обход связей: seed=%d, max_depth=%d, max_docs=%d",
        len(seed_ids),
        config.traversal.max_depth,
        config.traversal.max_docs,
    )
    result = Walker(gateway, config.traversal, config.exclude_rules).walk(seed_ids)
    if not result.cards:
        first_error = result.errors[0].message if result.errors else "причина неизвестна"
        raise ExportError(f"ни одна карточка не получена; первая ошибка: {first_error}")

    allowed = set(config.files.allowed_extensions)
    file_records: dict[UUID, list[FileRecord]] = {}
    for index, (card_id, visited) in enumerate(result.cards.items(), start=1):
        save_card(export_root, visited.snapshot)
        logger.info("[%d/%d] Файлы карточки %s", index, len(result.cards), card_id)
        file_records[card_id] = download_card_files(
            gateway, card_id, visited.snapshot.files, allowed, export_root
        )

    logger.info("Проверяю скачанные файлы")
    inspect_records(export_root, file_records, config.files, config.coverage)

    manifest = build_manifest(result, file_records, config, synthetic=synthetic)
    graph = build_links_graph(result)
    manifest_path = export_root / MANIFEST_NAME
    links_graph_path = export_root / LINKS_GRAPH_NAME
    write_json(manifest_path, manifest.model_dump(mode="json"))
    write_json(links_graph_path, graph.model_dump(mode="json"))

    report = validate(manifest, graph, config.coverage)
    report_path = export_root / REPORT_NAME
    report_path.write_text(render_report(manifest, report), encoding="utf-8", newline="\n")
    review_path = export_root / REVIEW_NAME
    # BOM нужен, чтобы Excel открыл кириллицу без вопросов о кодировке
    review_path.write_text(render_review_csv(manifest), encoding="utf-8-sig", newline="\n")

    archive_path = make_archive(export_root, config.output_dir / config.archive_name)
    logger.info(
        "Готово: документов %d, файлов скачано %d, итог валидации %s, архив %s",
        manifest.stats.documents,
        manifest.stats.files_downloaded,
        report.overall,
        archive_path,
    )
    return RunSummary(
        export_root=export_root,
        manifest_path=manifest_path,
        links_graph_path=links_graph_path,
        report_path=report_path,
        review_path=review_path,
        archive_path=archive_path,
        documents=manifest.stats.documents,
        files_downloaded=manifest.stats.files_downloaded,
        overall=report.overall,
    )


__all__ = ["ExportError", "GatewayError", "RunSummary", "run_export"]
