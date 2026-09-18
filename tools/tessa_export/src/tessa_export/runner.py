"""Оркестрация экспорта: обход → карточки → файлы → проверка файлов → манифест и граф →
валидация и отчёт → архив.

Два режима. `run_export` — полный прогон по seed-списку (§8 ТЗ): каталог результата очищается,
манифест пишется с нуля. `run_incremental_export` — накопительный прогон (синхронизация приказов,
запрос заказчика 2026-09-18): каталог не очищается, уже выгруженные карточки не запрашиваются
повторно, манифест и граф связей сливаются с предыдущим прогоном.
"""

from __future__ import annotations

import logging
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from uuid import UUID

from pydantic import ValidationError

from tessa_export.archive import make_archive
from tessa_export.config import ExcludeRule, ExportConfig, TraversalSettings
from tessa_export.files import FileRecord, download_card_files
from tessa_export.inspect_files import inspect_records
from tessa_export.manifest import (
    LinksGraph,
    Manifest,
    build_links_graph,
    build_manifest,
    merge_links_graphs,
    merge_manifests,
)
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
    archive_path: Path | None
    documents: int
    files_downloaded: int
    overall: Literal["PASS", "FAIL"]
    documents_added: int = 0
    documents_known: int = 0


@dataclass(frozen=True)
class PreviousExport:
    """Результат предыдущих прогонов в том же каталоге."""

    manifest: Manifest
    graph: LinksGraph

    @property
    def card_ids(self) -> set[UUID]:
        return {document.card_id for document in self.manifest.documents}


def load_previous(export_root: Path) -> PreviousExport | None:
    """Читает манифест и граф прошлых прогонов; None — каталог пуст, экспорт начинается с нуля."""
    manifest_path = export_root / MANIFEST_NAME
    graph_path = export_root / LINKS_GRAPH_NAME
    if not manifest_path.is_file():
        return None
    try:
        manifest = Manifest.model_validate_json(manifest_path.read_bytes())
        graph = (
            LinksGraph.model_validate_json(graph_path.read_bytes())
            if graph_path.is_file()
            else LinksGraph(edges=[], dangling_edges=[])
        )
    except (ValidationError, ValueError) as exc:
        raise ExportError(
            f"{manifest_path} не читается как манифест экспорта ({exc}); "
            "укажите пустой каталог результата или удалите содержимое"
        ) from exc
    return PreviousExport(manifest=manifest, graph=graph)


def run_export(
    config: ExportConfig,
    seed_ids: list[UUID],
    gateway: TessaGateway,
    *,
    synthetic: bool = False,
    source: str = "tessa",
) -> RunSummary:
    export_root = export_dir(config.output_dir)
    if export_root.exists():
        logger.info("Очищаю предыдущий результат: %s", export_root)
        shutil.rmtree(export_root)
    return _export(
        config,
        seed_ids,
        gateway,
        traversal=config.traversal,
        rules=config.exclude_rules,
        previous=None,
        synthetic=synthetic,
        source=source,
        archive=True,
    )


def run_incremental_export(
    config: ExportConfig,
    card_ids: list[UUID],
    gateway: TessaGateway,
    *,
    traversal: TraversalSettings,
    extra_rules: Sequence[ExcludeRule] = (),
    source: str = "tessa",
    archive: bool = False,
) -> RunSummary:
    """Накопительный прогон: запрашиваются только карточки, которых ещё нет в каталоге."""
    export_root = export_dir(config.output_dir)
    previous = load_previous(export_root)
    known = previous.card_ids if previous else set()
    fresh = [card_id for card_id in dict.fromkeys(card_ids) if card_id not in known]
    logger.info(
        "Накопительный экспорт в %s: в каталоге %d документов, из %d запрошенных новых %d",
        export_root,
        len(known),
        len(card_ids),
        len(fresh),
    )
    return _export(
        config,
        fresh,
        gateway,
        traversal=traversal,
        rules=[*extra_rules, *config.exclude_rules],
        previous=previous,
        synthetic=False,
        source=source,
        archive=archive,
        known=len(card_ids) - len(fresh),
    )


def _export(
    config: ExportConfig,
    seed_ids: list[UUID],
    gateway: TessaGateway,
    *,
    traversal: TraversalSettings,
    rules: Sequence[ExcludeRule],
    previous: PreviousExport | None,
    synthetic: bool,
    source: str,
    archive: bool,
    known: int = 0,
) -> RunSummary:
    export_root = export_dir(config.output_dir)
    export_root.mkdir(parents=True, exist_ok=True)

    logger.info("Проверяю источник данных: %s", source)
    gateway.check_connection()

    logger.info(
        "Обход связей: seed=%d, max_depth=%d, max_docs=%d",
        len(seed_ids),
        traversal.max_depth,
        traversal.max_docs,
    )
    result = Walker(gateway, traversal, list(rules)).walk(seed_ids)
    # пустой результат — это провал, только если карточки запрашивались и ни одна не получена:
    # исключённые правилом (например, по состоянию) означают, что обход отработал штатно
    if seed_ids and not result.cards and not result.excluded and previous is None:
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

    manifest = build_manifest(
        result, file_records, config, synthetic=synthetic, source=source, traversal=traversal
    )
    graph = build_links_graph(result)
    added = manifest.stats.documents
    if previous is not None:
        manifest = merge_manifests(previous.manifest, manifest)
        graph = merge_links_graphs(
            previous.graph, graph, {document.card_id for document in manifest.documents}
        )
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

    archive_path = make_archive(export_root, config.output_dir / config.archive_name) if archive else None
    logger.info(
        "Готово: документов %d (новых %d), файлов скачано %d, итог валидации %s, архив %s",
        manifest.stats.documents,
        added,
        manifest.stats.files_downloaded,
        report.overall,
        archive_path or "не собирался",
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
        documents_added=added,
        documents_known=known,
    )


__all__ = [
    "ExportError",
    "GatewayError",
    "PreviousExport",
    "RunSummary",
    "load_previous",
    "run_export",
    "run_incremental_export",
]
