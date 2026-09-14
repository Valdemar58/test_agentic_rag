"""Валидация голден-корпуса (§8.3) и отчёт покрытия (§8.2).

Проверки 8.3 дают PASS/FAIL, ориентиры 8.2 — WARN с рекомендацией по добору. Итог PASS,
если ни одна проверка 8.3 не провалена; дефицит покрытия не блокирует экспорт.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal
from uuid import UUID

from tessa_export.config import CoverageSettings
from tessa_export.manifest import DocumentEntry, LinksGraph, Manifest

Status = Literal["PASS", "FAIL", "WARN", "INFO"]

IMPORTANT_FIELDS = ("number", "doc_date", "subject", "status_id", "department")
CANCEL_OR_CHANGE = re.compile(r"отмен|измен", re.IGNORECASE)
ATTACHMENT = re.compile(r"прилож", re.IGNORECASE)


@dataclass
class Check:
    name: str
    status: Status
    summary: str
    details: list[str] = field(default_factory=list)


@dataclass
class CoverageRow:
    name: str
    target: int | str
    actual: int | str
    ok: bool
    hint: str = ""


@dataclass
class ValidationReport:
    checks: list[Check]
    coverage: list[CoverageRow]
    overall: Literal["PASS", "FAIL"]

    @property
    def failed(self) -> list[Check]:
        return [check for check in self.checks if check.status == "FAIL"]

    @property
    def coverage_deficits(self) -> list[CoverageRow]:
        return [row for row in self.coverage if not row.ok]


def _label(document: DocumentEntry) -> str:
    return f"{document.doc_kind} №{document.number or '?'} ({document.card_id})"


def check_files_downloaded(manifest: Manifest) -> Check:
    failed = [
        f"{_label(document)}: «{file.name}» — {file.skipped_detail}"
        for document in manifest.documents
        for file in document.files
        if file.skipped_reason == "error"
    ]
    skipped = manifest.stats.skipped_by_reason
    summary = (
        f"скачано {manifest.stats.files_downloaded} из {manifest.stats.files_total}; "
        f"пропущено по формату: {skipped.get('format', 0)}, виртуальных: {skipped.get('virtual', 0)}"
    )
    if failed:
        return Check("Файлы скачаны", "FAIL", summary + f"; ошибок скачивания: {len(failed)}", failed)
    return Check("Файлы скачаны", "PASS", summary)


def check_files_open(manifest: Manifest) -> Check:
    broken = [
        f"{_label(document)}: «{file.name}» — {file.smoke_error}"
        for document in manifest.documents
        for file in document.files
        if file.downloaded and file.smoke_ok is False
    ]
    total = sum(1 for document in manifest.documents for file in document.files if file.downloaded)
    if broken:
        return Check("Файлы открываются", "FAIL", f"не открываются {len(broken)} из {total}", broken)
    return Check("Файлы открываются", "PASS", f"открыты все {total} скачанных файла(ов)")


def check_cards_valid(manifest: Manifest) -> Check:
    empties: list[str] = []
    for document in manifest.documents:
        missing = [name for name in IMPORTANT_FIELDS if getattr(document, name) in (None, "")]
        if missing:
            empties.append(f"{_label(document)}: пустые поля {', '.join(missing)}")
    summary = f"все {len(manifest.documents)} карточек сериализованы по схеме CardData"
    if empties:
        return Check(
            "Карточки валидны по схеме", "WARN", summary + f"; с пустыми полями: {len(empties)}", empties
        )
    return Check("Карточки валидны по схеме", "PASS", summary)


def check_links_complete(manifest: Manifest, graph: LinksGraph) -> Check:
    in_set = {document.card_id for document in manifest.documents}
    known: set[UUID] = in_set | {item.card_id for item in manifest.excluded}
    known |= {item.card_id for item in manifest.errors} | {item.to_card_id for item in manifest.skipped_links}
    unexplained = [
        f"{edge.from_id} → {edge.to_id} ({edge.relation_type or 'без типа'})"
        for edge in graph.dangling_edges
        if edge.from_id not in known or edge.to_id not in known
    ]
    if unexplained:
        return Check(
            "Полнота обхода связей", "FAIL", f"висячих связей без причины: {len(unexplained)}", unexplained
        )
    by_reason: dict[str, int] = {}
    for item in manifest.skipped_links:
        by_reason[item.reason] = by_reason.get(item.reason, 0) + 1
    details = [
        f"{item.from_card_id or 'seed'} → {item.to_card_id}: {item.reason} ({item.detail})"
        for item in manifest.skipped_links
    ]
    summary = (
        f"рёбер внутри сета: {len(graph.edges)}; непройденных связей: {len(manifest.skipped_links)} "
        + ", ".join(f"{reason}: {count}" for reason, count in sorted(by_reason.items()))
    )
    status: Status = "WARN" if by_reason.get("max_depth") or by_reason.get("max_docs") else "PASS"
    return Check("Полнота обхода связей", status, summary, details)


def check_dedup(manifest: Manifest) -> Check:
    ids = [document.card_id for document in manifest.documents]
    duplicate_ids = sorted({str(item) for item in ids if ids.count(item) > 1})
    if duplicate_ids:
        return Check("Дедупликация", "FAIL", "повторяющиеся карточки в сете", duplicate_ids)
    multi_path = sum(1 for document in manifest.documents if len(document.entry_paths) > 1)
    summary = (
        f"карточек: {len(ids)}, уникальны; с несколькими путями входа: {multi_path}; "
        f"файлов-дублей по хэшу: {manifest.stats.duplicate_files}"
    )
    return Check("Дедупликация", "PASS", summary)


def check_exclusions(manifest: Manifest) -> Check:
    in_set = {document.card_id for document in manifest.documents}
    leaked = [str(item.card_id) for item in manifest.excluded if item.card_id in in_set]
    if leaked:
        return Check("Исключения", "FAIL", "исключённые документы попали в сет", leaked)
    details = [f"{item.card_id}: {item.reason}" for item in manifest.excluded]
    return Check("Исключения", "PASS", f"исключено документов: {len(manifest.excluded)}", details)


def check_fetch_errors(manifest: Manifest) -> Check:
    if not manifest.errors:
        return Check("Получение карточек", "PASS", "все карточки получены")
    seed_ids = set(manifest.seed_ids)
    details = [
        f"{item.card_id} (глубина {item.depth}, {item.kind}): {item.message}" for item in manifest.errors
    ]
    seed_failed = [item for item in manifest.errors if item.card_id in seed_ids]
    if seed_failed:
        return Check("Получение карточек", "FAIL", f"не получены seed-карточки: {len(seed_failed)}", details)
    return Check(
        "Получение карточек", "WARN", f"не получены связанные карточки: {len(manifest.errors)}", details
    )


def _is_scan(file_extension: str, has_text_layer: bool | None, image_extensions: set[str]) -> bool:
    return file_extension in image_extensions or (file_extension == "pdf" and has_text_layer is False)


def coverage_rows(manifest: Manifest, graph: LinksGraph, coverage: CoverageSettings) -> list[CoverageRow]:
    documents = manifest.documents
    by_kind: dict[str, list[DocumentEntry]] = {}
    for document in documents:
        by_kind.setdefault(document.doc_kind, []).append(document)
    rows: list[CoverageRow] = []
    for kind, target in coverage.targets.items():
        actual = len(by_kind.get(kind, []))
        rows.append(
            CoverageRow(kind, target, actual, actual >= target, "добавьте документы этого вида в seed")
        )

    orders = by_kind.get("Приказы", [])
    order_ids = {document.card_id for document in orders}
    cancelled = sum(1 for document in orders if document.is_cancelled)
    rows.append(
        CoverageRow(
            "Приказы: отменённых",
            coverage.min_cancelled_orders,
            cancelled,
            cancelled >= coverage.min_cancelled_orders,
        )
    )
    linked = {
        edge.from_id
        for edge in graph.edges
        if edge.from_id in order_ids
        and edge.to_id in order_ids
        and CANCEL_OR_CHANGE.search(edge.relation_type or "")
    } | {
        edge.to_id
        for edge in graph.edges
        if edge.from_id in order_ids
        and edge.to_id in order_ids
        and CANCEL_OR_CHANGE.search(edge.relation_type or "")
    }
    rows.append(
        CoverageRow(
            "Приказы: в связке «отменяет/изменяет»",
            coverage.min_linked_orders,
            len(linked),
            len(linked) >= coverage.min_linked_orders,
        )
    )
    orders_terms = sum(1 for document in orders if any(file.has_terms_section for file in document.files))
    rows.append(
        CoverageRow(
            "Приказы: с разделом терминов",
            coverage.min_orders_with_terms,
            orders_terms,
            orders_terms >= coverage.min_orders_with_terms,
        )
    )

    regulations = by_kind.get("Положения / ЛНА", [])
    regulations_terms = sum(
        1 for document in regulations if any(file.has_terms_section for file in document.files)
    )
    rows.append(
        CoverageRow(
            "Положения: с разделом терминов",
            coverage.min_regulations_with_terms,
            regulations_terms,
            regulations_terms >= coverage.min_regulations_with_terms,
        )
    )

    instructions = by_kind.get("Инструкции", [])
    instructions_tables = sum(
        1 for document in instructions if any(file.has_tables for file in document.files)
    )
    rows.append(
        CoverageRow(
            "Инструкции: с таблицами",
            coverage.min_instructions_with_tables,
            instructions_tables,
            instructions_tables >= coverage.min_instructions_with_tables,
        )
    )

    contracts = by_kind.get("Договоры", [])
    contract_ids = {document.card_id for document in contracts}
    with_attachments = sum(
        1
        for document in contracts
        if sum(1 for file in document.files if file.downloaded) >= 2
        or any(
            edge.from_id == document.card_id and ATTACHMENT.search(edge.relation_type or "")
            for edge in graph.edges
        )
        or any(
            edge.to_id == document.card_id and ATTACHMENT.search(edge.reverse_type or "")
            for edge in graph.edges
        )
    )
    rows.append(
        CoverageRow(
            "Договоры: с допсоглашениями/приложениями",
            coverage.min_contracts_with_attachments,
            with_attachments,
            with_attachments >= coverage.min_contracts_with_attachments,
        )
    )
    del contract_ids

    memos = by_kind.get("Служебные записки", [])
    memo_ids = {document.card_id for document in memos}
    memos_linked = len(
        {edge.from_id for edge in graph.edges if edge.from_id in memo_ids}
        | {edge.to_id for edge in graph.edges if edge.to_id in memo_ids}
    )
    rows.append(
        CoverageRow(
            "Служебные записки: со ссылками",
            coverage.min_memos_with_links,
            memos_linked,
            memos_linked >= coverage.min_memos_with_links,
        )
    )

    downloaded = [file for document in documents for file in document.files if file.downloaded]
    image_extensions = {"png", "jpg", "jpeg", "tif", "tiff", "bmp", "gif"}
    scans = sum(1 for file in downloaded if _is_scan(file.extension, file.has_text_layer, image_extensions))
    digital = sum(1 for file in downloaded if file.has_text_layer is True)
    total = len(downloaded) or 1
    digital_share = digital / total
    scan_share = scans / total
    rows.append(
        CoverageRow(
            "Файлы с текстовым слоем, доля",
            f"≥{coverage.min_digital_share:.0%}",
            f"{digital_share:.0%} ({digital})",
            digital_share >= coverage.min_digital_share,
        )
    )
    rows.append(
        CoverageRow(
            "Сканы, доля",
            f"≥{coverage.min_scan_share:.0%}",
            f"{scan_share:.0%} ({scans})",
            scan_share >= coverage.min_scan_share,
            "добавьте документы со сканами для проверки ветки dots.mocr",
        )
    )

    with_tables = sum(1 for document in documents if any(file.has_tables for file in document.files))
    rows.append(
        CoverageRow(
            "Документы с таблицами",
            coverage.min_docs_with_tables,
            with_tables,
            with_tables >= coverage.min_docs_with_tables,
        )
    )

    dates = [document.doc_date for document in documents if document.doc_date]
    span_years = (max(dates).year - min(dates).year) if dates else 0
    rows.append(
        CoverageRow(
            "Разброс дат, лет",
            f"≥{coverage.min_date_span_years}",
            span_years,
            span_years >= coverage.min_date_span_years,
            "для вопросов на актуальность нужны документы разных лет",
        )
    )
    return rows


def validate(manifest: Manifest, graph: LinksGraph, coverage: CoverageSettings) -> ValidationReport:
    checks = [
        check_files_downloaded(manifest),
        check_files_open(manifest),
        check_cards_valid(manifest),
        check_links_complete(manifest, graph),
        check_dedup(manifest),
        check_exclusions(manifest),
        check_fetch_errors(manifest),
    ]
    rows = coverage_rows(manifest, graph, coverage)
    overall: Literal["PASS", "FAIL"] = "FAIL" if any(check.status == "FAIL" for check in checks) else "PASS"
    return ValidationReport(checks=checks, coverage=rows, overall=overall)
