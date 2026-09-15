"""Отчёт о прогоне инжеста (AC-3.1): JSON для машин и markdown для людей, в рабочем каталоге."""

from __future__ import annotations

import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from ingest.run import RunReport

REPORTS_DIR = "reports"


def report_payload(report: RunReport) -> dict[str, object]:
    return {
        "run_id": report.run_id,
        "outcome": report.outcome,
        "created_at": datetime.now(UTC).isoformat(),
        "seconds": round(report.seconds, 1),
        "parse_seconds": round(report.parse_seconds, 1),
        "index_seconds": round(report.index_seconds, 1),
        "files_total": report.files_total,
        "indexed": report.indexed,
        "unchanged": report.unchanged,
        "skipped": report.skipped,
        "failed": report.failed,
        "removed": report.removed,
        "chunks_total": report.chunks_total,
        "success_share": round(report.success_share, 4),
        "error_text": report.error_text,
        "issues": [
            {"kind": issue.kind, "path": issue.path, "card_id": str(issue.card_id), "detail": issue.detail}
            for issue in report.issues
        ],
        "removed_files": report.removed_files,
        "files": [
            {
                "path": outcome.plan.file.relative_path,
                "card_id": str(outcome.plan.file.card_id),
                "file_row_id": str(outcome.plan.file.row_id),
                "role": outcome.plan.role,
                "status": outcome.status,
                "reason": outcome.reason,
                "route": outcome.route,
                "chunks": outcome.chunks,
                "parents": outcome.parents,
                "pages": outcome.pages,
                "seconds": round(outcome.seconds, 1),
                "parse_errors": outcome.parse_errors,
            }
            for outcome in report.outcomes
        ],
    }


def render_markdown(report: RunReport) -> str:
    lines = [f"# Отчёт инжеста #{report.run_id}", ""]
    lines.extend(f"- {line}" for line in report.summary_lines())
    errors = [outcome for outcome in report.outcomes if outcome.status == "error"]
    if errors:
        lines += ["", "## Файлы с ошибками", "", "| Файл | Маршрут | Причина |", "|---|---|---|"]
        for outcome in errors:
            reason = (outcome.reason or "").replace("|", "\\|")
            lines.append(f"| {outcome.plan.file.relative_path} | {outcome.route or '—'} | {reason} |")
    routes = Counter(outcome.route for outcome in report.outcomes if outcome.indexed)
    if routes:
        lines += ["", "## Проиндексировано по маршрутам", ""]
        lines.extend(
            f"- {route}: {count}" for route, count in sorted(routes.items(), key=lambda x: str(x[0]))
        )
    if report.issues:
        lines += ["", "## Не допущено загрузчиком архива", ""]
        lines.extend(f"- {issue.title}: {issue.path} — {issue.detail}" for issue in report.issues)
    if report.removed_files:
        lines += ["", "## Удалено из индекса", ""]
        lines.extend(f"- {item}" for item in report.removed_files)
    return "\n".join(lines) + "\n"


def write_report(report: RunReport, work_dir: Path) -> tuple[Path, Path]:
    target = work_dir / REPORTS_DIR
    target.mkdir(parents=True, exist_ok=True)
    json_path = target / f"ingest_{report.run_id}.json"
    md_path = target / f"ingest_{report.run_id}.md"
    json_path.write_text(json.dumps(report_payload(report), ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    return json_path, md_path
