"""Отчёт инжеста (AC-3.1): JSON и markdown с ошибками по файлам и причинами."""

from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

from ingest.corpus import CorpusFile, CorpusIssue
from ingest.files import FilePlan
from ingest.pipeline import FileOutcome
from ingest.report import render_markdown, write_report
from ingest.run import RunReport


def _outcome(name: str, status: str, **extra: object) -> FileOutcome:
    card = uuid4()
    file = CorpusFile(
        card_id=card,
        row_id=uuid4(),
        name=name,
        extension=name.rsplit(".", 1)[-1],
        category="Документ",
        relative_path=f"files/{card}/{name}",
        path=Path(name),
        sha256="0" * 64,
        size=1,
        has_text_layer=None,
        page_count=None,
        duplicate_of=None,
        smoke_note=None,
    )
    return FileOutcome(plan=FilePlan(file=file, role="main"), status=status, **extra)  # type: ignore[arg-type]


def test_report_files_contain_summary_errors_and_routes(tmp_path: Path) -> None:
    report = RunReport(
        run_id=7, outcome="partial", files_total=3, indexed=1, failed=1, skipped=1, chunks_total=5
    )
    report.outcomes = [
        _outcome("приказ.docx", "indexed", route="native", chunks=5, parents=2, seconds=1.2),
        _outcome(
            "скан.pdf",
            "error",
            route="vlm",
            reason="разбор (vlm): failure; timeout | сервер",
            parse_errors=["timeout"],
        ),
    ]
    report.issues = [
        CorpusIssue("card_missing", "files/x/лишний.docx", uuid4(), None, "файл карточки не найден")
    ]
    report.removed_files = ["a/b"]
    json_path, md_path = write_report(report, tmp_path)
    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert data["run_id"] == 7 and data["outcome"] == "partial" and data["success_share"] == 0.5
    assert [item["status"] for item in data["files"]] == ["indexed", "error"]
    assert data["files"][1]["reason"].startswith("разбор (vlm)") and data["files"][1]["parse_errors"] == [
        "timeout"
    ]
    assert data["issues"][0]["kind"] == "card_missing" and data["removed_files"] == ["a/b"]
    markdown = md_path.read_text(encoding="utf-8")
    assert "# Отчёт инжеста #7" in markdown and "## Файлы с ошибками" in markdown
    assert "скан.pdf | vlm | разбор (vlm): failure; timeout \\| сервер" in markdown
    assert "- native: 1" in markdown and "нет карточки" in markdown and "- a/b" in markdown
    assert "Доля успешно обработанных (AC-3.1): 50.0% из 2" in render_markdown(report)
