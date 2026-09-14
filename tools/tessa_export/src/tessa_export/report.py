"""Рендер validation_report.md: однозначный итог, проверки 8.3, покрытие 8.2, справочные списки."""

from __future__ import annotations

from tessa_export.manifest import Manifest
from tessa_export.validation import ValidationReport

STATUS_MARK = {"PASS": "✅ PASS", "FAIL": "❌ FAIL", "WARN": "⚠️ WARN", "INFO": "ℹ️ INFO"}
MAX_DETAILS = 50


def _by_extension(manifest: Manifest) -> str:
    items = sorted(manifest.stats.skipped_by_extension.items())
    return ", ".join(f"{ext}: {count}" for ext, count in items) or "—"


def _details(lines: list[str]) -> list[str]:
    shown = lines[:MAX_DETAILS]
    rendered = [f"  - {line}" for line in shown]
    if len(lines) > MAX_DETAILS:
        rendered.append(f"  - … ещё {len(lines) - MAX_DETAILS}")
    return rendered


def render_report(manifest: Manifest, report: ValidationReport) -> str:
    verdict = "ПРИГОДЕН" if report.overall == "PASS" else "НЕ ПРИГОДЕН"
    lines: list[str] = ["# Отчёт валидации голден-корпуса", ""]
    if manifest.synthetic:
        lines += ["> **Данные синтетические** (режим самопроверки), для метрик не использовать.", ""]
    lines += [
        f"**ИТОГ: {verdict}** — проверок 8.3 провалено: {len(report.failed)}, "
        f"дефицитов покрытия 8.2: {len(report.coverage_deficits)}.",
        "",
        f"Создан: {manifest.created_at.isoformat()}. Seed-карточек: {len(manifest.seed_ids)}. "
        f"Обход: глубина ≤ {manifest.traversal.get('max_depth')}, "
        f"лимит {manifest.traversal.get('max_docs')}, "
        f"направления {', '.join(manifest.traversal.get('directions', []))}.",
        "",
        "## Сводка",
        "",
        "| Показатель | Значение |",
        "|---|---|",
        f"| Документов в сете | {manifest.stats.documents} |",
        f"| Исключено правилами | {manifest.stats.excluded} |",
        f"| Ошибок получения карточек | {manifest.stats.errors} |",
        "| Файлов всего / скачано / пропущено | "
        f"{manifest.stats.files_total} / {manifest.stats.files_downloaded} / "
        f"{manifest.stats.files_skipped} |",
        f"| Пропущено по формату | {_by_extension(manifest)} |",
        f"| Файлов-дублей по хэшу | {manifest.stats.duplicate_files} |",
        f"| Непройденных связей | {manifest.stats.skipped_links} |",
        "",
        "Таблица для ручного отбора состава сета: `documents_review.csv` рядом с этим отчётом.",
        "",
        "## Проверки 8.3",
        "",
        "| Статус | Проверка | Результат |",
        "|---|---|---|",
    ]
    for check in report.checks:
        lines.append(f"| {STATUS_MARK[check.status]} | {check.name} | {check.summary} |")
    for check in report.checks:
        if check.details and check.status != "PASS":
            lines += ["", f"### {check.name}: детали", ""]
            lines += _details(check.details)
    lines += [
        "",
        "## Покрытие 8.2 (ориентиры; дефицит — предупреждение, не ошибка)",
        "",
        "| Статус | Показатель | Ориентир | Факт | Рекомендация |",
        "|---|---|---|---|---|",
    ]
    for row in report.coverage:
        mark = "✅" if row.ok else "⚠️"
        lines.append(f"| {mark} | {row.name} | {row.target} | {row.actual} | {'' if row.ok else row.hint} |")

    lines += ["", "## Состав сета", "", "| Категория Тессы (DocTypeTitle) | Документов |", "|---|---|"]
    for kind, count in sorted(manifest.stats.doc_kinds.items(), key=lambda item: -item[1]):
        lines.append(f"| {kind} | {count} |")
    lines += [
        "",
        "| Категория 8.2 по виду и теме (документ может быть в нескольких) | Документов |",
        "|---|---|",
    ]
    for kind, count in sorted(manifest.stats.coverage_kinds.items(), key=lambda item: -item[1]):
        lines.append(f"| {kind} | {count} |")
    lines += ["", "| Статус документа (правило `status` конфига) | Документов |", "|---|---|"]
    for status, count in sorted(manifest.stats.doc_statuses.items(), key=lambda item: -item[1]):
        lines.append(f"| {status} | {count} |")
    lines += ["", "| Тип карточки | Документов |", "|---|---|"]
    for type_name, count in sorted(manifest.stats.card_types.items(), key=lambda item: -item[1]):
        lines.append(f"| {type_name} | {count} |")

    lines += ["", "## Встреченные статусы документов (StatusID → название)", ""]
    lines += [
        f"- `{status_id}` → {name or '—'}" for status_id, name in manifest.stats.status_values.items()
    ] or ["- нет"]
    lines += ["", "## Встреченные типы связей (прямое → обратное название)", ""]
    lines += [f"- {name} → {reverse or '—'}" for name, reverse in manifest.stats.relation_types.items()] or [
        "- нет"
    ]

    if manifest.excluded:
        lines += ["", "## Исключённые документы", ""]
        lines += [
            f"- {item.card_id}: {item.reason} (глубина {item.depth}, {item.doc_type_title or item.type_name})"
            for item in manifest.excluded
        ]
    if manifest.errors:
        lines += ["", "## Ошибки получения карточек", ""]
        lines += [
            f"- {item.card_id} (глубина {item.depth}, {item.kind}): {item.message}"
            for item in manifest.errors
        ]
    if manifest.skipped_links:
        lines += ["", "## Непройденные связи", ""]
        lines += _details(
            [
                f"{item.from_card_id or 'seed'} → {item.to_card_id}: {item.reason} ({item.detail}); "
                f"связь: {item.relation or 'без типа'}"
                for item in manifest.skipped_links
            ]
        )
    lines.append("")
    return "\n".join(lines)
