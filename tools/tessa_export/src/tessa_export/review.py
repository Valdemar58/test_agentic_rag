"""Таблица документов сета для ручного отбора состава (решение заказчика 2026-09-14).

Одна строка на документ, открывается в Excel (разделитель «;», UTF-8 с BOM). Заказчик отмечает
лишние документы, их ID (или целые типы карточек) переносятся в exclude_rules конфига, после чего
повторный экспорт даёт воспроизводимый сет без них.
"""

from __future__ import annotations

import csv
import io

from tessa_export.manifest import DocumentEntry, Manifest

REVIEW_COLUMNS = [
    "card_id",
    "тип карточки",
    "категория Тессы",
    "номер",
    "дата",
    "тема",
    "статус",
    "состояние маршрута",
    "глубина",
    "пришёл из",
    "связь",
    "файлов скачано",
    "категории 8.2",
    "исключить",
]


def _entry_via(document: DocumentEntry, by_id: dict[str, DocumentEntry]) -> tuple[str, str]:
    """Откуда документ попал в сет: seed или первый путь входа (документ-источник и тип связи)."""
    if any(path.kind == "seed" for path in document.entry_paths):
        return "seed", ""
    path = document.entry_paths[0]
    parent = by_id.get(str(path.via_card_id)) if path.via_card_id else None
    origin = f"{parent.doc_kind} №{parent.number or '?'}" if parent else str(path.via_card_id or "")
    return origin, path.relation or "без типа"


def render_review_csv(manifest: Manifest) -> str:
    by_id = {str(document.card_id): document for document in manifest.documents}
    buffer = io.StringIO()
    writer = csv.writer(buffer, delimiter=";", lineterminator="\n")
    writer.writerow(REVIEW_COLUMNS)
    ordered = sorted(
        manifest.documents, key=lambda item: (item.doc_kind, item.number or "", str(item.card_id))
    )
    for document in ordered:
        origin, relation = _entry_via(document, by_id)
        writer.writerow(
            [
                str(document.card_id),
                document.type_name or "",
                document.doc_kind,
                document.number or "",
                document.doc_date.isoformat() if document.doc_date else "",
                document.subject or "",
                document.doc_status,
                document.state_name or "",
                document.depth,
                origin,
                relation,
                sum(1 for file in document.files if file.downloaded),
                ", ".join(document.coverage_kinds),
                "",
            ]
        )
    return buffer.getvalue()
