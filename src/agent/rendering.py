"""Представление результатов инструментов для LLM (6.2) и человекочитаемые статусы шагов (FR-7).

LLM в цикле видит компактный русский текст с псевдонимами D#/S# вместо JSON с UUID; полные данные
остаются в реестре свидетельств для итогового ответа и цитат. Лимиты символов — `agent.tool_output`.
"""

from __future__ import annotations

import datetime as dt
import json
import re
from dataclasses import dataclass, field
from typing import Any

from agent.evidence import EvidenceRegistry, status_text
from common.config import AnswerSettings, ToolOutputSettings
from ingest.metadata import document_label

TOOL_SEARCH = "hybrid_search"
TOOL_CARD = "get_document_card"
TOOL_RELATED = "get_related_documents"
TOOL_CONTENT = "get_document_content"
TOOL_GLOSSARY = "glossary_lookup"

RELATION_SECTIONS = {"OutgoingRefDocs": "исходящая", "IncomingRefDocs": "входящая"}
DIRECTION_RU = {"outgoing": "исходящая", "incoming": "входящая"}
CUT_MARK = " …(обрезано)"
GUID_RE = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")
FALLBACK_LABEL = "документ"
NO_EVIDENCE = "(свидетельств нет: инструменты ничего не нашли)"


@dataclass(frozen=True)
class Rendered:
    text: str = field(metadata={"doc": "что видит LLM"})
    summary: str = field(metadata={"doc": "итог шага для UI"})
    fragment_aliases: list[str] = field(default_factory=list)
    document_aliases: list[str] = field(default_factory=list)


def cut(text: str, limit: int) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + CUT_MARK


def _label(kind: str | None, number: str | None, date_iso: str | None) -> str:
    date = None
    if date_iso:
        try:
            date = dt.date.fromisoformat(date_iso[:10])
        except ValueError:
            date = None
    return document_label(kind or FALLBACK_LABEL, number, date)


def _unique(items: list[str]) -> list[str]:
    seen: dict[str, None] = {}
    for item in items:
        seen.setdefault(item, None)
    return list(seen)


def _is_service_value(key: str, value: Any) -> bool:
    return key.endswith("ID") or (isinstance(value, str) and bool(GUID_RE.match(value)))


def _compact_mapping(mapping: dict[str, Any]) -> str:
    parts = []
    for key, value in mapping.items():
        if value in (None, "", [], {}) or _is_service_value(key, value):
            continue
        parts.append(f"{key}: {value}")
    return "; ".join(parts)


# ---------- статусы шагов (FR-7: «Ищу в приказах… → Найдено 4 документа») ----------


def _doc_ref(value: Any, registry: EvidenceRegistry) -> str:
    if isinstance(value, str):
        known = registry.document_by_alias(value) or registry.document(value)
        if known is not None:
            return f"{known.alias} {known.label}".strip()
    return str(value)


def status_text_for(name: str, arguments: dict[str, Any], registry: EvidenceRegistry) -> str:
    """Что агент делает сейчас — одной строкой для шага UI."""
    if name == TOOL_SEARCH:
        filters = arguments.get("filters") or {}
        extra = ""
        if isinstance(filters, dict):
            active = {key: value for key, value in filters.items() if value}
            if active:
                extra = "; фильтры: " + json.dumps(active, ensure_ascii=False)
        return f"Ищу: «{arguments.get('query', '')}»{extra}"
    if name == TOOL_CARD:
        return f"Читаю карточку {_doc_ref(arguments.get('doc_id'), registry)}"
    if name == TOOL_RELATED:
        relation = arguments.get("relation_type")
        suffix = f" (тип «{relation}»)" if relation else ""
        return f"Смотрю связи документа {_doc_ref(arguments.get('doc_id'), registry)}{suffix}"
    if name == TOOL_CONTENT:
        section = arguments.get("section_id")
        if section:
            return f"Читаю раздел {section} документа {_doc_ref(arguments.get('doc_id'), registry)}"
        return f"Читаю текст документа {_doc_ref(arguments.get('doc_id'), registry)}"
    if name == TOOL_GLOSSARY:
        return f"Ищу термин «{arguments.get('term', '')}» в глоссарии"
    return f"Вызываю {name}"


# ---------- результаты инструментов ----------


def render(
    name: str,
    arguments: dict[str, Any],
    data: dict[str, Any],
    registry: EvidenceRegistry,
    limits: ToolOutputSettings,
) -> Rendered:
    if name == TOOL_SEARCH:
        return _render_search(data, registry, limits)
    if name == TOOL_CARD:
        return _render_card(data, registry, limits)
    if name == TOOL_RELATED:
        return _render_related(data, registry)
    if name == TOOL_CONTENT:
        return _render_content(data, registry, limits)
    if name == TOOL_GLOSSARY:
        return _render_glossary(data)
    text = json.dumps(data, ensure_ascii=False)
    return Rendered(text=cut(text, limits.content_chars), summary=f"{name}: ответ получен")


def _render_search(data: dict[str, Any], registry: EvidenceRegistry, limits: ToolOutputSettings) -> Rendered:
    hits: list[dict[str, Any]] = data.get("hits") or []
    lines = [
        f"Найдено фрагментов: {len(hits)} (кандидатов после гибридного поиска: {data.get('candidates', 0)})"
    ]
    filters = data.get("applied_filters") or {}
    if filters:
        lines[0] += "; фильтры: " + json.dumps(filters, ensure_ascii=False)
    for note in data.get("notes") or []:
        lines.append(f"Замечание: {note}")
    if not hits:
        lines.append("Ничего не найдено: переформулируй запрос (синонимы, другие слова) или измени фильтры.")
    fragments: list[str] = []
    documents: list[str] = []
    for hit in hits:
        label = _label(hit.get("doc_kind"), hit.get("doc_number"), hit.get("doc_date"))
        document = registry.register_document(
            hit["doc_id"],
            label=label,
            doc_kind=hit.get("doc_kind"),
            doc_number=hit.get("doc_number"),
            doc_date=hit.get("doc_date"),
            doc_status=hit.get("doc_status"),
            doc_status_name=hit.get("doc_status_name"),
            subject=hit.get("subject"),
            department=hit.get("department"),
        )
        fragment = registry.register_fragment(
            hit["chunk_id"],
            doc_id=hit["doc_id"],
            kind="hit",
            breadcrumbs=hit.get("breadcrumbs") or label,
            text=hit.get("text") or "",
            parent_id=hit.get("parent_id"),
            clause=hit.get("clause"),
            page_no=hit.get("page_no"),
            context=hit.get("context"),
            score=hit.get("score"),
        )
        subject = f" «{hit['subject']}»" if hit.get("subject") else ""
        department = f", {hit['department']}" if hit.get("department") else ""
        lines.append(f"[{fragment.alias}] {document.alias} {label} ({document.status}{department}){subject}")
        lines.append(f"  {fragment.breadcrumbs}")
        lines.append(f"  {cut(fragment.text, limits.fragment_chars)}")
        fragments.append(fragment.alias)
        documents.append(document.alias)
    documents = _unique(documents)
    summary = f"Найдено фрагментов: {len(hits)}, документов: {len(documents)}"
    return Rendered(
        text="\n".join(lines), summary=summary, fragment_aliases=fragments, document_aliases=documents
    )


def _relation_lines(sections: dict[str, Any], registry: EvidenceRegistry) -> tuple[list[str], list[str]]:
    lines: list[str] = []
    aliases: list[str] = []
    for section_name, direction in RELATION_SECTIONS.items():
        section = sections.get(section_name) or {}
        for row in section.get("rows") or []:
            doc_id = row.get("DocID")
            if not doc_id:
                continue
            related = registry.register_document(
                str(doc_id), label=row.get("DocDescription") or row.get("DocTypeName"), authoritative=False
            )
            relation = row.get("RefTypeName") or "тип не указан"
            lines.append(f"[{related.alias}] {related.label or FALLBACK_LABEL} — {relation} ({direction})")
            aliases.append(related.alias)
    return lines, aliases


def _render_card(data: dict[str, Any], registry: EvidenceRegistry, limits: ToolOutputSettings) -> Rendered:
    summary = data.get("summary") or {}
    document = registry.register_document(
        data["doc_id"],
        label=summary.get("label"),
        doc_kind=summary.get("doc_kind"),
        doc_number=summary.get("doc_number"),
        doc_date=summary.get("doc_date"),
        doc_status=summary.get("doc_status"),
        doc_status_name=summary.get("doc_status_name"),
        subject=summary.get("subject"),
        department=summary.get("department"),
    )
    status = document.status + (f" ({summary['doc_status_name']})" if summary.get("doc_status_name") else "")
    lines = [f"[{document.alias}] {document.label} — статус: {status}"]
    scalar_fields = [
        ("Тема", "subject"),
        ("Подразделение", "department"),
        ("Автор", "author"),
        ("Подписал", "signed_by"),
        ("Состояние маршрута", "state_name"),
        ("Согласование", "approval_state_name"),
        ("Срок действия", "validity_period"),
        ("Комментарий", "comment"),
    ]
    for title, key in scalar_fields:
        if summary.get(key):
            lines.append(f"{title}: {summary[key]}")
    for title, key in (("Согласующие", "approvers"), ("Ответственные", "responsible"), ("Файлы", "files")):
        if summary.get(key):
            lines.append(f"{title}: {', '.join(summary[key])}")
    sections: dict[str, Any] = (data.get("card") or {}).get("sections") or {}
    relation_lines, related_aliases = _relation_lines(sections, registry)
    if relation_lines:
        lines.append("Связи:")
        lines.extend(f"  {line}" for line in relation_lines)
        document.relations = relation_lines
    document.card_text = "\n".join(lines)
    detail: list[str] = []
    for section_name, section in sections.items():
        if section_name in RELATION_SECTIONS:
            continue
        fields = _compact_mapping(section.get("fields") or {})
        rows = [_compact_mapping(row) for row in section.get("rows") or []]
        rows = [row for row in rows if row]
        if fields or rows:
            detail.append(f"Секция {section_name}: {fields}" if fields else f"Секция {section_name}:")
            detail.extend(f"  - {row}" for row in rows)
    if detail:
        lines.append(cut("\n".join(detail), limits.card_chars))
    omitted = data.get("omitted_sections") or []
    if omitted:
        lines.append("Секции не показаны (full=true покажет): " + ", ".join(omitted))
    summary_text = f"Карточка: {document.label} ({document.status})"
    return Rendered(
        text="\n".join(lines),
        summary=summary_text,
        document_aliases=_unique([document.alias, *related_aliases]),
    )


def _render_related(data: dict[str, Any], registry: EvidenceRegistry) -> Rendered:
    document = registry.register_document(data["doc_id"])
    related: list[dict[str, Any]] = data.get("related") or []
    lines = [f"Связи документа [{document.alias}] {document.label or FALLBACK_LABEL}: {len(related)}"]
    aliases = [document.alias]
    relation_lines: list[str] = []
    for item in related:
        known = registry.register_document(
            str(item["doc_id"]),
            label=item.get("description") or item.get("doc_type_name"),
            authoritative=False,
        )
        relation = item.get("relation_type") or "тип не указан"
        direction = DIRECTION_RU.get(item.get("direction", ""), item.get("direction", ""))
        line = f"[{known.alias}] {known.label or FALLBACK_LABEL} — {relation} ({direction})"
        relation_lines.append(line)
        aliases.append(known.alias)
    lines.extend(f"  {line}" for line in relation_lines)
    if data.get("relation_types"):
        lines.append("Типы связей у документа: " + ", ".join(data["relation_types"]))
    if data.get("note"):
        lines.append(f"Замечание: {data['note']}")
    if relation_lines:
        document.relations = relation_lines
    types = ", ".join(data.get("relation_types") or []) or "без типа"
    summary = f"Связей: {len(related)} ({types})" if related else "Связей нет"
    return Rendered(text="\n".join(lines), summary=summary, document_aliases=_unique(aliases))


def _render_content(data: dict[str, Any], registry: EvidenceRegistry, limits: ToolOutputSettings) -> Rendered:
    label = data.get("label") or _label(data.get("doc_kind"), data.get("doc_number"), data.get("doc_date"))
    document = registry.register_document(
        data["doc_id"],
        label=label,
        doc_kind=data.get("doc_kind"),
        doc_number=data.get("doc_number"),
        doc_date=data.get("doc_date"),
        doc_status=data.get("doc_status"),
        subject=data.get("subject"),
    )
    sections: list[dict[str, Any]] = data.get("sections") or []
    offset = int(data.get("offset", 0))
    total = int(data.get("total_sections", len(sections)))
    header = f"[{document.alias}] {document.label} ({document.status}): разделов {total}"
    if sections:
        header += f", показаны {offset + 1}–{offset + len(sections)}"
    if data.get("truncated") and data.get("next_offset") is not None:
        header += f"; продолжение: offset={data['next_offset']}"
    lines = [header]
    if data.get("note"):
        lines.append(f"Замечание: {data['note']}")
    fragments: list[str] = []
    used = 0
    for index, section in enumerate(sections):
        fragment = registry.register_fragment(
            section["section_id"],
            doc_id=data["doc_id"],
            kind="section",
            breadcrumbs=section.get("breadcrumbs") or document.label,
            text=section.get("text") or "",
            parent_id=section["section_id"],
            page_no=section.get("page_no"),
        )
        fragments.append(fragment.alias)
        page = f" (стр. {section['page_no']})" if section.get("page_no") else ""
        block = f"[{fragment.alias}] {fragment.breadcrumbs}{page}\n{cut(fragment.text, limits.section_chars)}"
        if used + len(block) > limits.content_chars and index > 0:
            rest, next_offset = len(sections) - index, offset + index
            lines.append(f"(остальные {rest} разделов не показаны — запроси их через offset={next_offset})")
            break
        lines.append(block)
        used += len(block)
    summary = f"Прочитано разделов: {len(fragments)} из {total}"
    return Rendered(
        text="\n".join(lines), summary=summary, fragment_aliases=fragments, document_aliases=[document.alias]
    )


def _render_glossary(data: dict[str, Any]) -> Rendered:
    entries: list[dict[str, Any]] = data.get("entries") or []
    lines = [f"Глоссарий, термин «{data.get('term', '')}»: определений {len(entries)}"]
    for entry in entries:
        lines.append(f"- {entry.get('term')}: {entry.get('definition')} (источник: {entry.get('doc_label')})")
    if data.get("note"):
        lines.append(f"Замечание: {data['note']}")
    summary = f"Определений в глоссарии: {len(entries)}" if entries else "Термин в глоссарии не найден"
    return Rendered(text="\n".join(lines), summary=summary)


# ---------- свидетельства для итогового ответа ----------


def render_evidence(
    registry: EvidenceRegistry,
    fragment_aliases: list[str],
    document_aliases: list[str],
    settings: AnswerSettings,
) -> str:
    """Документы и фрагменты прогона для промпта итогового ответа, в бюджете `evidence_max_chars`."""
    if not fragment_aliases and not document_aliases:
        return NO_EVIDENCE
    lines = ["Документы:"]
    for alias in _unique(document_aliases):
        document = registry.document_by_alias(alias)
        if document is None:
            continue
        line = f"[{document.alias}] {document.label or FALLBACK_LABEL} — {document.status}"
        if document.subject:
            line += f"; тема: «{document.subject}»"
        if document.department:
            line += f"; подразделение: {document.department}"
        lines.append(line)
        for relation in document.relations[:8]:
            lines.append(f"    связь: {relation}")
    lines.append("Фрагменты:")
    used = sum(len(line) for line in lines)
    shown = 0
    for alias in _unique(fragment_aliases):
        fragment = registry.fragment_by_alias(alias)
        if fragment is None:
            continue
        document = registry.document(fragment.doc_id)
        status = document.status if document else status_text(None)
        page = f", стр. {fragment.page_no}" if fragment.page_no else ""
        head = f"[{fragment.alias}] ({fragment.doc_alias}, {status}{page}) {fragment.breadcrumbs}"
        block = f"{head}\n{fragment.text}"
        if fragment.kind == "hit" and fragment.context and settings.context_chars:
            context = fragment.context.strip()
            if context and context != fragment.text.strip():
                block += "\n  Контекст раздела: " + cut(context, settings.context_chars)
        if used + len(block) > settings.evidence_max_chars and shown:
            lines.append("(остальные фрагменты не поместились в бюджет ответа)")
            break
        lines.append(block)
        used += len(block)
        shown += 1
    return "\n".join(lines)
