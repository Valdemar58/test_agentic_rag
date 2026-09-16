"""`get_document_card` (FR-2.2): карточка в форме `CardData` сервиса карточек плюс краткая сводка.

Полная карточка (медиана 170 КБ JSON) не помещается в контекст LLM, поэтому по умолчанию в ответ
попадают секции из `card_service.default_sections` (N21 [ТРЕБУЕТ ПОДТВЕРЖДЕНИЯ]); поля внутри секций
не переименовываются и не теряются, полная карточка — по `full=True`. Сводка считается по тому же
маппингу, что метаданные чанков (`ingest.metadata.card_metadata`), поэтому статус документа в
карточке и в поиске совпадает.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from common.config import CardServiceSettings, DocStatus, StatusRuleSettings
from ingest.cards import CardRecord
from ingest.metadata import card_metadata
from tessa_export.manifest import LinksGraph

PERMISSIONS_FIELD = "permissions"
SECTIONS_FIELD = "sections"


class CardSummary(BaseModel):
    label: str = Field(description="Документ: вид, номер, дата")
    doc_kind: str
    doc_number: str | None
    doc_date: str | None
    subject: str | None
    doc_status: DocStatus = Field(
        description="active — действует, cancelled — отменён, draft — проект/в работе"
    )
    doc_status_name: str | None = Field(description="Статус документа, как в СЭД")
    state_name: str | None = Field(description="Состояние маршрута (KrStates)")
    approval_state_name: str | None = Field(description="Состояние согласования")
    department: str | None
    author: str | None
    signed_by: str | None
    validity_period: str | None
    comment: str | None
    approvers: list[str]
    responsible: list[str]
    files: list[str] = Field(description="Имена файлов карточки")


class DocumentCard(BaseModel):
    doc_id: str
    summary: CardSummary
    card: dict[str, Any] = Field(
        description="Карточка в форме CardData сервиса карточек (секции — выбранные)"
    )
    included_sections: list[str]
    omitted_sections: list[str] = Field(description="Секции, не вошедшие в ответ; доступны через full=true")
    full: bool


def _select_sections(
    all_sections: dict[str, Any], wanted: list[str] | None
) -> tuple[dict[str, Any], list[str]]:
    if wanted is None:
        return dict(all_sections), []
    wanted_folded = {name.casefold() for name in wanted}
    included = {name: section for name, section in all_sections.items() if name.casefold() in wanted_folded}
    omitted = [name for name in all_sections if name not in included]
    return included, omitted


def build_document_card(
    card_json: dict[str, Any],
    *,
    sections: list[str] | None,
    full: bool,
    settings: CardServiceSettings,
    status_rules: StatusRuleSettings,
) -> DocumentCard:
    record = CardRecord.model_validate(card_json)
    meta = card_metadata(record, LinksGraph(edges=[], dangling_edges=[]), status_rules)
    wanted = None if full else list(sections or settings.default_sections)
    all_sections = card_json.get(SECTIONS_FIELD) or {}
    included, omitted = _select_sections(all_sections, wanted)
    card = {key: value for key, value in card_json.items() if key != SECTIONS_FIELD}
    card[SECTIONS_FIELD] = included
    if not full:
        card.pop(PERMISSIONS_FIELD, None)
    summary = CardSummary(
        label=meta.label,
        doc_kind=meta.doc_kind,
        doc_number=meta.doc_number,
        doc_date=meta.doc_date.isoformat() if meta.doc_date else None,
        subject=meta.subject,
        doc_status=meta.doc_status,
        doc_status_name=meta.doc_status_name,
        state_name=meta.state_name,
        approval_state_name=meta.approval_state_name,
        department=meta.department,
        author=meta.author,
        signed_by=meta.signed_by,
        validity_period=meta.validity_period,
        comment=meta.comment,
        approvers=meta.approvers,
        responsible=meta.responsible,
        files=[file.name for file in record.files if not file.is_virtual],
    )
    return DocumentCard(
        doc_id=str(record.id),
        summary=summary,
        card=card,
        included_sections=list(included),
        omitted_sections=omitted,
        full=full,
    )
