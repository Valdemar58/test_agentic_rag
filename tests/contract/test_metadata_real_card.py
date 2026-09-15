"""Маппинг метаданных на обезличенной реальной карточке заказчика (приказ №144, `OrderMKC`)."""

from __future__ import annotations

import datetime as dt
from typing import Any

import pytest

from common.config import DEFAULT_CONFIG_PATH, load_app_config
from contracts.card_service import CardServiceContract, card_data_from_tessa_response
from ingest.cards import CardRecord
from ingest.corpus import CorpusDocument
from ingest.metadata import document_metadata
from tessa_export.manifest import DocumentEntry, LinksGraph

pytestmark = pytest.mark.contract


def test_real_card_maps_to_metadata(contract: CardServiceContract, card_example_raw: dict[str, Any]) -> None:
    card_data = card_data_from_tessa_response(card_example_raw, contract)
    card = CardRecord.model_validate(card_data.model_dump(mode="json"))
    entry = DocumentEntry(
        card_id=card.id,
        type_name=card.type_name,
        type_caption=card.type_caption,
        doc_type_title=card.common_text("DocTypeTitle"),
        doc_kind=card.common_text("DocTypeTitle") or "",
        coverage_kinds=[],
        number=card.common_text("FullNumber"),
        doc_date=None,
        subject=card.common_text("Subject"),
        department=None,
        status_id=None,
        status_name=None,
        state_id=None,
        doc_status="active",
        is_cancelled=False,
        state_name=None,
        approval_state=None,
        depth=0,
        entry_paths=[],
        card_path=f"cards/{card.id}.json",
        card_raw_path=f"cards_raw/{card.id}.json",
        files=[],
    )
    document = CorpusDocument(card_id=card.id, entry=entry, card=card, files=[])
    meta = document_metadata(
        document, LinksGraph(edges=[], dangling_edges=[]), load_app_config(DEFAULT_CONFIG_PATH).ingest.status
    )

    assert meta.card_type_name == "OrderMKC" and meta.doc_kind == "Приказ"
    assert meta.doc_number == "144" and isinstance(meta.doc_date, dt.date)
    # приказ отменён приказом №173: StatusID «Отмененный» при StateName маршрута Registered
    assert meta.doc_status == "cancelled" and meta.doc_status_name == "Отмененный"
    assert meta.state_name == "$KrStates_Doc_Registered" and meta.comment and "173" in meta.comment
    assert meta.department and meta.author and meta.signed_by
    assert all(name.startswith("С.С. Сотрудник") for name in meta.approvers + meta.responsible)
    assert meta.approvers and meta.responsible and meta.direction_activity == ["Безопасность"]
    directions = {relation.direction for relation in meta.relations}
    assert directions == {"outgoing", "incoming"}
    outgoing = next(relation for relation in meta.relations if relation.direction == "outgoing")
    assert outgoing.relation == "в отмену"
    assert meta.acl_groups == [] and meta.label.startswith("Приказ №144 от ")
