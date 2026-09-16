"""`get_document_card` (5.4): секции по умолчанию, полная карточка, сводка по маппингу метаданных."""

from __future__ import annotations

from datetime import UTC, datetime

from common.config import DEFAULT_CONFIG_PATH, load_app_config
from mcp_server.card_view import build_document_card
from tessa_export.config import CANCELLED_STATUS_ID
from tessa_export.fake import link, make_file, make_snapshot, stable_uuid

CONFIG = load_app_config(DEFAULT_CONFIG_PATH)
ORDER, OTHER = stable_uuid("view", "order"), stable_uuid("view", "other")


def _card() -> dict[str, object]:
    snapshot = make_snapshot(
        ORDER,
        number="144",
        doc_date=datetime(2026, 1, 15, tzinfo=UTC),
        status_id=CANCELLED_STATUS_ID,
        status_name="Отмененный",
        outgoing=[link(OTHER, ref_type_name="в отмену", ref_type_reverse_name="отменено")],
        files=[
            make_file(ORDER, "ДокШаблон Приказ №144.docx"),
            make_file(ORDER, "Лист согласования.html", category=None, is_virtual=True, size=-1),
        ],
        extra_fields={"Comment": "Отменен приказом № 173.", "ValidityPeriod": None},
    )
    card = dict(snapshot.card_data_json)
    sections = dict(card["sections"])
    sections["KrStagesVirtual"] = {"name": None, "type": 1, "rows": [{"Name": "этап"}] * 50, "fields": None}
    card["sections"] = sections
    return card


def test_default_view_keeps_selected_sections_and_summarizes() -> None:
    view = build_document_card(
        _card(), sections=None, full=False, settings=CONFIG.card_service, status_rules=CONFIG.ingest.status
    )
    assert view.doc_id == str(ORDER) and not view.full
    assert view.included_sections == ["DocumentCommonInfo", "OutgoingRefDocs", "IncomingRefDocs"]
    assert view.omitted_sections == ["KrStagesVirtual"] and "permissions" not in view.card
    assert view.card["type_name"] == "OrderMKC" and view.card["files"] and "sections" in view.card
    # поля секций не переименованы и не потеряны
    common = view.card["sections"]["DocumentCommonInfo"]["fields"]
    assert common["FullNumber"] == "144" and common["StatusNameStatus"] == "Отмененный"
    summary = view.summary
    assert summary.label == "Приказ №144 от 15.01.2026" and summary.doc_status == "cancelled"
    assert summary.doc_status_name == "Отмененный" and summary.comment == "Отменен приказом № 173."
    assert summary.department == "Отдел охраны труда" and summary.author == "С.С. Сотрудник1"
    assert summary.files == ["ДокШаблон Приказ №144.docx"]


def test_full_view_and_explicit_sections() -> None:
    full = build_document_card(
        _card(), sections=None, full=True, settings=CONFIG.card_service, status_rules=CONFIG.ingest.status
    )
    assert full.full and full.omitted_sections == [] and "KrStagesVirtual" in full.card["sections"]
    assert "permissions" in full.card
    chosen = build_document_card(
        _card(),
        sections=["outgoingrefdocs", "нет такой"],
        full=False,
        settings=CONFIG.card_service,
        status_rules=CONFIG.ingest.status,
    )
    assert chosen.included_sections == ["OutgoingRefDocs"]
    assert set(chosen.omitted_sections) == {"DocumentCommonInfo", "IncomingRefDocs", "KrStagesVirtual"}
