"""Метаданные чанка из карточки (4.5): каждое поле маппинга, статус, связи, крошки, payload."""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from docling_core.types.doc.document import DoclingDocument
from docling_core.types.doc.labels import DocItemLabel

from common.config import DEFAULT_CONFIG_PATH, load_app_config
from ingest.cards import CardRecord
from ingest.chunking import StructuralChunker
from ingest.corpus import CorpusDocument, CorpusFile
from ingest.files import FilePlan
from ingest.metadata import (
    ChunkPayload,
    child_payload,
    document_metadata,
    file_metadata,
    parent_payload,
    root_crumbs,
)
from ingest.parents import build_chunk_set
from ingest.tokens import WordTokenCounter
from tessa_export.manifest import DocumentEntry, LinkEdgeEntry, LinksGraph

CONFIG = load_app_config(DEFAULT_CONFIG_PATH)
STATUS = CONFIG.ingest.status
CHUNKING = CONFIG.ingest.chunking
CARD = UUID("aef446c9-91e5-5862-9ea7-a705268e052e")
MEMO = UUID("23654ff0-038c-5012-b0c1-2c7178761477")
ORDER_173 = UUID("a1d65c04-909b-5144-85d3-04ee27cf92e7")
ORDER_151 = UUID("2ebfd7fc-bcb4-511e-84ec-1c8b5826ac0d")
ACTIVE_STATUS = "1229fb49-3d80-47c9-9816-3c5467954679"
CANCELLED_STATUS = "de9d3b6d-532b-4cb8-aa7b-e055e8986e48"


def _card(**overrides: Any) -> CardRecord:
    common: dict[str, Any] = {
        "FullNumber": "144",
        "DocDate": "2026-01-15T00:00:00+00:00",
        "CreationDate": "2026-01-10T00:00:00+00:00",
        "Subject": "Об утверждении Положения об охране труда",
        "DocTypeTitle": "Приказ",
        "DepartmentID": "3927deac-0457-5b44-afe0-6fd3318625d6",
        "DepartmentName": "Отдел охраны труда",
        "AuthorName": "С.С. Сотрудник1",
        "RegistratorName": "С.С. Сотрудник2",
        "SignedByName": "С.С. Сотрудник3",
        "StatusID": ACTIVE_STATUS,
        "StatusNameStatus": "Действующий",
        "StateID": 6,
        "StateName": "$KrStates_Doc_Registered",
        "Comment": None,
    }
    common.update(overrides.pop("common", {}))
    data: dict[str, Any] = {
        "id": str(CARD),
        "type_name": "OrderMKC",
        "type_caption": "Приказ",
        "version": 3,
        "modified": "2026-01-15T10:00:00+00:00",
        "created_by_name": "System",
        "sections": {
            "DocumentCommonInfo": {"fields": common},
            "OutgoingRefDocs": {
                "rows": [
                    {
                        "DocID": str(MEMO),
                        "RefTypeName": "Документ-основание",
                        "RefTypeReverseName": "Приказ",
                        "DocTypeName": "Служебная записка",
                    }
                ]
            },
            "IncomingRefDocs": {"rows": [{"DocID": str(ORDER_173)}, {"DocID": str(ORDER_151)}]},
            "Approv": {"rows": [{"UserName": "С.С. Сотрудник4"}, {"UserName": "С.С. Сотрудник5"}]},
            "ResponsibleErrand": {"rows": [{"UserName": "С.С. Сотрудник6"}]},
            "DirectionActivityDCI": {"rows": [{"DirectionActivityName": "Безопасность"}]},
            "KrApprovalCommonInfoVirtual": {"fields": {"StateID": 2, "StateName": "$KrStates_Doc_Approved"}},
        },
        "files": [],
    }
    data.update(overrides)
    return CardRecord.model_validate(data)


def _document(card: CardRecord) -> CorpusDocument:
    entry = DocumentEntry(
        card_id=card.id,
        type_name=card.type_name,
        type_caption=card.type_caption,
        doc_type_title="Приказ",
        doc_kind="Приказ",
        coverage_kinds=[],
        number="144",
        doc_date=dt.date(2026, 1, 15),
        subject="тема",
        department=None,
        status_id=None,
        status_name=None,
        state_id=6,
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
    return CorpusDocument(card_id=card.id, entry=entry, card=card, files=[])


GRAPH = LinksGraph(
    edges=[
        LinkEdgeEntry(
            from_id=ORDER_173,
            to_id=CARD,
            relation_type="в отмену",
            reverse_type="отменено",
            relation_type_id=None,
            source="outgoing",
        ),
        LinkEdgeEntry(
            from_id=CARD,
            to_id=MEMO,
            relation_type="Документ-основание",
            reverse_type="Приказ",
            relation_type_id=None,
            source="outgoing",
        ),
    ],
    dangling_edges=[],
)


def _plan(role: str = "main", name: str = "ДокШаблон Приказ №144.docx") -> FilePlan:
    file = CorpusFile(
        card_id=CARD,
        row_id=uuid4(),
        name=name,
        extension=name.rsplit(".", 1)[-1],
        category="Документ",
        relative_path=f"files/{CARD}/{name}",
        path=Path(name),
        sha256="e" * 64,
        size=1,
        has_text_layer=None,
        page_count=None,
        duplicate_of=None,
        smoke_note=None,
    )
    return FilePlan(file=file, role=role, also_in=(MEMO,))  # type: ignore[arg-type]


def test_every_mapping_field_comes_from_the_card() -> None:
    meta = document_metadata(_document(_card()), GRAPH, STATUS)
    assert meta.doc_id == meta.tessa_card_id == str(CARD)
    assert (meta.card_type_name, meta.card_type_caption, meta.doc_kind) == ("OrderMKC", "Приказ", "Приказ")
    assert meta.doc_number == "144" and meta.doc_date == dt.date(2026, 1, 15)
    assert meta.doc_date_ts == int(dt.datetime(2026, 1, 15, tzinfo=dt.UTC).timestamp())
    assert meta.doc_status == "active" and meta.doc_status_name == "Действующий"
    assert (meta.state_id, meta.state_name) == (6, "$KrStates_Doc_Registered")
    assert meta.approval_state == "$KrStates_Doc_Approved" and meta.approval_state_name == "Согласован"
    assert (
        meta.department == "Отдел охраны труда"
        and meta.department_id == "3927deac-0457-5b44-afe0-6fd3318625d6"
    )
    assert meta.author == "С.С. Сотрудник1" and meta.signed_by == "С.С. Сотрудник3"
    assert meta.acl_groups == []
    assert (meta.subject or "").startswith("Об утверждении") and meta.comment is None
    assert meta.direction_activity == ["Безопасность"]
    assert meta.approvers == ["С.С. Сотрудник4", "С.С. Сотрудник5"] and meta.responsible == [
        "С.С. Сотрудник6"
    ]
    assert meta.validity_period is None and meta.card_version == 3
    assert meta.card_modified == dt.datetime(2026, 1, 15, 10, tzinfo=dt.UTC)
    assert meta.label == "Приказ №144 от 15.01.2026"


def test_relations_outgoing_typed_and_incoming_typed_via_links_graph() -> None:
    meta = document_metadata(_document(_card()), GRAPH, STATUS)
    relations = {(relation.doc_id, relation.direction): relation for relation in meta.relations}
    outgoing = relations[(str(MEMO), "outgoing")]
    assert outgoing.relation == "Документ-основание" and outgoing.doc_type == "Служебная записка"
    # входящая связь от №173: тип — обратное имя из карточки-источника («отменено»)
    assert relations[(str(ORDER_173), "incoming")].relation == "отменено"
    # входящая без ребра в графе — тип неизвестен
    assert relations[(str(ORDER_151), "incoming")].relation is None
    assert len(meta.relations) == 3


def test_status_rule_and_fallbacks() -> None:
    cancelled = _card(common={"StatusID": CANCELLED_STATUS, "StatusNameStatus": "Отмененный"})
    assert document_metadata(_document(cancelled), GRAPH, STATUS).doc_status == "cancelled"
    by_state = _card(common={"StatusID": None, "StatusNameStatus": None, "StateID": 17})
    assert document_metadata(_document(by_state), GRAPH, STATUS).doc_status == "cancelled"
    draft = _card(common={"StatusID": None, "StatusNameStatus": None, "StateID": 1})
    meta = document_metadata(_document(draft), GRAPH, STATUS)
    assert meta.doc_status == "draft" and meta.doc_status_name is None
    # номер проекта, дата создания, регистратор, статус согласования из состояния маршрута
    minimal = _card(
        common={
            "FullNumber": None,
            "SecondaryFullNumber": "П-7",
            "DocDate": None,
            "AuthorName": None,
            "DepartmentName": None,
            "DepartmentID": None,
            "StateID": 1,
            "StateName": "$KrStates_Doc_Active",
        },
        sections={
            "DocumentCommonInfo": {"fields": {}},
        },
    )
    # секции перезаписаны целиком: остаётся только DocumentCommonInfo без полей
    meta = document_metadata(_document(minimal), GRAPH, STATUS)
    assert meta.doc_number is None and meta.doc_date is None and meta.doc_date_ts is None
    assert meta.author == "System" and meta.relations == [] and meta.approvers == []
    assert meta.doc_status == "draft" and meta.approval_state is None and meta.label == "Приказ"

    fallback = _card(
        common={"FullNumber": None, "SecondaryFullNumber": "П-7", "DocDate": None, "AuthorName": None}
    )
    fallback_meta = document_metadata(_document(fallback), GRAPH, STATUS)
    assert fallback_meta.doc_number == "П-7" and fallback_meta.doc_date == dt.date(2026, 1, 10)
    assert fallback_meta.author == "С.С. Сотрудник2"
    unknown_state = _card(
        sections={
            **_card().model_dump()["sections"],
            "KrApprovalCommonInfoVirtual": {"fields": {"StateID": 16, "StateName": "$KrStates_Doc_X"}},
        }
    )
    assert document_metadata(_document(unknown_state), GRAPH, STATUS).approval_state_name == "$KrStates_Doc_X"


def test_root_crumbs_and_payloads_for_child_and_parent() -> None:
    meta = document_metadata(_document(_card()), GRAPH, STATUS)
    main_plan, appendix_plan = _plan(), _plan("appendix", "Приложение - Положение.docx")
    assert root_crumbs(meta, main_plan) == ("Приказ №144 от 15.01.2026",)
    assert root_crumbs(meta, appendix_plan) == (
        "Приказ №144 от 15.01.2026",
        "Приложение «Приложение - Положение.docx»",
    )

    doc = DoclingDocument(name="приказ")
    doc.add_heading("Об утверждении", level=1)
    doc.add_heading("3. Контроль", level=2)
    doc.add_text(
        label=DocItemLabel.TEXT, text="3.2. Отдел кадров направляет копию приказа во все подразделения."
    )
    chunker = StructuralChunker(CHUNKING, WordTokenCounter())
    chunks = chunker.chunk(doc, root_crumbs(meta, appendix_plan), file_sha256="e" * 64)
    chunk_set = build_chunk_set(chunks, CHUNKING, WordTokenCounter(), file_sha256="e" * 64)
    file_meta = file_metadata(appendix_plan, "native")
    assert (
        file_meta.file_role == "appendix"
        and file_meta.also_in == [str(MEMO)]
        and file_meta.parse_route == "native"
    )

    child = child_payload(meta, file_meta, chunk_set.children[0], CHUNKING.breadcrumb_separator)
    assert isinstance(child, ChunkPayload)
    assert child.doc_id == str(CARD) and child.doc_status == "active" and child.acl_groups == []
    assert child.doc_date == "2026-01-15" and child.card_modified == "2026-01-15T10:00:00+00:00"
    assert (
        child.section_path == ["Раздел 3. Контроль"]
        and child.clause == "3.2"
        and child.chunk_level == "child"
    )
    expected_crumbs = (
        "Приказ №144 от 15.01.2026",
        "Приложение «Приложение - Положение.docx»",
        "Раздел 3. Контроль",
        "п. 3.2",
    )
    assert child.breadcrumbs == CHUNKING.breadcrumb_separator.join(expected_crumbs)
    assert child.parent_id == chunk_set.parents[0].chunk_id and child.chunk_kind == "structural"
    assert child.file_sha256 == "e" * 64 and child.file_name == "Приложение - Положение.docx"
    assert child.relations[0].doc_id == str(MEMO)

    parent = parent_payload(meta, file_meta, chunk_set.parents[0], CHUNKING.breadcrumb_separator)
    assert parent.chunk_level == "parent" and parent.child_ids == [child.chunk_id] and parent.clause is None
    assert parent.section_path == ["Раздел 3. Контроль"] and parent.parent_id is None
    assert set(parent.model_dump()) == set(child.model_dump())
