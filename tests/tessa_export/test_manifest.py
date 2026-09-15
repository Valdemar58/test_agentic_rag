"""Манифест и граф связей (§8.1.5): пути входа, виды документов, статусы, дубли, статистика."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

from tessa_export.config import CANCELLED_STATUS_ID, CoverageSettings, ExportConfig, StatusSettings
from tessa_export.fake import (
    ACTIVE_STATUS_ID,
    FakeGateway,
    build_demo_scenario,
    make_file,
    make_snapshot,
    stable_uuid,
)
from tessa_export.files import FileRecord, download_card_files
from tessa_export.manifest import (
    LinksGraph,
    Manifest,
    build_links_graph,
    build_manifest,
    classify_coverage_kinds,
    mark_duplicates,
    resolve_doc_status,
)
from tessa_export.storage import write_json
from tessa_export.walker import Walker, WalkResult

A, B, C, D, E, F, G = (stable_uuid("card", name) for name in "ABCDEFG")


def _config() -> ExportConfig:
    return ExportConfig.model_validate(
        {"tessa": {"base_url": "http://t"}, "external": {"tessa_sdk_path": "/a", "card_service_path": "/b"}}
    )


def _walk_and_download(
    gateway: FakeGateway, seed: list[UUID], root: Path, config: ExportConfig
) -> tuple[WalkResult, dict[UUID, list[FileRecord]]]:
    result = Walker(gateway, config.traversal, config.exclude_rules).walk(seed)
    allowed = set(config.files.allowed_extensions)
    records = {
        card_id: download_card_files(gateway, card_id, visited.snapshot.files, allowed, root)
        for card_id, visited in result.cards.items()
    }
    return result, records


def test_classify_coverage_kinds_by_title_and_subject() -> None:
    coverage = CoverageSettings()
    kinds = classify_coverage_kinds
    assert kinds("Приказ", None, "О назначении ответственных", coverage) == ["Приказы"]
    # категории 8.2 содержательные: приказ об утверждении инструкции считается в обеих
    instruction = kinds("Приказ", None, "Об утверждении и введении в действие Инструкции по ДОУ", coverage)
    assert instruction == ["Приказы", "Инструкции"]
    assert kinds("Приказ", None, "О внесении изменений в ПВТР (донорство)", coverage) == [
        "Приказы",
        "Положения / ЛНА",
    ]
    assert kinds("Приказ", None, "Об утверждении Регламента работы ПДО", coverage) == [
        "Приказы",
        "Положения / ЛНА",
    ]
    # «акт» внутри слова («актуализация») — не акт
    assert kinds("Приказ", None, "Об актуализации перечня", coverage) == ["Приказы"]
    assert kinds(None, "Инструкция", None, coverage) == ["Инструкции"]
    assert kinds("Договорной документ", None, None, coverage) == ["Договоры"]
    assert kinds("Акт приёма-передачи", None, None, coverage) == ["Акты"]
    assert kinds("Иной документ", None, "Акт на списание материалов", coverage) == ["Акты"]
    assert kinds("Служебная записка", None, "Об оплате", coverage) == ["Служебные записки"]
    assert kinds("Первичный документ", "PrimaryDocumentMKC", None, coverage) == ["Прочие виды"]
    assert kinds(None, None, None, coverage) == ["Прочие виды"]


def test_doc_status_rule(tmp_path: Path) -> None:
    status = StatusSettings()
    assert resolve_doc_status(str(CANCELLED_STATUS_ID), 6, status) == "cancelled"
    assert resolve_doc_status(str(CANCELLED_STATUS_ID).upper(), 6, status) == "cancelled"
    assert resolve_doc_status(None, 5, status) == "cancelled"  # $KrStates_Doc_Canceled
    assert resolve_doc_status(None, 17, status) == "cancelled"  # Аннулирован
    assert resolve_doc_status(str(ACTIVE_STATUS_ID), 6, status) == "active"
    assert resolve_doc_status(None, 8, status) == "active"  # Signed, у типов без StatusID
    assert resolve_doc_status(None, 13, status) == "active"  # Списан в дело
    assert resolve_doc_status(None, 1, status) == "draft"  # $KrStates_Doc_Active = на согласовании
    assert resolve_doc_status(None, 0, status) == "draft"
    assert resolve_doc_status(None, None, status) == "draft"

    gateway = FakeGateway()
    signed, in_approval, annulled = uuid4(), uuid4(), uuid4()
    gateway.add(make_snapshot(signed, state_id=8, state_name="$KrStates_Doc_Signed"))
    gateway.add(make_snapshot(in_approval, state_id=1, state_name="$KrStates_Doc_Active"))
    gateway.add(make_snapshot(annulled, state_id=17, state_name="Аннулирован"))
    config = _config()
    result, records = _walk_and_download(gateway, [signed, in_approval, annulled], tmp_path, config)
    manifest = build_manifest(result, records, config)
    by_id = {document.card_id: document for document in manifest.documents}
    assert by_id[signed].doc_status == "active" and by_id[signed].state_id == 8
    assert by_id[in_approval].doc_status == "draft" and by_id[in_approval].is_cancelled is False
    assert by_id[annulled].doc_status == "cancelled" and by_id[annulled].is_cancelled is True
    assert manifest.stats.doc_statuses == {"active": 1, "draft": 1, "cancelled": 1}


def test_manifest_from_demo_scenario(tmp_path: Path) -> None:
    gateway, seed = build_demo_scenario()
    config = _config()
    result, records = _walk_and_download(gateway, seed, tmp_path, config)
    manifest = build_manifest(result, records, config, synthetic=True, now=datetime(2026, 9, 14, tzinfo=UTC))

    assert manifest.synthetic is True
    assert manifest.seed_ids == seed
    assert manifest.traversal["max_depth"] == 2
    by_id = {document.card_id: document for document in manifest.documents}

    order = by_id[A]
    assert order.doc_kind == "Приказ"  # категория Тессы как есть
    assert "Приказы" in order.coverage_kinds
    assert order.number == "144"
    assert order.is_cancelled is True and order.doc_status == "cancelled"
    assert order.status_name == "Отмененный"
    assert order.doc_date is not None and order.doc_date.year == 2026
    assert order.department
    assert order.entry_paths[0].kind == "seed"
    assert order.card_path == f"cards/{A}.json"
    file_names = {file.name for file in order.files}
    assert "Лист согласования.html" in file_names
    virtual = next(file for file in order.files if file.is_virtual)
    assert virtual.downloaded is False and virtual.skipped_reason == "virtual"
    pdf = next(file for file in order.files if file.extension == "pdf")
    assert pdf.downloaded and pdf.sha256 and pdf.path == f"files/{A}/{pdf.name}"

    memo = by_id[E]
    assert memo.doc_kind == "Служебная записка"
    assert "Служебные записки" in memo.coverage_kinds
    assert {path.via_card_id for path in memo.entry_paths} == {A, G}
    assert memo.files == []

    assert "Положения / ЛНА" in by_id[C].coverage_kinds
    assert "Акты" in by_id[G].coverage_kinds

    stats = manifest.stats
    assert stats.documents == len(result.cards)
    assert stats.files_total == sum(len(document.files) for document in manifest.documents)
    assert stats.files_downloaded + stats.files_skipped == stats.files_total
    assert stats.skipped_by_reason["virtual"] == 1
    assert stats.skipped_by_extension["sig"] == 1
    assert stats.doc_kinds["Приказ"] == 2
    assert stats.coverage_kinds["Приказы"] == 2
    assert stats.doc_statuses["cancelled"] >= 1
    assert stats.card_types["OrderMKC"] == len(manifest.documents)
    assert "Отмененный" in stats.status_values.values()
    assert stats.relation_types["в отмену"] == "отменено"
    assert stats.relation_types["приложение"] == "приложение к"
    assert stats.skipped_links == len(result.skipped_links)
    assert any(item.reason == "max_depth" for item in manifest.skipped_links)


def test_manifest_and_graph_round_trip(tmp_path: Path) -> None:
    gateway, seed = build_demo_scenario()
    config = _config()
    result, records = _walk_and_download(gateway, seed, tmp_path, config)
    manifest = build_manifest(result, records, config)
    graph = build_links_graph(result)

    write_json(tmp_path / "manifest.json", manifest.model_dump(mode="json"))
    write_json(tmp_path / "links_graph.json", graph.model_dump(mode="json"))
    restored = Manifest.model_validate_json((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    restored_graph = LinksGraph.model_validate_json(
        (tmp_path / "links_graph.json").read_text(encoding="utf-8")
    )
    assert restored == manifest
    assert restored_graph == graph

    edge = next(item for item in graph.edges if (item.from_id, item.to_id) == (A, B))
    assert edge.relation_type == "в отмену" and edge.reverse_type == "отменено"
    in_set = {document.card_id for document in manifest.documents}
    assert all(item.from_id in in_set and item.to_id in in_set for item in graph.edges)
    assert any(item.to_id == D for item in graph.dangling_edges)


def test_duplicate_files_marked_across_cards(tmp_path: Path) -> None:
    gateway = FakeGateway()
    first, second = uuid4(), uuid4()
    same = b"%PDF same content"
    gateway.add(make_snapshot(first, files=[make_file(first, "a.pdf")]), {"a.pdf": same})
    gateway.add(make_snapshot(second, files=[make_file(second, "b.pdf")]), {"b.pdf": same})
    config = _config()
    result, records = _walk_and_download(gateway, [first, second], tmp_path, config)
    manifest = build_manifest(result, records, config)
    assert manifest.stats.duplicate_files == 1
    docs = {document.card_id: document for document in manifest.documents}
    assert docs[first].files[0].duplicate_of is None
    assert docs[second].files[0].duplicate_of == f"{first}/{docs[first].files[0].row_id}"
    assert mark_duplicates(manifest.documents) == 1


def test_errors_and_excluded_in_manifest(tmp_path: Path) -> None:
    from tessa_export.config import ExcludeRule
    from tessa_export.models import CardAccessError

    gateway, seed = build_demo_scenario()
    gateway.card_errors[B] = CardAccessError("нет прав")
    config = _config()
    # G — seed-карточка, поэтому правило должно явно распространяться на seed
    config.exclude_rules.append(
        ExcludeRule(reason="без актов", doc_type_titles=["Акт"], applies_to_seed=True)
    )
    result, records = _walk_and_download(gateway, seed, tmp_path, config)
    manifest = build_manifest(result, records, config)
    assert manifest.exclude_rules == 1
    assert [item.card_id for item in manifest.excluded] == [G]
    assert manifest.excluded[0].reason == "без актов"
    assert manifest.errors[0].card_id == B and manifest.errors[0].kind == "access"
    assert manifest.stats.excluded == 1 and manifest.stats.errors == 1
