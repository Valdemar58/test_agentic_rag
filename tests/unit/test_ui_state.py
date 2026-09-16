"""Снимок и восстановление свидетельств (7.1): псевдонимы сохраняются, счётчики продолжаются."""

from __future__ import annotations

from agent.evidence import EvidenceRegistry, EvidenceSnapshot
from common.config import DEFAULT_CONFIG_PATH, load_app_config
from ui.state import AGENT_METADATA_KEY, restore_session

CONFIG = load_app_config(DEFAULT_CONFIG_PATH)


def _registry() -> EvidenceRegistry:
    registry = EvidenceRegistry(max_documents=10)
    registry.register_document("doc-1", label="Приказ №1", doc_status="active")
    registry.register_document("doc-2", label="Приказ №2", doc_status="cancelled")
    registry.register_fragment(
        "chunk-1", doc_id="doc-1", kind="hit", breadcrumbs="Приказ №1 → п. 1", text="первый"
    )
    registry.register_fragment(
        "chunk-2", doc_id="doc-2", kind="hit", breadcrumbs="Приказ №2 → п. 2", text="второй"
    )
    registry.register_fragment(
        "chunk-3", doc_id="doc-1", kind="section", breadcrumbs="Приказ №1 → 2", text="раздел"
    )
    return registry


def test_snapshot_takes_requested_fragments_and_their_documents() -> None:
    snapshot = _registry().snapshot(["D2"], ["S1"])
    assert [item.alias for item in snapshot.documents] == ["D2", "D1"]
    assert [item.alias for item in snapshot.fragments] == ["S1"]
    assert snapshot.model_validate(snapshot.model_dump(mode="json")) == snapshot


def test_restore_keeps_aliases_and_continues_numbering() -> None:
    snapshot = _registry().snapshot(["D1", "D2"], ["S2", "S3"])
    restored = EvidenceRegistry(max_documents=10)
    restored.restore(snapshot)
    assert [item.alias for item in restored.documents()] == ["D1", "D2"]
    assert [item.alias for item in restored.fragments()] == ["S2", "S3"]
    assert restored.resolve("S3") == "chunk-3" and restored.resolve("D2") == "doc-2"
    # новые записи получают следующие номера, а не повторяют восстановленные
    assert restored.register_document("doc-3", label="Приказ №3").alias == "D3"
    assert (
        restored.register_fragment("chunk-9", doc_id="doc-3", kind="hit", breadcrumbs="x", text="y").alias
        == "S4"
    )
    # повторное восстановление того же снимка ничего не дублирует, известный документ обновляется
    snapshot.documents[0].card_text = "сводка карточки"
    restored.restore(snapshot)
    document = restored.document_by_alias("D1")
    assert document is not None and document.card_text == "сводка карточки"
    assert len(restored.fragments()) == 3


def test_restore_skips_alias_conflicts_and_orphan_fragments() -> None:
    registry = EvidenceRegistry(max_documents=10)
    registry.register_document("other", label="Другой документ")  # занимает D1
    snapshot = _registry().snapshot(["D1"], ["S1"])
    registry.restore(snapshot)
    assert registry.document("doc-1") is None and registry.fragment("chunk-1") is None
    assert registry.document_by_alias("D1") is not None
    orphan = EvidenceSnapshot(fragments=list(snapshot.fragments))
    registry.restore(orphan)
    assert registry.fragment("chunk-1") is None


def test_restore_session_skips_broken_and_foreign_steps() -> None:
    snapshot = _registry().snapshot(["D1"], ["S1"]).model_dump(mode="json")
    steps = [
        {"type": "user_message", "output": "вопрос", "metadata": {}},
        {"type": "assistant_message", "output": "ответ без записи", "metadata": {}},
        {
            "type": "assistant_message",
            "output": "битая запись",
            "metadata": {AGENT_METADATA_KEY: {"evidence": 1}},
        },
        {
            "type": "assistant_message",
            "output": "ответ [1]",
            "metadata": {
                AGENT_METADATA_KEY: {"question": "вопрос", "evidence": snapshot, "document_aliases": ["D1"]}
            },
        },
    ]
    session = restore_session(CONFIG, "t1", steps)
    assert session.id == "t1" and len(session.memory.turns) == 1
    assert session.memory.turns[0].answer == "ответ [1]" and session.registry.resolve("S1") == "chunk-1"
