"""Реестр свидетельств: псевдонимы, замена в аргументах, вытеснение по LRU."""

from __future__ import annotations

from agent.evidence import EvidenceRegistry


def test_aliases_are_stable_and_resolve_to_ids() -> None:
    registry = EvidenceRegistry(max_documents=10)
    order = registry.register_document("doc-1", label="Приказ №144", doc_status="active")
    again = registry.register_document("doc-1", label="Приказ №144 от 15.01.2026", subject="Об охране труда")
    assert order is again and order.alias == "D1" and order.label == "Приказ №144 от 15.01.2026"
    assert order.subject == "Об охране труда" and order.status == "действует"
    weak = registry.register_document("doc-1", label="описание из связи", authoritative=False)
    assert weak.label == "Приказ №144 от 15.01.2026", "слабый источник не перекрывает подпись"
    fragment = registry.register_fragment("chunk-1", doc_id="doc-1", kind="hit", breadcrumbs="п. 1", text="т")
    assert fragment.alias == "S1" and fragment.doc_alias == "D1"
    assert registry.resolve("D1") == "doc-1" and registry.resolve(" S1 ") == "chunk-1"
    assert registry.resolve("D9") == "D9" and registry.resolve("не псевдоним") == "не псевдоним"
    resolved = registry.resolve_arguments(
        {"doc_id": "D1", "section_id": "S1", "filters": {"doc_ids": ["D1", "x"], "statuses": ["active"]}}
    )
    assert resolved == {
        "doc_id": "doc-1",
        "section_id": "chunk-1",
        "filters": {"doc_ids": ["doc-1", "x"], "statuses": ["active"]},
    }


def test_trim_evicts_least_recently_used_documents_with_fragments() -> None:
    registry = EvidenceRegistry(max_documents=2)
    for index in range(1, 4):
        registry.register_document(f"doc-{index}", label=f"Документ {index}")
        registry.register_fragment(
            f"chunk-{index}", doc_id=f"doc-{index}", kind="hit", breadcrumbs="", text="т"
        )
    registry.touch("doc-1")  # D1 снова использован — вытесняется D2
    assert registry.trim() == ["D2"]
    assert [item.alias for item in registry.documents()] == ["D1", "D3"]
    assert registry.fragment_by_alias("S2") is None and registry.fragment_by_alias("S1") is not None
    fresh = registry.register_document("doc-4", label="Документ 4")
    assert fresh.alias == "D4", "псевдонимы не переиспользуются после вытеснения"
