"""Гибридный поиск (5.3): RRF + rerank, фильтры, AC-2.2, parent-контекст — на встроенном Qdrant."""

from __future__ import annotations

import datetime as dt

import pytest

from common.config import DEFAULT_CONFIG_PATH, load_app_config
from ingest.metadata import ChunkPayload
from mcp_server.retrieval import SearchFilters, match_known_values
from tests.unit.index_data import InMemoryCorpus, standard_corpus

CONFIG = load_app_config(DEFAULT_CONFIG_PATH)
Corpus = InMemoryCorpus


@pytest.fixture
def corpus() -> Corpus:
    return standard_corpus()


def test_default_search_returns_only_active_documents_with_parent_context(corpus: Corpus) -> None:
    result = corpus.searcher().search("отчёт по охране труда до пятого числа")
    assert result.hits and result.applied_filters == {"statuses": ["active"]}
    assert {hit.doc_status for hit in result.hits} == {"active"}
    # AC-2.2: отменённый приказ 109 и проект не попадают без явного фильтра
    assert corpus.docs["order_109_cancelled"] not in {hit.doc_id for hit in result.hits}
    assert corpus.docs["draft_160"] not in {hit.doc_id for hit in result.hits}
    top = result.hits[0]
    assert top.doc_id == corpus.docs["order_144"] and "пятого" in top.text
    assert top.clause == "1.1" and top.breadcrumbs.startswith("Приказ №144")
    assert top.context and "Контроль оставляю" in top.context and top.parent_id
    assert top.fusion_rank >= 1 and 0 <= top.score <= 1
    assert result.candidates >= len(result.hits) and result.notes == []


def test_explicit_status_filter_returns_cancelled_and_drafts(corpus: Corpus) -> None:
    cancelled = corpus.searcher().search("отчёт до десятого числа", SearchFilters(statuses=["cancelled"]))
    assert {hit.doc_id for hit in cancelled.hits} == {corpus.docs["order_109_cancelled"]}
    everything = corpus.searcher().search(
        "отчёт", SearchFilters(statuses=["active", "cancelled", "draft"]), top_k=10
    )
    assert {hit.doc_status for hit in everything.hits} == {"active", "cancelled", "draft"}


def test_doc_kind_and_department_filters_are_matched_loosely(corpus: Corpus) -> None:
    searcher = corpus.searcher()
    assert searcher.known_values("doc_kind") == ["Договорной документ", "Приказ"]
    contracts = searcher.search("отчёт о поставке", SearchFilters(doc_kinds=["договор"]))
    assert {hit.doc_kind for hit in contracts.hits} == {"Договорной документ"}
    assert contracts.applied_filters["doc_kinds"] == ["Договорной документ"]
    orders = searcher.search("отчёт", SearchFilters(doc_kinds=["ПРИКАЗ"], departments=["охраны труда"]))
    assert orders.hits and {hit.doc_kind for hit in orders.hits} == {"Приказ"}
    assert orders.applied_filters["departments"] == ["Отдел охраны труда"]

    unknown = searcher.search("отчёт", SearchFilters(doc_kinds=["Инструкция"]))
    assert unknown.hits == [] and unknown.notes and "Инструкция" in unknown.notes[0]
    assert "«Приказ»" in unknown.notes[0]


def test_date_range_and_doc_id_filters(corpus: Corpus) -> None:
    searcher = corpus.searcher()
    in_2026 = searcher.search(
        "отчёт", SearchFilters(date_from=dt.date(2026, 1, 1), date_to=dt.date(2026, 12, 31))
    )
    assert {hit.doc_id for hit in in_2026.hits} == {corpus.docs["order_144"]}
    assert in_2026.applied_filters["date_from"] == "2026-01-01"
    on_the_day = searcher.search(
        "отчёт", SearchFilters(date_from=dt.date(2026, 1, 15), date_to=dt.date(2026, 1, 15))
    )
    assert {hit.doc_id for hit in on_the_day.hits} == {corpus.docs["order_144"]}
    before = searcher.search("отчёт", SearchFilters(date_to=dt.date(2024, 1, 1)))
    assert {hit.doc_id for hit in before.hits} == {corpus.docs["contract_d1"]}
    only = searcher.search("отчёт", SearchFilters(doc_ids=[corpus.docs["contract_d1"]]))
    assert {hit.doc_id for hit in only.hits} == {corpus.docs["contract_d1"]}


def test_top_k_is_capped_and_reranker_orders_hits(corpus: Corpus) -> None:
    searcher = corpus.searcher(top_k=1, max_top_k=2)
    assert len(searcher.search("отчёт").hits) == 1
    assert len(searcher.search("отчёт", top_k=50).hits) == 2
    # reranker (доля слов запроса) ставит пункт про контроль выше, хотя по словам «отчёт» он проигрывает
    result = corpus.searcher().search("контроль оставляю за собой")
    assert "Контроль оставляю" in result.hits[0].text
    assert result.hits[0].score >= result.hits[-1].score

    without_parent = corpus.searcher(return_parent=False).search("контроль")
    assert without_parent.hits and all(hit.context is None for hit in without_parent.hits)


def test_match_known_values_rules() -> None:
    known = ["Приказ", "Договорной документ", "Служебная записка"]
    assert match_known_values(["приказ"], known) == (["Приказ"], [])
    assert match_known_values(["договор"], known) == (["Договорной документ"], [])
    assert match_known_values(["Служебная записка о чём-то"], known) == (["Служебная записка"], [])
    assert match_known_values(["Акт"], known) == (["Акт"], ["Акт"])
    assert match_known_values(["  "], known) == ([], [])


def test_hit_payload_is_the_ingest_payload(corpus: Corpus) -> None:
    result = corpus.searcher().search("контроль")
    hit = result.hits[0]
    records = corpus.index.client.retrieve(CONFIG.qdrant.collection, ids=[hit.chunk_id], with_payload=True)
    payload = ChunkPayload.model_validate(records[0].payload)
    assert (hit.text, hit.breadcrumbs, hit.file_name) == (
        payload.body,
        payload.breadcrumbs,
        payload.file_name,
    )
