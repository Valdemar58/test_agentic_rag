"""Обход связей: циклы, глубина, лимит документов, направления, исключения, ошибки (AC-EXP.1)."""

from __future__ import annotations

from uuid import UUID

import pytest

from tessa_export.config import ExcludeRule, TraversalSettings
from tessa_export.fake import FakeGateway, build_demo_scenario, link, make_snapshot, stable_uuid
from tessa_export.models import GatewayConnectionError
from tessa_export.walker import Walker, WalkResult, match_exclusion

A, B, C, D, E, F, G = (stable_uuid("card", name) for name in "ABCDEFG")


def _walk(
    gateway: FakeGateway,
    seed: list[UUID],
    *,
    max_depth: int = 2,
    max_docs: int = 200,
    directions: list[str] | None = None,
    rules: list[ExcludeRule] | None = None,
) -> WalkResult:
    traversal = TraversalSettings.model_validate(
        {"max_depth": max_depth, "max_docs": max_docs, "directions": directions or ["outgoing", "incoming"]}
    )
    return Walker(gateway, traversal, rules).walk(seed)


def test_cycle_is_visited_once() -> None:
    gateway, _ = build_demo_scenario()
    result = _walk(gateway, [A], max_depth=5)
    assert gateway.get_calls.count(A) == 1
    assert gateway.get_calls.count(B) == 1
    assert len(gateway.get_calls) == len(set(gateway.get_calls))
    edge_keys = {(edge.from_card_id, edge.to_card_id) for edge in result.edges}
    assert (A, B) in edge_keys and (B, A) in edge_keys
    assert len(edge_keys) == len(result.edges)


def test_depth_limit_and_entry_paths() -> None:
    gateway, seed = build_demo_scenario()
    result = _walk(gateway, seed, max_depth=2)
    depths = {card_id: visited.depth for card_id, visited in result.cards.items()}
    assert depths[A] == 0 and depths[G] == 0
    assert depths[B] == 1 and depths[E] == 1
    assert depths[C] == 2
    assert D not in result.cards
    skipped = [item for item in result.skipped_links if item.reason == "max_depth"]
    assert any(item.from_card_id == C and item.to_card_id == D for item in skipped)
    # E достижим из A и из G: оба пути входа в манифесте
    paths = result.cards[E].entry_paths
    assert {path.via_card_id for path in paths} == {A, G}
    assert all(path.kind == "link" for path in paths)
    assert result.cards[A].entry_paths[0].kind == "seed"
    # рёбра только внутри сета; ребро C→D висячее
    assert all(edge.from_card_id in result.cards and edge.to_card_id in result.cards for edge in result.edges)
    assert any(edge.from_card_id == C and edge.to_card_id == D for edge in result.dangling_edges)


def test_max_docs_limit_stops_cleanly() -> None:
    gateway, _ = build_demo_scenario()
    result = _walk(gateway, [A], max_docs=2, max_depth=5)
    assert result.attempted == 2
    reasons = {item.reason for item in result.skipped_links}
    assert "max_docs" in reasons
    assert all(item.detail for item in result.skipped_links)


def test_seed_over_limit_is_reported() -> None:
    gateway, _ = build_demo_scenario()
    result = _walk(gateway, [A, G, F], max_docs=2, max_depth=0)
    assert set(result.cards) == {A, G}
    over = [item for item in result.skipped_links if item.from_card_id is None]
    assert over and over[0].to_card_id == F and over[0].reason == "max_docs"


def test_directions_control_traversal() -> None:
    gateway, _ = build_demo_scenario()
    both = _walk(gateway, [A], max_depth=2)
    assert G in both.cards  # A → E (исходящая) → G (входящая у E)
    gateway_out, _ = build_demo_scenario()
    outgoing_only = _walk(gateway_out, [A], max_depth=2, directions=["outgoing"])
    assert G not in outgoing_only.cards
    assert E in outgoing_only.cards


def test_typed_edge_wins_over_untyped_incoming() -> None:
    gateway, _ = build_demo_scenario()
    result = _walk(gateway, [A], max_depth=1)
    edge = next(item for item in result.edges if (item.from_card_id, item.to_card_id) == (B, A))
    assert edge.relation_type == "в отмену"
    assert edge.reverse_type == "отменено"
    assert edge.source == "outgoing"


def test_exclusion_rule_stops_links_and_keeps_only_identity() -> None:
    gateway, seed = build_demo_scenario()
    rules = [ExcludeRule(reason="без положений", doc_type_titles=["положение"])]
    result = _walk(gateway, seed, max_depth=3, rules=rules)
    assert C in result.excluded and C not in result.cards
    excluded = result.excluded[C]
    assert excluded.reason == "без положений"
    assert excluded.doc_type_title == "Положение"
    assert D not in result.cards  # ссылка из исключённого документа не обходится
    assert any(item.reason == "excluded" and item.from_card_id == C for item in result.skipped_links)
    assert any(edge.to_card_id == C for edge in result.dangling_edges)


def test_match_exclusion_criteria() -> None:
    snapshot = make_snapshot(A, type_name="OrderMKC", doc_type_title="Приказ")
    assert match_exclusion(snapshot, [ExcludeRule(card_ids=[A])]) == "правило 1"
    assert match_exclusion(snapshot, [ExcludeRule(reason="тип", card_type_names=["OrderMKC"])]) == "тип"
    status = snapshot.common_text("StatusID")
    assert status is not None
    assert match_exclusion(snapshot, [ExcludeRule(field="DocumentCommonInfo.StatusID", values=[status])])
    assert match_exclusion(snapshot, [ExcludeRule(doc_type_titles=["Договор"])]) is None


def test_errors_do_not_stop_walk() -> None:
    gateway, _ = build_demo_scenario()
    gateway.card_errors[B] = GatewayConnectionError("сеть недоступна")
    missing = stable_uuid("card", "missing")
    gateway.add(make_snapshot(A, outgoing=[link(B), link(missing)]))
    result = _walk(gateway, [A], max_depth=2)
    kinds = {error.card_id: error.kind for error in result.errors}
    assert kinds[B] == "connection"
    assert kinds[missing] == "not_found"
    assert A in result.cards
    assert result.errors[0].entry_paths


@pytest.mark.parametrize("max_depth", [0, 1])
def test_depth_zero_and_one(max_depth: int) -> None:
    gateway, _ = build_demo_scenario()
    result = _walk(gateway, [A], max_depth=max_depth)
    assert A in result.cards
    assert (B in result.cards) == (max_depth >= 1)
