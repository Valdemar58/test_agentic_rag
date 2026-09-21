"""Перечень приказов из представления Тессы: разбор строк, фильтры, пагинация, предохранители."""

from __future__ import annotations

from typing import Any
from uuid import UUID

import pytest

from tessa_export.config import OrdersSettings, ViewParameter, ViewValue
from tessa_export.fake import FakeGateway, stable_uuid
from tessa_export.models import GatewayError, rows_by_column
from tessa_export.orders import OrdersError, collect_orders, state_rules

COLUMNS = ["DocID", "DocDescription", "DocTypeName", "StateID"]


def _row(name: str, *, state: int = 6, kind: str = "Приказ") -> dict[str, Any]:
    return {
        "DocID": str(stable_uuid("order", name)),
        "DocDescription": f"Приказ № {name}",
        "DocTypeName": kind,
        "StateID": state,
    }


def _gateway(rows: list[dict[str, Any]], *, paging: bool = True) -> FakeGateway:
    gateway = FakeGateway()
    gateway.add_view("Orders", COLUMNS, rows, caption="Приказы", paging=paging)
    return gateway


def _settings(**overrides: Any) -> OrdersSettings:
    return OrdersSettings.model_validate({"view_alias": "Orders", "page_limit": 2, **overrides})


def test_rows_by_column_maps_positional_rows() -> None:
    rows = rows_by_column(["A", "B"], [[1, 2], [3]])
    assert rows == [{"A": 1, "B": 2}, {"A": 3}]


def test_collect_orders_keeps_only_allowed_states_and_reports_counts() -> None:
    """По умолчанию выгружается только состояние 6 «Зарегистрировано» — действующие приказы."""
    rows = [_row("1", state=6), _row("2", state=1), _row("3", state=0), _row("4", state=6)]
    listing = collect_orders(_gateway(rows), _settings())

    assert listing.card_ids == [stable_uuid("order", "1"), stable_uuid("order", "4")]
    assert listing.excluded_by_state == 2
    assert listing.received == 4
    assert not listing.truncated
    assert "отсеяно: по состоянию 2" in "\n".join(listing.summary_lines())


def test_collect_orders_supports_black_list_of_states() -> None:
    rows = [_row("1", state=6), _row("2", state=5), _row("3", state=8)]
    listing = collect_orders(_gateway(rows), _settings(include_state_ids=[], exclude_state_ids=[5]))

    assert listing.card_ids == [stable_uuid("order", "1"), stable_uuid("order", "3")]
    assert listing.excluded_by_state == 1


def test_collect_orders_applies_client_side_match() -> None:
    rows = [_row("1"), _row("2", kind="Служебная записка"), _row("3")]
    listing = collect_orders(_gateway(rows), _settings(match={"DocTypeName": ["приказ"]}))

    assert listing.card_ids == [stable_uuid("order", "1"), stable_uuid("order", "3")]
    assert listing.excluded_by_match == 1


def test_collect_orders_deduplicates_view_without_paging() -> None:
    """Представление без пагинации отдаёт весь набор на каждой странице: повтор = конец списка."""
    rows = [_row(str(number)) for number in range(1, 4)]
    gateway = _gateway(rows, paging=False)
    listing = collect_orders(gateway, _settings(page_limit=3))

    # вторая страница вернула тот же набор: строки отброшены как повторы, обход остановлен
    assert len(listing.rows) == len(set(listing.card_ids)) == 3
    assert (listing.duplicates, listing.pages) == (3, 2)
    assert not listing.truncated


def test_collect_orders_reads_all_pages_and_respects_max_documents() -> None:
    """Смещение считается в строках: окно второй страницы начинается со строки page_limit + 1."""
    rows = [_row(str(number)) for number in range(1, 8)]
    gateway = _gateway(rows)
    listing = collect_orders(gateway, _settings(page_limit=2))
    assert len(listing.rows) == 7
    assert listing.pages == 4
    assert [offset for _, offset, _ in gateway.view_calls] == [1, 3, 5, 7]
    assert listing.duplicates == 0
    assert listing.reported_total == 7 and not listing.incomplete

    limited = collect_orders(_gateway(rows), _settings(page_limit=2, max_documents=3))
    assert len(limited.rows) == 3
    assert limited.truncated

    pages_capped = collect_orders(_gateway(rows), _settings(page_limit=2, max_pages=2))
    assert len(pages_capped.rows) == 4
    assert pages_capped.truncated


def test_collect_orders_warns_when_the_view_reports_more_rows() -> None:
    """Предохранитель обрезал чтение — в сводке видно, сколько строк осталось непрочитанными."""
    rows = [_row(str(number)) for number in range(1, 20)]
    listing = collect_orders(_gateway(rows), _settings(page_limit=2, max_pages=3))

    assert listing.reported_total == 19
    assert listing.unique == 6
    assert listing.incomplete
    assert "представление сообщает 19 строк, прочитано 6" in "\n".join(listing.summary_lines())


def test_collect_orders_passes_filters_and_paging_to_view() -> None:
    gateway = _gateway([_row("1")])
    settings = _settings(
        parameters=[ViewParameter(name="DocType", values=[ViewValue(value="order", text="Приказ")])],
        sort_column="DocDate",
    )
    collect_orders(gateway, settings)

    alias, page_offset, parameters = gateway.view_calls[0]
    assert (alias, page_offset) == ("Orders", 1)
    assert parameters[0].name == "DocType" and parameters[0].operand == "Equality"


def test_collect_orders_reports_unknown_columns_and_missing_alias() -> None:
    with pytest.raises(OrdersError, match="нет колонок Kind, State"):
        collect_orders(_gateway([_row("1")]), _settings(state_column="State", match={"Kind": ["приказ"]}))
    with pytest.raises(OrdersError, match="orders.view_alias"):
        collect_orders(_gateway([]), OrdersSettings())
    with pytest.raises(GatewayError, match="не найдено"):
        collect_orders(_gateway([_row("1")]), _settings(view_alias="Missing"))


def test_collect_orders_skips_rows_without_document_id() -> None:
    rows = [_row("1"), {"DocID": "не-uuid", "StateID": 8}, {"StateID": 8}]
    listing = collect_orders(_gateway(rows), _settings())
    assert listing.card_ids == [stable_uuid("order", "1")]
    assert listing.without_id == 2


def test_state_rules_keep_only_allowed_states_on_the_card() -> None:
    (rule,) = state_rules(OrdersSettings())
    assert rule.field == "DocumentCommonInfo.StateID"
    assert (rule.values, rule.values_mode) == (["6"], "none_of")
    assert rule.applies_to_seed
    assert "не из списка выгрузки: 6 Зарегистрировано" in rule.reason

    black_list = state_rules(OrdersSettings(include_state_ids=[], exclude_state_ids=[5, 17]))
    assert len(black_list) == 1
    assert (black_list[0].values, black_list[0].values_mode) == (["5", "17"], "any_of")
    assert state_rules(OrdersSettings(include_state_ids=[])) == []


def test_fake_gateway_lists_views() -> None:
    gateway = _gateway([_row("1")])
    assert [view.alias for view in gateway.list_views()] == ["Orders"]
    page = gateway.view_page("Orders", page_offset=1, page_limit=10)
    assert page.columns == COLUMNS
    assert UUID(page.rows[0]["DocID"]) == stable_uuid("order", "1")
