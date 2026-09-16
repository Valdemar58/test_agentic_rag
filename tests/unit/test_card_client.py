"""Клиент сервиса карточек (5.2): карточка без потерь, связи из секций, тип входящей связи через источник."""

from __future__ import annotations

import base64
from collections.abc import AsyncIterator
from typing import Any
from uuid import UUID

import httpx
import pytest
import respx

from common.config import DEFAULT_CONFIG_PATH, load_app_config
from mcp_server.cards import (
    CardCache,
    CardNotFoundError,
    CardServiceAuthError,
    CardServiceClient,
    CardServiceError,
    relation_matches,
)
from tessa_export.fake import link, make_snapshot, stable_uuid

BASE_URL = "http://cards.test"
SETTINGS = load_app_config(DEFAULT_CONFIG_PATH).card_service
ORDER_144, ORDER_109, ORDER_173, MEMO_9, OUTSIDE = (
    stable_uuid("client", name) for name in ("order144", "order109", "order173", "memo9", "outside")
)


def _cards() -> dict[UUID, dict[str, Any]]:
    """Приказ 173 отменяет 109; 144 дополняет 109; служебная записка — основание 144; 109 ссылается вовне."""
    order_173 = make_snapshot(
        ORDER_173,
        number="173",
        outgoing=[link(ORDER_109, ref_type_name="В отмену", ref_type_reverse_name="Отменено")],
    )
    order_144 = make_snapshot(
        ORDER_144,
        number="144",
        outgoing=[link(ORDER_109, ref_type_name="дополнение", ref_type_reverse_name="дополнено")],
        incoming=[link(MEMO_9, ref_type_name=None, ref_type_reverse_name=None)],
    )
    order_109 = make_snapshot(
        ORDER_109,
        number="109",
        outgoing=[link(OUTSIDE, ref_type_name="ссылается на", ref_type_reverse_name="упоминается в")],
        incoming=[
            link(ORDER_173, ref_type_name=None, ref_type_reverse_name=None),
            link(ORDER_144, ref_type_name=None, ref_type_reverse_name=None),
            link(OUTSIDE, ref_type_name=None, ref_type_reverse_name=None),
        ],
    )
    memo_9 = make_snapshot(
        MEMO_9,
        type_name="InhouseDocumentMKC",
        type_caption="Служебная записка",
        number="СЗ-9",
        outgoing=[link(ORDER_144, ref_type_name="Приказ", ref_type_reverse_name="Документ-основание")],
    )
    return {item.card_id: item.card_data_json for item in (order_173, order_144, order_109, memo_9)}


CARDS = _cards()


def _serve(request: httpx.Request) -> httpx.Response:
    payload = httpx.Response(200, content=request.content).json()
    card_id = UUID(payload["card_id"])
    if card_id == OUTSIDE:
        return httpx.Response(
            404, json={"error": "resource_not_found", "message": f"card не найден: {card_id}"}
        )
    return httpx.Response(200, json=CARDS[card_id])


@pytest.fixture
async def client() -> AsyncIterator[CardServiceClient]:
    instance = CardServiceClient(BASE_URL, SETTINGS, username="robot", password="secret")
    try:
        yield instance
    finally:
        await instance.aclose()


@respx.mock
async def test_get_card_returns_service_json_without_loss(client: CardServiceClient) -> None:
    route = respx.post(f"{BASE_URL}/core/cards/get").mock(side_effect=_serve)
    card = await client.get_card(ORDER_173)
    assert card == CARDS[ORDER_173]
    request = route.calls.last.request
    assert request.headers["authorization"] == "Basic " + base64.b64encode(b"robot:secret").decode()
    body = httpx.Response(200, content=request.content).json()
    assert body == {"card_id": str(ORDER_173), "mode": 1}
    # второй запрос той же карточки — из кэша
    assert await client.get_card(ORDER_173) is card and route.call_count == 1
    assert len(client.cache) == 1
    await client.get_card(ORDER_173, use_cache=False)
    assert route.call_count == 2


@respx.mock
async def test_errors_are_mapped_to_readable_exceptions(client: CardServiceClient) -> None:
    respx.post(f"{BASE_URL}/core/cards/get").mock(side_effect=_serve)
    with pytest.raises(CardNotFoundError) as not_found:
        await client.get_card(OUTSIDE)
    assert not_found.value.card_id == OUTSIDE and not_found.value.status_code == 404

    respx.post(f"{BASE_URL}/core/cards/get").mock(
        return_value=httpx.Response(
            401, json={"error": "tessa_authentication_error", "message": "нет доступа"}
        )
    )
    with pytest.raises(CardServiceAuthError, match="401.*нет доступа"):
        await client.get_card(ORDER_144)

    respx.post(f"{BASE_URL}/core/cards/get").mock(return_value=httpx.Response(502, text="bad gateway"))
    with pytest.raises(CardServiceError, match="502.*bad gateway") as failed:
        await client.get_card(ORDER_144)
    assert failed.value.status_code == 502

    respx.post(f"{BASE_URL}/core/cards/get").mock(side_effect=httpx.ConnectError("refused"))
    with pytest.raises(CardServiceError, match="недоступен"):
        await client.get_card(ORDER_144)

    respx.post(f"{BASE_URL}/core/cards/get").mock(return_value=httpx.Response(200, json=[1, 2]))
    with pytest.raises(CardServiceError, match="не объект"):
        await client.get_card(ORDER_144)


@respx.mock
async def test_related_documents_resolve_incoming_type_via_source_card(client: CardServiceClient) -> None:
    route = respx.post(f"{BASE_URL}/core/cards/get").mock(side_effect=_serve)
    related = await client.get_related_documents(ORDER_109)
    by_id = {(item.doc_id, item.direction): item for item in related}
    outgoing = by_id[(OUTSIDE, "outgoing")]
    assert outgoing.relation_type == "ссылается на" and not outgoing.resolved_from_source
    assert outgoing.description and outgoing.doc_type_name == "Приказ"
    # входящие: тип восстановлен по исходящей связи карточки-источника (обратное имя)
    assert by_id[(ORDER_173, "incoming")].relation_type == "Отменено"
    assert by_id[(ORDER_144, "incoming")].relation_type == "дополнено"
    assert all(by_id[(doc, "incoming")].resolved_from_source for doc in (ORDER_173, ORDER_144))
    # источник вне сета недоступен — связь остаётся без типа, ошибки нет
    assert by_id[(OUTSIDE, "incoming")].relation_type is None
    assert not by_id[(OUTSIDE, "incoming")].resolved_from_source
    assert len(related) == 4
    # 109 + два источника + неудачный запрос вовне; повторный вызов — только из кэша
    assert route.call_count == 4
    await client.get_related_documents(ORDER_109)
    assert route.call_count == 5  # карточка вне сета не кэшируется (404), остальные из кэша


@respx.mock
async def test_relation_type_filter_is_case_insensitive(client: CardServiceClient) -> None:
    respx.post(f"{BASE_URL}/core/cards/get").mock(side_effect=_serve)
    cancelled = await client.get_related_documents(ORDER_109, relation_type="отменено")
    assert [item.doc_id for item in cancelled] == [ORDER_173]
    assert await client.get_related_documents(ORDER_109, relation_type="в отмену") == []
    cancelling = await client.get_related_documents(ORDER_173, relation_type="в отмену")
    assert [(item.doc_id, item.direction) for item in cancelling] == [(ORDER_109, "outgoing")]
    # источник без соответствующей исходящей связи — тип не восстановлен
    memo_links = await client.get_related_documents(ORDER_144)
    incoming = next(item for item in memo_links if item.direction == "incoming")
    assert incoming.doc_id == MEMO_9 and incoming.relation_type == "Документ-основание"
    assert relation_matches("В отмену", "в отмену") and not relation_matches(None, "в отмену")
    assert relation_matches(None, None)


def test_card_cache_evicts_least_recently_used() -> None:
    cache = CardCache(2)
    a, b, c = (stable_uuid("cache", name) for name in "abc")
    cache.put(a, {"id": "a"})
    cache.put(b, {"id": "b"})
    assert cache.get(a) == {"id": "a"}
    cache.put(c, {"id": "c"})
    assert cache.get(b) is None and cache.get(a) is not None and len(cache) == 2
    disabled = CardCache(0)
    disabled.put(a, {})
    assert len(disabled) == 0 and disabled.get(a) is None


async def test_client_with_injected_http_does_not_close_it() -> None:
    async with httpx.AsyncClient(base_url=BASE_URL) as http:
        client = CardServiceClient(BASE_URL, SETTINGS, http=http)
        await client.aclose()
        assert not http.is_closed
