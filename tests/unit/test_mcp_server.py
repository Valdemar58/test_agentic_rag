"""MCP-сервер (5.3–5.4): пять инструментов с русскими описаниями, вызовы через MCP, ошибки читаемы."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any
from uuid import UUID

import httpx
import pytest
import respx
from fastmcp import Client

from common.config import DEFAULT_CONFIG_PATH, load_app_config
from mcp_server.cards import CardServiceClient
from mcp_server.content import DocumentReader
from mcp_server.glossary import EMPTY_NOTE, EmptyGlossary
from mcp_server.server import HEALTH_PATH, SERVER_NAME, Services, build_server
from tessa_export.fake import link, make_snapshot, stable_uuid
from tests.unit.index_data import InMemoryCorpus, standard_corpus

CONFIG = load_app_config(DEFAULT_CONFIG_PATH)
CARDS_URL = "http://cards.test"
TOOLS = {
    "hybrid_search",
    "get_document_card",
    "get_related_documents",
    "get_document_content",
    "glossary_lookup",
}
ORDER_173, ORDER_109 = stable_uuid("mcp", "order173"), stable_uuid("mcp", "order109")
CARDS: dict[UUID, dict[str, Any]] = {
    ORDER_173: make_snapshot(
        ORDER_173,
        number="173",
        outgoing=[link(ORDER_109, ref_type_name="в отмену", ref_type_reverse_name="отменено")],
    ).card_data_json,
    ORDER_109: make_snapshot(
        ORDER_109,
        number="109",
        incoming=[link(ORDER_173, ref_type_name=None, ref_type_reverse_name=None)],
    ).card_data_json,
}


def _serve(request: httpx.Request) -> httpx.Response:
    card_id = UUID(httpx.Response(200, content=request.content).json()["card_id"])
    if card_id not in CARDS:
        return httpx.Response(
            404, json={"error": "resource_not_found", "message": f"card не найден: {card_id}"}
        )
    return httpx.Response(200, json=CARDS[card_id])


@pytest.fixture
def corpus() -> InMemoryCorpus:
    return standard_corpus()


def _services(corpus: InMemoryCorpus) -> Services:
    cards = CardServiceClient(CARDS_URL, CONFIG.card_service, username="robot", password="x")
    reader = DocumentReader(corpus.index.client, CONFIG.qdrant, CONFIG.retrieval)
    return Services(searcher=corpus.searcher(), cards=cards, reader=reader, glossary=EmptyGlossary())


@pytest.fixture
async def client(corpus: InMemoryCorpus) -> AsyncIterator[Client[Any]]:
    services = _services(corpus)
    server = build_server(CONFIG, services)
    with respx.mock(assert_all_called=False) as router:
        router.post(f"{CARDS_URL}/core/cards/get").mock(side_effect=_serve)
        async with Client(server) as connected:
            yield connected
    await services.aclose()


async def test_five_tools_are_listed_with_russian_descriptions(client: Client[Any]) -> None:
    tools = {tool.name: tool for tool in await client.list_tools()}
    assert set(tools) == TOOLS
    descriptions = {name: tool.description or "" for name, tool in tools.items()}
    assert all("Вызывай" in text for text in descriptions.values())
    search = tools["hybrid_search"]
    assert (
        "действующих" in descriptions["hybrid_search"]
        and "По умолчанию top_k" in descriptions["hybrid_search"]
    )
    properties = search.inputSchema["properties"]
    assert set(properties) >= {"query", "filters", "top_k"} and search.inputSchema["required"] == ["query"]
    schema_text = json.dumps(search.inputSchema, ensure_ascii=False)
    assert '"statuses"' in schema_text and "«Приказ»" in schema_text and "cancelled" in schema_text
    assert "DocumentCommonInfo" in descriptions["get_document_card"]
    assert "в отмену" in descriptions["get_related_documents"]
    assert "next_offset" in descriptions["get_document_content"]
    assert "аббревиатур" in descriptions["glossary_lookup"]


async def test_hybrid_search_via_mcp_returns_structured_hits(
    client: Client[Any], corpus: InMemoryCorpus
) -> None:
    result = await client.call_tool("hybrid_search", {"query": "отчёт по охране труда до пятого числа"})
    assert not result.is_error and result.structured_content
    hits = result.structured_content["hits"]
    assert hits and hits[0]["doc_id"] == corpus.docs["order_144"] and "пятого" in hits[0]["text"]
    assert {hit["doc_status"] for hit in hits} == {"active"}
    assert result.structured_content["applied_filters"] == {"statuses": ["active"]}

    cancelled = await client.call_tool(
        "hybrid_search",
        {"query": "отчёт до десятого числа", "filters": {"statuses": ["cancelled"]}, "top_k": 3},
    )
    assert cancelled.structured_content
    assert {hit["doc_id"] for hit in cancelled.structured_content["hits"]} == {
        corpus.docs["order_109_cancelled"]
    }


async def test_document_card_and_related_documents_via_mcp(client: Client[Any]) -> None:
    card = await client.call_tool("get_document_card", {"doc_id": str(ORDER_173)})
    assert card.structured_content
    assert card.structured_content["summary"]["label"] == "Приказ №173 от 15.01.2026"
    assert card.structured_content["summary"]["doc_status"] == "active"
    assert set(card.structured_content["card"]["sections"]) == {
        "DocumentCommonInfo",
        "OutgoingRefDocs",
        "IncomingRefDocs",
    }
    assert (
        card.structured_content["card"]["sections"]["OutgoingRefDocs"]["rows"][0]["RefTypeName"] == "в отмену"
    )

    related = await client.call_tool("get_related_documents", {"doc_id": str(ORDER_109)})
    assert related.structured_content
    items = related.structured_content["related"]
    assert [(item["doc_id"], item["relation_type"], item["direction"]) for item in items] == [
        (str(ORDER_173), "отменено", "incoming")
    ]
    assert related.structured_content["relation_types"] == ["отменено"]

    filtered = await client.call_tool(
        "get_related_documents", {"doc_id": str(ORDER_109), "relation_type": "дополнено"}
    )
    assert filtered.structured_content
    assert filtered.structured_content["related"] == [] and "отменено" in filtered.structured_content["note"]


async def test_document_content_and_glossary_via_mcp(client: Client[Any], corpus: InMemoryCorpus) -> None:
    content = await client.call_tool("get_document_content", {"doc_id": corpus.docs["order_144"]})
    assert content.structured_content
    sections = content.structured_content["sections"]
    assert (
        sections
        and "Контроль оставляю" in sections[0]["text"]
        and not content.structured_content["truncated"]
    )
    single = await client.call_tool(
        "get_document_content", {"doc_id": corpus.docs["order_144"], "section_id": sections[0]["section_id"]}
    )
    assert single.structured_content and single.structured_content["total_sections"] == 1

    glossary = await client.call_tool("glossary_lookup", {"term": "СИЗ"})
    assert glossary.structured_content == {"term": "СИЗ", "entries": [], "note": EMPTY_NOTE}


async def test_tool_errors_are_readable(client: Client[Any], corpus: InMemoryCorpus) -> None:
    missing = await client.call_tool(
        "get_document_card", {"doc_id": "00000000-0000-0000-0000-000000000000"}, raise_on_error=False
    )
    assert missing.is_error and "не найдена" in str(missing.content[0])
    bad_id = await client.call_tool("get_related_documents", {"doc_id": "не uuid"}, raise_on_error=False)
    assert bad_id.is_error and "UUID" in str(bad_id.content[0])
    bad_section = await client.call_tool(
        "get_document_content",
        {"doc_id": corpus.docs["order_144"], "section_id": "00000000-0000-0000-0000-000000000000"},
        raise_on_error=False,
    )
    assert bad_section.is_error and "не найден" in str(bad_section.content[0])
    failed = await client.call_tool(
        "hybrid_search", {"query": "отчёт", "filters": {"statuses": ["неизвестно"]}}, raise_on_error=False
    )
    assert failed.is_error


async def test_health_route_is_served_by_http_app(corpus: InMemoryCorpus) -> None:
    services = _services(corpus)
    app = build_server(CONFIG, services).http_app(path=CONFIG.mcp.path)
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://mcp.test") as http:
            response = await http.get(HEALTH_PATH)
    assert response.status_code == 200 and response.json() == {"status": "ok", "server": SERVER_NAME}
    await services.aclose()
