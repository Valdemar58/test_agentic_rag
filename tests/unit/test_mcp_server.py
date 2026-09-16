"""MCP-сервер (5.3): инструменты видны клиенту с русскими описаниями, hybrid_search работает через MCP."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest
from fastmcp import Client

from common.config import DEFAULT_CONFIG_PATH, load_app_config
from mcp_server.cards import CardServiceClient
from mcp_server.server import HEALTH_PATH, SERVER_NAME, Services, build_server
from tests.unit.index_data import InMemoryCorpus, standard_corpus

CONFIG = load_app_config(DEFAULT_CONFIG_PATH)


@pytest.fixture
def corpus() -> InMemoryCorpus:
    return standard_corpus()


@pytest.fixture
async def client(corpus: InMemoryCorpus) -> AsyncIterator[Client[Any]]:
    cards = CardServiceClient("http://cards.invalid", CONFIG.card_service, username="robot", password="x")
    server = build_server(CONFIG, Services(searcher=corpus.searcher(), cards=cards))
    async with Client(server) as connected:
        yield connected
    await cards.aclose()


async def test_tools_are_listed_with_russian_descriptions(client: Client[Any]) -> None:
    tools = {tool.name: tool for tool in await client.list_tools()}
    assert "hybrid_search" in tools
    search = tools["hybrid_search"]
    assert (
        search.description
        and "действующих" in search.description
        and "get_document_content" in search.description
    )
    assert "По умолчанию top_k" in search.description
    properties = search.inputSchema["properties"]
    assert set(properties) >= {"query", "filters", "top_k"} and search.inputSchema["required"] == ["query"]
    schema_text = json.dumps(search.inputSchema, ensure_ascii=False)
    assert '"statuses"' in schema_text and "«Приказ»" in schema_text and "cancelled" in schema_text


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


async def test_invalid_arguments_are_reported_not_crashing(client: Client[Any]) -> None:
    failed = await client.call_tool(
        "hybrid_search", {"query": "отчёт", "filters": {"statuses": ["неизвестно"]}}, raise_on_error=False
    )
    assert failed.is_error


async def test_health_route_is_served_by_http_app(corpus: InMemoryCorpus) -> None:
    cards = CardServiceClient("http://cards.invalid", CONFIG.card_service, username="robot", password="x")
    server = build_server(CONFIG, Services(searcher=corpus.searcher(), cards=cards))
    app = server.http_app(path=CONFIG.mcp.path)
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://mcp.test") as http:
            response = await http.get(HEALTH_PATH)
    assert response.status_code == 200 and response.json() == {"status": "ok", "server": SERVER_NAME}
    await cards.aclose()
