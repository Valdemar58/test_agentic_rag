"""MCP-сервер инструментов агента (FR-2): FastMCP, транспорт streamable-http (N14).

Инструменты: `hybrid_search` (5.3), `get_document_card`, `get_related_documents`,
`get_document_content`, `glossary_lookup` (5.4). Каждый инструмент документирован по-русски так,
чтобы LLM выбирала его по описанию (AC-2.1). Сервер запускается автономно (`python -m mcp_server`,
AC-2.3); зависимости собираются в `Services`, в тестах подменяются фейками.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from fastmcp import FastMCP
from qdrant_client import QdrantClient
from starlette.requests import Request
from starlette.responses import JSONResponse

from common.config import AppConfig
from common.settings import Settings
from ingest.embeddings import build_embedder
from mcp_server.cards import CardServiceClient
from mcp_server.reranker import BgeReranker
from mcp_server.retrieval import HybridSearcher, SearchFilters, SearchResult

SERVER_NAME = "sed-documents"
HEALTH_PATH = "/health"
INSTRUCTIONS = (
    "Инструменты поиска и чтения документов СЭД организации: приказы, положения, инструкции, "
    "договоры, служебные записки, письма. Сначала ищи фрагменты через hybrid_search, затем при "
    "необходимости дочитывай раздел или документ через get_document_content, реквизиты и статус "
    "смотри в get_document_card, связи (отменён, изменён, приложение) — в get_related_documents."
)


@dataclass
class Services:
    searcher: HybridSearcher
    cards: CardServiceClient

    def warm_up(self) -> None:
        """Загружает модели до первого запроса, чтобы первый ответ агенту не ждал их инициализации."""
        self.searcher.warm_up()

    async def aclose(self) -> None:
        await self.cards.aclose()


def build_services(config: AppConfig, settings: Settings) -> Services:
    """Живые зависимости: Qdrant, bge-m3 и reranker на CPU (§2), клиент сервиса карточек."""
    client = QdrantClient(url=settings.resolve_qdrant_url(), timeout=int(config.qdrant.timeout_s))
    embedder = build_embedder(
        config.models.local_path(config.models.embedding), config.embedding, ingest=False
    )
    reranker = BgeReranker(config.models.local_path(config.models.reranker), config.reranker)
    searcher = HybridSearcher(client, config.qdrant, config.retrieval, embedder, reranker)
    cards = CardServiceClient(
        settings.resolve_card_service_url(config),
        config.card_service,
        username=settings.card_service_username,
        password=settings.card_service_password.get_secret_value(),
    )
    return Services(searcher=searcher, cards=cards)


def _description(function: Callable[..., Any], *extra: str) -> str:
    """Docstring инструмента плюс строки с параметрами из конфига — то, что видит LLM."""
    return "\n\n".join([inspect.cleandoc(function.__doc__ or ""), *extra])


def build_server(config: AppConfig, services: Services) -> FastMCP:
    mcp = FastMCP(SERVER_NAME, instructions=INSTRUCTIONS)
    retrieval = config.retrieval

    def hybrid_search(
        query: str, filters: SearchFilters | None = None, top_k: int | None = None
    ) -> SearchResult:
        """Поиск фрагментов текста в документах СЭД: приказах, положениях, инструкциях, регламентах,
        договорах, служебных записках и письмах.

        Вызывай для любого вопроса о содержании документов: что предписано, кто назначен
        ответственным, какие сроки и условия, что утверждено приказом. Запрос формулируй по-русски
        ключевыми словами из вопроса (без лишних слов). По умолчанию ищет только в действующих
        документах; отменённые документы и проекты доступны через filters.statuses. Фильтры по виду
        документа, дате и подразделению сужают поиск. Возвращает до top_k фрагментов с оценкой
        релевантности, координатами (документ, раздел, пункт, страница) и текстом раздела-родителя.
        Полный текст документа читай через get_document_content, реквизиты и статус карточки —
        через get_document_card, связанные документы — через get_related_documents.
        """
        return services.searcher.search(query, filters, top_k)

    mcp.tool(
        hybrid_search,
        description=_description(
            hybrid_search, f"По умолчанию top_k = {retrieval.top_k}, максимум {retrieval.max_top_k}."
        ),
    )

    @mcp.custom_route(HEALTH_PATH, methods=["GET"])
    async def health(request: Request) -> JSONResponse:
        return JSONResponse({"status": "ok", "server": SERVER_NAME})

    return mcp
