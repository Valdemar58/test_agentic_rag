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
from uuid import UUID

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from starlette.requests import Request
from starlette.responses import JSONResponse

from common.config import AppConfig
from common.settings import Settings
from ingest.embeddings import build_embedder
from ingest.glossary_index import GlossaryIndex
from mcp_server.card_view import DocumentCard, build_document_card
from mcp_server.cards import CardNotFoundError, CardServiceClient, CardServiceError, RelatedDocument
from mcp_server.content import ContentNotFoundError, DocumentContent, DocumentReader
from mcp_server.glossary import EmptyGlossary, GlossaryLookup, GlossaryResult, QdrantGlossary
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


class RelatedDocumentsResult(BaseModel):
    doc_id: str
    related: list[RelatedDocument]
    relation_types: list[str] = Field(description="Типы связей, встретившиеся у документа (без фильтра)")
    note: str | None = None


@dataclass
class Services:
    searcher: HybridSearcher
    cards: CardServiceClient
    reader: DocumentReader
    glossary: GlossaryLookup

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
    reader = DocumentReader(client, config.qdrant, config.retrieval)
    glossary: GlossaryLookup = EmptyGlossary()
    if config.glossary.enabled:
        index = GlossaryIndex(client, config.qdrant, config.embedding.dense_dim)
        glossary = QdrantGlossary(index, embedder, config.glossary)
    return Services(searcher=searcher, cards=cards, reader=reader, glossary=glossary)


def _parse_doc_id(doc_id: str) -> UUID:
    try:
        return UUID(doc_id)
    except ValueError as exc:
        raise ToolError(f"doc_id должен быть UUID карточки СЭД, получено: {doc_id!r}") from exc


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

    async def get_document_card(
        doc_id: str, sections: list[str] | None = None, full: bool = False
    ) -> DocumentCard:
        """Карточка документа СЭД по его ID: реквизиты (вид, номер, дата, тема, подразделение,
        автор, подписант), статус документа (действует, отменён, проект), состояние маршрута и
        согласования, согласующие и ответственные, связи с другими документами, список файлов.

        Вызывай, когда нужны реквизиты или статус документа, найденного поиском (doc_id из
        hybrid_search), или чтобы проверить, действует ли документ. В summary — краткая сводка,
        в card — карточка в форме CardData сервиса карточек без переименования полей. По умолчанию
        возвращаются основные секции карточки; sections — выбрать секции по именам, full=true —
        карточка целиком (включая историю согласования, это большой ответ).
        """
        card_id = _parse_doc_id(doc_id)
        try:
            card_json = await services.cards.get_card(card_id)
        except CardNotFoundError as exc:
            raise ToolError(f"Карточка {doc_id} не найдена в СЭД") from exc
        except CardServiceError as exc:
            raise ToolError(f"Сервис карточек недоступен: {exc}") from exc
        return build_document_card(
            card_json,
            sections=sections,
            full=full,
            settings=config.card_service,
            status_rules=config.ingest.status,
        )

    mcp.tool(
        get_document_card,
        description=_description(
            get_document_card,
            "Секции по умолчанию: " + ", ".join(config.card_service.default_sections) + ".",
        ),
    )

    async def get_related_documents(doc_id: str, relation_type: str | None = None) -> RelatedDocumentsResult:
        """Связанные документы по связям из карточки СЭД: какой приказ отменил или дополнил данный,
        к какому документу он приложение или основание, ответ на какое письмо, основной договор
        или дополнительное соглашение.

        Вызывай, чтобы узнать, чем заменён или дополнен документ, найти документ-основание,
        доп. соглашения к договору или ответ на письмо. relation_type — фильтр по типу связи без
        учёта регистра; типы как в справочнике СЭД: «в отмену» / «отменено», «дополнение» /
        «дополнено», «Приказ» / «Документ-основание», «Основной договор» / «Доп. соглашение»,
        «запрос» / «ответ». Тип указан с точки зрения запрошенного документа; для входящих связей
        он восстановлен по карточке документа-источника. Реквизиты связанного документа —
        через get_document_card по его doc_id.
        """
        card_id = _parse_doc_id(doc_id)
        try:
            related = await services.cards.get_related_documents(card_id, relation_type)
            if relation_type is not None:
                everything = await services.cards.get_related_documents(card_id)
            else:
                everything = related
        except CardNotFoundError as exc:
            raise ToolError(f"Карточка {doc_id} не найдена в СЭД") from exc
        except CardServiceError as exc:
            raise ToolError(f"Сервис карточек недоступен: {exc}") from exc
        types = sorted({item.relation_type for item in everything if item.relation_type})
        note = None
        if relation_type is not None and not related and everything:
            note = (
                f"связей типа «{relation_type}» нет; у документа есть связи: {', '.join(types) or 'без типа'}"
            )
        return RelatedDocumentsResult(doc_id=doc_id, related=related, relation_types=types, note=note)

    mcp.tool(get_related_documents, description=_description(get_related_documents))

    def get_document_content(doc_id: str, section_id: str | None = None, offset: int = 0) -> DocumentContent:
        """Полный текст документа или одного его раздела из проиндексированного корпуса.

        Вызывай после hybrid_search, когда найденного фрагмента недостаточно: чтобы дочитать
        раздел целиком (section_id — parent_id или chunk_id из результата поиска) или прочитать
        весь документ (без section_id — разделы по порядку: основной текст, затем приложения).
        Длинный документ отдаётся частями: если truncated = true, вызови ещё раз с offset =
        next_offset. Текст берётся из индекса, а не из СЭД: для документов без текста в индексе
        ответ пустой с пояснением.
        """
        try:
            return services.reader.read(doc_id, section_id, offset)
        except ContentNotFoundError as exc:
            raise ToolError(str(exc)) from exc

    mcp.tool(
        get_document_content,
        description=_description(
            get_document_content,
            f"За один вызов не больше {retrieval.content_max_tokens} токенов "
            f"и {retrieval.content_max_sections} разделов.",
        ),
    )

    def glossary_lookup(term: str) -> GlossaryResult:
        """Расшифровка корпоративной аббревиатуры или термина по глоссарию организации, собранному
        из разделов «Термины и определения» и «Сокращения» документов.

        Вызывай, если в вопросе встретилась аббревиатура или внутренний термин, значение которого
        нужно уточнить перед поиском (например, СИЗ, ПВТР, ЛНА, ДОУ). Возвращает найденные
        определения с документами-источниками; если термина нет, entries пуст и note объясняет
        причину — тогда ищи через hybrid_search по самому термину.
        """
        return services.glossary.lookup(term)

    mcp.tool(glossary_lookup, description=_description(glossary_lookup))

    @mcp.custom_route(HEALTH_PATH, methods=["GET"])
    async def health(request: Request) -> JSONResponse:
        return JSONResponse({"status": "ok", "server": SERVER_NAME})

    return mcp
