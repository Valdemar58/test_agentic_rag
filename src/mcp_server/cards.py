"""Клиент сервиса карточек (FR-2.2, FR-2.4): тонкая обёртка над `POST /core/cards/get`.

Один URL (`CARD_SERVICE_URL`) и учётка HTTP Basic из окружения — мок в dev и реальный сервис
заказчика в проде неотличимы для этого кода (§9, NFR-4). Карточка отдаётся как JSON схемы
`CardData` без потери полей; связи разбираются из секций `OutgoingRefDocs` (с типом) и
`IncomingRefDocs` (без типа): тип входящей связи восстанавливается по карточке-источнику,
которая берётся из кэша (N7). Эндпоинта связей у сервиса нет (проверено по коду).
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from typing import Any, Literal
from uuid import UUID

import httpx
from pydantic import BaseModel, Field

from common.config import CardServiceSettings
from ingest.cards import INCOMING_SECTION, OUTGOING_SECTION, CardRecord
from tessa_export.models import LinkInfo, parse_links

logger = logging.getLogger(__name__)

CARD_GET_PATH = "/core/cards/get"
READ_ONLY_MODE = 1  # CardGetMode.READ_ONLY: без блокировок и побочных эффектов на стороне Тессы
Direction = Literal["outgoing", "incoming"]


class CardServiceError(Exception):
    """Сервис карточек недоступен или ответил ошибкой."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class CardNotFoundError(CardServiceError):
    """Карточка с таким id не найдена (404 сервиса)."""

    def __init__(self, card_id: UUID) -> None:
        super().__init__(f"карточка {card_id} не найдена", status_code=404)
        self.card_id = card_id


class CardServiceAuthError(CardServiceError):
    """Учётные данные отклонены (401/403)."""


class RelatedDocument(BaseModel):
    """Связанный документ с точки зрения запрошенного документа."""

    doc_id: UUID = Field(description="ID карточки связанного документа")
    description: str | None = Field(default=None, description="Описание документа из секции связей")
    doc_type_name: str | None = Field(default=None, description="Вид связанного документа, если известен")
    relation_type: str | None = Field(
        default=None, description="Тип связи от запрошенного документа к связанному (как в справочнике Тессы)"
    )
    direction: Direction = Field(description="outgoing — ссылка из документа, incoming — ссылка на документ")
    resolved_from_source: bool = Field(
        default=False, description="Тип входящей связи восстановлен по карточке документа-источника"
    )


def _normalize(value: str | None) -> str:
    return (value or "").strip().casefold()


def relation_matches(relation_type: str | None, wanted: str | None) -> bool:
    """Фильтр по типу связи без учёта регистра: в справочнике «В отмену» и «в отмену» (О4)."""
    return wanted is None or _normalize(relation_type) == _normalize(wanted)


class CardCache:
    """LRU-кэш карточек в памяти; размер 0 выключает кэширование."""

    def __init__(self, size: int) -> None:
        self._size = size
        self._items: OrderedDict[UUID, dict[str, Any]] = OrderedDict()

    def get(self, card_id: UUID) -> dict[str, Any] | None:
        item = self._items.get(card_id)
        if item is not None:
            self._items.move_to_end(card_id)
        return item

    def put(self, card_id: UUID, card: dict[str, Any]) -> None:
        if self._size <= 0:
            return
        self._items[card_id] = card
        self._items.move_to_end(card_id)
        while len(self._items) > self._size:
            self._items.popitem(last=False)

    def clear(self) -> None:
        self._items.clear()

    def __len__(self) -> int:
        return len(self._items)


class CardServiceClient:
    """Асинхронный клиент; `http` подменяется в тестах (respx) и закрывается вызывающей стороной."""

    def __init__(
        self,
        base_url: str,
        settings: CardServiceSettings,
        *,
        username: str = "",
        password: str = "",
        http: httpx.AsyncClient | None = None,
    ) -> None:
        self._settings = settings
        self._owns_http = http is None
        self._http = http or httpx.AsyncClient(
            base_url=base_url.rstrip("/"), auth=(username, password), timeout=settings.timeout_s
        )
        self._cache = CardCache(settings.cache_size)
        if not username:
            logger.warning("Учётная запись сервиса карточек пуста: мок примет, реальный сервис ответит 401")

    @property
    def cache(self) -> CardCache:
        return self._cache

    async def aclose(self) -> None:
        if self._owns_http:
            await self._http.aclose()

    async def get_card(self, card_id: UUID, *, use_cache: bool = True) -> dict[str, Any]:
        """Карточка как JSON `CardData` реального сервиса — без потери полей (FR-2.2)."""
        if use_cache:
            cached = self._cache.get(card_id)
            if cached is not None:
                return cached
        payload = {"card_id": str(card_id), "mode": READ_ONLY_MODE}
        try:
            response = await self._http.post(CARD_GET_PATH, json=payload)
        except httpx.HTTPError as exc:
            raise CardServiceError(f"сервис карточек недоступен: {exc}") from exc
        if response.status_code == 404:
            raise CardNotFoundError(card_id)
        if response.status_code in (401, 403):
            raise CardServiceAuthError(
                f"сервис карточек отклонил учётные данные ({response.status_code}): {_error_text(response)}",
                status_code=response.status_code,
            )
        if response.status_code != 200:
            raise CardServiceError(
                f"сервис карточек ответил {response.status_code}: {_error_text(response)}",
                status_code=response.status_code,
            )
        card = response.json()
        if not isinstance(card, dict):
            raise CardServiceError("сервис карточек вернул не объект карточки")
        self._cache.put(card_id, card)
        return card

    async def get_related_documents(
        self, card_id: UUID, relation_type: str | None = None
    ) -> list[RelatedDocument]:
        """Связи документа из секций карточки (FR-2.4); `relation_type` — фильтр без учёта регистра."""
        record = CardRecord.model_validate(await self.get_card(card_id))
        related: list[RelatedDocument] = []
        for link in parse_links(record.rows(OUTGOING_SECTION)):
            related.append(_related(link, link.ref_type_name, "outgoing", resolved=False))
        for link in parse_links(record.rows(INCOMING_SECTION)):
            relation, from_source = await self._incoming_relation(card_id, link)
            related.append(_related(link, relation, "incoming", resolved=from_source))
        return [item for item in related if relation_matches(item.relation_type, relation_type)]

    async def _incoming_relation(self, card_id: UUID, link: LinkInfo) -> tuple[str | None, bool]:
        """Тип входящей связи = обратное имя типа исходящей связи в карточке-источнике (N7).

        Возвращает тип и признак, что он взят из карточки-источника, а не из самой строки."""
        if link.ref_type_reverse_name or link.ref_type_name:
            return link.ref_type_reverse_name or link.ref_type_name, False
        try:
            source = CardRecord.model_validate(await self.get_card(link.doc_id))
        except CardServiceError as exc:
            logger.warning("Тип входящей связи %s → %s не восстановлен: %s", link.doc_id, card_id, exc)
            return None, False
        for outgoing in parse_links(source.rows(OUTGOING_SECTION)):
            if outgoing.doc_id == card_id:
                return outgoing.ref_type_reverse_name or outgoing.ref_type_name, True
        return None, False


def _related(
    link: LinkInfo, relation_type: str | None, direction: Direction, *, resolved: bool
) -> RelatedDocument:
    return RelatedDocument(
        doc_id=link.doc_id,
        description=link.description,
        doc_type_name=link.doc_type_name,
        relation_type=relation_type,
        direction=direction,
        resolved_from_source=resolved,
    )


def _error_text(response: httpx.Response) -> str:
    try:
        body = response.json()
    except ValueError:
        return response.text[:200]
    if isinstance(body, dict):
        return str(body.get("message") or body.get("error") or body)[:200]
    return str(body)[:200]
