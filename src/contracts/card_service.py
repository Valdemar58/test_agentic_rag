"""Контракт карточек: схемы сервиса карточек заказчика и модели SDK Тессы.

Единый контракт для экспорт-скрипта, мока сервиса, маппинга метаданных чанков и контрактных
тестов (§8.0 ТЗ). Схемы импортируются лениво после подключения внешних путей, поэтому модуль
безопасно импортировать и там, где внешний код отсутствует.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from contracts.external_paths import ExternalPaths, ensure_external_paths


class ContractsUnavailableError(RuntimeError):
    """Внешние пути не заданы или не содержат ожидаемых пакетов."""


@dataclass(frozen=True)
class CardServiceContract:
    """Классы схем сервиса карточек и модели SDK, нужные RAG."""

    card_data: type[BaseModel]
    card_section_data: type[BaseModel]
    card_file_data: type[BaseModel]
    card_permission_data: type[BaseModel]
    card_get_in: type[BaseModel]
    card_get_file_content_in: type[BaseModel]
    card_file_versions_out: type[BaseModel]
    sdk_card: type[BaseModel]
    sdk_card_get_response: type[BaseModel]
    normalize_tessa_json: Callable[[Any], Any]


_contract_cache: CardServiceContract | None = None


def load_contract(paths: ExternalPaths | None = None) -> CardServiceContract:
    """Подключает внешние пути и импортирует схемы. Результат кэшируется."""
    global _contract_cache
    if _contract_cache is not None:
        return _contract_cache
    status = ensure_external_paths(paths)
    if not status.available:
        raise ContractsUnavailableError(status.reason or "внешний код недоступен")

    from robot_skills.core.cards import schemas
    from tessa_client.models.card import Card
    from tessa_client.models.card_responses import CardGetResponse
    from tessa_client.typed_json import normalize_tessa_json

    _contract_cache = CardServiceContract(
        card_data=schemas.CardData,
        card_section_data=schemas.CardSectionData,
        card_file_data=schemas.CardFileData,
        card_permission_data=schemas.CardPermissionData,
        card_get_in=schemas.CardGetIn,
        card_get_file_content_in=schemas.CardGetFileContentIn,
        card_file_versions_out=schemas.CardFileVersionsOut,
        sdk_card=Card,
        sdk_card_get_response=CardGetResponse,
        normalize_tessa_json=normalize_tessa_json,
    )
    return _contract_cache


def card_data_from_tessa_response(
    raw_response: dict[str, Any], contract: CardServiceContract | None = None
) -> BaseModel:
    """Сырой JSON ответа POST /api/v1/cards/get → CardData тем же путём, что и в сервисе заказчика.

    Путь: normalize_tessa_json → CardGetResponse → Card.model_dump() → CardData.model_validate.
    """
    active = contract if contract is not None else load_contract()
    response = active.sdk_card_get_response.model_validate(active.normalize_tessa_json(raw_response))
    card: Any = getattr(response, "card", None)
    if card is None:
        raise ValueError("В ответе Тессы нет карточки (Card = null)")
    return active.card_data.model_validate(card.model_dump())
