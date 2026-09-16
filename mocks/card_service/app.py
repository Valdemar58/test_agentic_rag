"""Сборка мока: контракт по внешним путям → хранилище из архива → FastAPI-приложение.

Модуль безопасно импортировать без внешнего кода: схемы `robot_skills` подключаются
только в `create_app`/`load_store` (§8.0 ТЗ).
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI

from common.config import MockCardServiceSettings
from contracts.card_service import CardServiceContract, load_contract
from mocks.card_service.store import CardStore


def load_store(root: Path, contract: CardServiceContract | None = None) -> CardStore:
    """Читает архив экспорта; без внешних путей — `ContractsUnavailableError`."""
    return CardStore.load(root, contract or load_contract())


def create_app(store: CardStore, settings: MockCardServiceSettings) -> FastAPI:
    load_contract()
    from mocks.card_service import routes

    return routes.build_app(store, settings)
