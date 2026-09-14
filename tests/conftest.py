"""Общие фикстуры. Внешние контракты опциональны: без них зависящие тесты
скипаются с понятным сообщением, а не падают."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from contracts.card_service import CardServiceContract, ContractsUnavailableError, load_contract

FIXTURES_DIR = Path(__file__).parent / "fixtures"
# Обезличенный сырой ответ Тессы cards/get для реального приказа (scripts/anonymize_card_example.py)
ORDER_CARD_RESPONSE = FIXTURES_DIR / "tessa" / "order_card_response.json"


@pytest.fixture(scope="session")
def contract() -> CardServiceContract:
    try:
        return load_contract()
    except ContractsUnavailableError as exc:
        pytest.skip(f"Внешние контракты недоступны, задайте TESSA_SDK_PATH и CARD_SERVICE_PATH: {exc}")


@pytest.fixture(scope="session")
def card_example_raw() -> dict[str, Any]:
    data: dict[str, Any] = json.loads(ORDER_CARD_RESPONSE.read_text(encoding="utf-8"))
    return data
