"""Общие фикстуры. Внешние контракты и локальный пример карточки опциональны:
без них зависящие тесты скипаются с понятным сообщением, а не падают."""

from __future__ import annotations

import json
from typing import Any

import pytest

from contracts.card_service import CardServiceContract, ContractsUnavailableError, load_contract
from contracts.external_paths import ExternalPaths


@pytest.fixture(scope="session")
def contract() -> CardServiceContract:
    try:
        return load_contract()
    except ContractsUnavailableError as exc:
        pytest.skip(f"Внешние контракты недоступны, задайте TESSA_SDK_PATH и CARD_SERVICE_PATH: {exc}")


@pytest.fixture(scope="session")
def card_example_raw() -> dict[str, Any]:
    path = ExternalPaths().tessa_card_example_path
    if not path.is_file():
        pytest.skip(f"Локальный пример карточки не найден: {path} (файл не коммитится)")
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data
