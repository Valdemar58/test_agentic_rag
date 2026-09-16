"""AC-2.4: мок и реальный сервис карточек описываются одними pydantic-схемами.

Сравниваются OpenAPI мока и реального приложения (`create_app(...)` из `CARD_SERVICE_PATH`,
с фейковыми настройками и без подключения к Тессе) по подмножеству маршрутов, которое реализует мок:
тело запроса, схема ответа, параметры пути, требование Basic auth и определения всех схем компонентов,
до которых можно дойти по `$ref`. Ответы мока валидируются схемой `CardData` реального сервиса.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from common.config import DEFAULT_CONFIG_PATH, load_app_config
from contracts.card_service import CardServiceContract
from mocks.card_service.app import create_app, load_store
from synthetic.corpus import ORDER_144, generate_corpus

pytestmark = pytest.mark.contract

SETTINGS = load_app_config(DEFAULT_CONFIG_PATH).mock_card_service
AUTH = ("robot", "secret")
# Переменные окружения реального сервиса: значения фиктивные, соединения не устанавливаются
REAL_SERVICE_ENV = {
    "TESSA_BASE_URL": "http://tessa.invalid",
    "ASU_BASE_URL": "http://asu.invalid",
    "ASU_API_KEY": "contract-test",
    "DATABASE_URL": "postgresql+asyncpg://user:pass@db.invalid/db",
}
COMPARED_OPERATION_KEYS = ("requestBody", "responses", "security", "parameters")


def _collect_refs(node: Any, found: set[str]) -> None:
    if isinstance(node, dict):
        ref = node.get("$ref")
        if isinstance(ref, str) and ref.startswith("#/components/schemas/"):
            found.add(ref.rsplit("/", 1)[1])
        for value in node.values():
            _collect_refs(value, found)
    elif isinstance(node, list):
        for item in node:
            _collect_refs(item, found)


def _reachable_schemas(spec: dict[str, Any], operations: list[dict[str, Any]]) -> set[str]:
    """Имена схем компонентов, достижимые из операций по `$ref` (транзитивно)."""
    pending: set[str] = set()
    for operation in operations:
        _collect_refs(operation, pending)
    seen: set[str] = set()
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        seen.add(name)
        _collect_refs(spec["components"]["schemas"].get(name, {}), pending)
    return seen


@pytest.fixture(scope="module")
def real_spec(contract: CardServiceContract) -> dict[str, Any]:
    previous = {key: os.environ.get(key) for key in REAL_SERVICE_ENV}
    os.environ.update(REAL_SERVICE_ENV)
    try:
        from robot_skills.config import Settings
        from robot_skills.main import create_app as create_real_app

        app = create_real_app(Settings(_env_file=None), tessa_client_factory=lambda user, password: None)
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    spec: dict[str, Any] = app.openapi()
    return spec


@pytest.fixture(scope="module")
def mock_app(contract: CardServiceContract, tmp_path_factory: pytest.TempPathFactory) -> Iterator[Any]:
    corpus = tmp_path_factory.mktemp("contract") / "corpus"
    generate_corpus(corpus)
    yield create_app(load_store(corpus, contract), SETTINGS)


def test_mock_routes_are_a_subset_with_identical_schemas(mock_app: Any, real_spec: dict[str, Any]) -> None:
    mock_spec = mock_app.openapi()
    mock_operations: list[tuple[str, str, dict[str, Any]]] = [
        (path, method, operation)
        for path, methods in mock_spec["paths"].items()
        for method, operation in methods.items()
    ]
    assert {path for path, _, _ in mock_operations} >= {
        "/health",
        "/core/cards/get",
        "/core/cards/get-file-content",
        "/core/card-types",
        "/core/card-types/{type_id}",
        "/core/views/{view_alias}/get-data",
    }
    for path, method, operation in mock_operations:
        assert path in real_spec["paths"], f"маршрута {path} нет в реальном сервисе"
        real_operation = real_spec["paths"][path].get(method)
        assert real_operation is not None, f"{method.upper()} {path} нет в реальном сервисе"
        for key in COMPARED_OPERATION_KEYS:
            assert operation.get(key) == real_operation.get(key), f"{method.upper()} {path}: {key} отличается"

    mock_schemas = _reachable_schemas(mock_spec, [operation for _, _, operation in mock_operations])
    assert {
        "CardData",
        "CardGetIn",
        "CardGetFileContentIn",
        "CardTypeOut",
        "ViewGetDataIn",
        "ViewResultOut",
    } <= (mock_schemas)
    for name in sorted(mock_schemas):
        assert mock_spec["components"]["schemas"][name] == real_spec["components"]["schemas"][name], (
            f"схема {name} отличается"
        )
    assert mock_spec["components"]["securitySchemes"] == real_spec["components"]["securitySchemes"]


def test_mock_responses_validate_against_real_schemas(mock_app: Any, contract: CardServiceContract) -> None:
    from robot_skills.core.card_types.schemas import CardTypeOut
    from robot_skills.core.views.schemas import ViewResultOut

    with TestClient(mock_app) as client:
        card = client.post("/core/cards/get", json={"card_id": str(ORDER_144)}, auth=AUTH).json()
        validated = contract.card_data.model_validate(card)
        assert validated.model_dump(mode="json")["id"] == str(ORDER_144)
        types = client.get("/core/card-types", auth=AUTH).json()
        assert [CardTypeOut.model_validate(item).name for item in types]
        view = client.post(f"/core/views/{SETTINGS.ref_type_view}/get-data", json={}, auth=AUTH).json()
        assert ViewResultOut.model_validate(view).columns == SETTINGS.ref_type_columns


def test_store_is_read_only_against_the_export(tmp_path: Path, contract: CardServiceContract) -> None:
    """Мок не пишет в архив: после старта и запросов состав каталога экспорта не меняется."""
    corpus = tmp_path / "corpus"
    generate_corpus(corpus)
    before = sorted(str(item.relative_to(corpus)) for item in corpus.rglob("*"))
    with TestClient(create_app(load_store(corpus, contract), SETTINGS)) as client:
        client.post("/core/cards/get", json={"card_id": str(ORDER_144)}, auth=AUTH)
    assert sorted(str(item.relative_to(corpus)) for item in corpus.rglob("*")) == before
