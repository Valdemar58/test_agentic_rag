"""Мок сервиса карточек (§9, задача 5.1): маршруты и схемы реального сервиса на синтетическом корпусе.

Тесты требуют внешних путей: мок использует схемы `robot_skills` как есть.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from pathlib import Path
from urllib.parse import quote
from uuid import uuid4

import pytest
import structlog
from fastapi.testclient import TestClient

from common.config import DEFAULT_CONFIG_PATH, load_app_config
from contracts.card_service import CardServiceContract
from mocks.card_service.app import create_app, load_store
from mocks.card_service.store import CardStore
from synthetic.corpus import CONTRACT_D1, ORDER_144, ORDER_173, generate_corpus
from tessa_export.storage import CARDS_DIR, export_dir

pytestmark = pytest.mark.contract

# HTTPBasic Starlette (и реального сервиса) принимает только ASCII-учётки
AUTH = ("robot", "secret")
SETTINGS = load_app_config(DEFAULT_CONFIG_PATH).mock_card_service


@pytest.fixture(autouse=True)
def _reset_logging() -> Iterator[None]:
    yield
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()
    structlog.reset_defaults()


@pytest.fixture(scope="module")
def corpus_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    output = tmp_path_factory.mktemp("mock") / "corpus"
    generate_corpus(output)
    return output


@pytest.fixture(scope="module")
def store(contract: CardServiceContract, corpus_dir: Path) -> CardStore:
    return load_store(corpus_dir, contract)


@pytest.fixture(scope="module")
def client(store: CardStore) -> Iterator[TestClient]:
    with TestClient(create_app(store, SETTINGS)) as client:
        yield client


def test_store_loads_cards_files_and_reference_data(store: CardStore, corpus_dir: Path) -> None:
    assert store.synthetic and store.invalid_cards == 0
    assert len(store.cards) == 13 and "СИНТЕТИЧЕСКИЕ" in store.summary_lines()[0]
    assert {item.name for item in store.card_types} == {
        "OrderMKC",
        "ContractMKC",
        "InhouseDocumentMKC",
        "IncomingMKC",
        "OutgoingMKC",
        "BackgroundJustificationMKC",
    }
    names = {item.name: item.reverse_name for item in store.ref_types}
    assert names["в отмену"] == "отменено" and names["Основной договор"] == "Доп. соглашение"
    assert all(item.id is not None for item in store.ref_types)
    # файлы: у виртуального листа согласования и .sig содержимого в архиве нет
    virtual = [item for item in store.files.values() if item.path is None]
    assert virtual and all(item.skipped_reason for item in virtual)
    assert all(item.path.is_file() for item in store.files.values() if item.path is not None)


def test_health_and_basic_auth(client: TestClient) -> None:
    assert client.get("/health").json() == {"status": "ok"}
    unauthorized = client.post("/core/cards/get", json={"card_id": str(ORDER_144)})
    assert unauthorized.status_code == 401 and unauthorized.headers["www-authenticate"].startswith("Basic")
    for credentials in (("user", "any-password"), ("DOMAIN\\someone", "")):
        assert (
            client.post("/core/cards/get", json={"card_id": str(ORDER_144)}, auth=credentials).status_code
            == 200
        )


def test_get_card_returns_card_data_as_exported(
    client: TestClient, contract: CardServiceContract, corpus_dir: Path
) -> None:
    response = client.post("/core/cards/get", json={"card_id": str(ORDER_173), "mode": 1}, auth=AUTH)
    assert response.status_code == 200 and response.headers["x-request-id"]
    served = contract.card_data.model_validate(response.json())
    on_disk = contract.card_data.model_validate(
        json.loads((export_dir(corpus_dir) / CARDS_DIR / f"{ORDER_173}.json").read_text(encoding="utf-8"))
    )
    assert served.model_dump(mode="json") == on_disk.model_dump(mode="json")
    body = response.json()
    assert body["type_name"] == "OrderMKC" and body["sections"]["DocumentCommonInfo"]["fields"]["FullNumber"]
    outgoing = body["sections"]["OutgoingRefDocs"]["rows"]
    assert outgoing[0]["RefTypeName"] == "в отмену" and outgoing[0]["DocID"]
    assert body["files"] and all("version_row_id" in item for item in body["files"])


def test_get_card_not_found_uses_service_error_format(client: TestClient) -> None:
    missing = uuid4()
    response = client.post("/core/cards/get", json={"card_id": str(missing)}, auth=AUTH)
    assert response.status_code == 404
    body = response.json()
    assert body["error"] == "resource_not_found" and str(missing) in body["message"]
    assert body["request_id"] == response.headers["x-request-id"]
    # без card_id реальный сервис тоже не находит карточку
    assert client.post("/core/cards/get", json={"card_type_name": "OrderMKC"}, auth=AUTH).status_code == 404
    invalid = client.post("/core/cards/get", json={"card_id": "не uuid"}, auth=AUTH)
    assert invalid.status_code == 422 and invalid.json()["error"] == "request_validation_error"
    assert invalid.json()["validation_items"]


def test_get_file_content_streams_latest_version(client: TestClient, store: CardStore) -> None:
    card = store.card(CONTRACT_D1)
    assert card is not None
    xlsx = next(item for item in card.files if item.name.endswith(".xlsx"))
    stored = store.file(CONTRACT_D1, xlsx.row_id)
    assert stored is not None and stored.path is not None
    payload = {
        "card_id": str(CONTRACT_D1),
        "file_id": str(xlsx.row_id),
        "version_row_id": str(xlsx.version_row_id),
    }
    response = client.post("/core/cards/get-file-content", json=payload, auth=AUTH)
    assert response.status_code == 200
    assert response.content == stored.path.read_bytes()
    assert (
        response.headers["content-disposition"] == f"attachment; filename*=UTF-8''{quote(xlsx.name, safe='')}"
    )
    assert "spreadsheet" in response.headers["content-type"]

    stale = client.post(
        "/core/cards/get-file-content", json={**payload, "version_row_id": str(uuid4())}, auth=AUTH
    )
    assert stale.status_code == 404 and "последняя версия" in stale.json()["message"]
    unknown = client.post(
        "/core/cards/get-file-content", json={**payload, "file_id": str(uuid4())}, auth=AUTH
    )
    assert unknown.status_code == 404 and unknown.json()["error"] == "resource_not_found"


def test_get_file_content_without_archived_content_is_404(client: TestClient, store: CardStore) -> None:
    card = store.card(ORDER_144)
    assert card is not None
    virtual = next(item for item in card.files if item.is_virtual)
    payload = {
        "card_id": str(ORDER_144),
        "file_id": str(virtual.row_id),
        "version_row_id": str(virtual.version_row_id),
    }
    response = client.post("/core/cards/get-file-content", json=payload, auth=AUTH)
    assert response.status_code == 404 and "нет содержимого" in response.json()["message"]


def test_card_types_from_corpus(client: TestClient, store: CardStore) -> None:
    listed = client.get("/core/card-types", auth=AUTH)
    assert listed.status_code == 200
    by_name = {item["name"]: item for item in listed.json()}
    assert by_name["OrderMKC"]["caption"] == "Приказ" and len(by_name) == len(store.card_types)
    single = client.get(f"/core/card-types/{by_name['ContractMKC']['id']}", auth=AUTH)
    assert single.status_code == 200 and single.json()["caption"] == "Договорной документ"
    assert client.get(f"/core/card-types/{uuid4()}", auth=AUTH).status_code == 404


def test_ref_type_view_with_paging(client: TestClient, store: CardStore) -> None:
    full = client.post(f"/core/views/{SETTINGS.ref_type_view}/get-data", json={"parameters": []}, auth=AUTH)
    assert full.status_code == 200
    body = full.json()
    assert body["columns"] == SETTINGS.ref_type_columns and body["row_count"] == 0
    assert [row[1] for row in body["rows"]] == [item.name for item in store.ref_types]
    assert ["в отмену", "отменено"] in [row[1:] for row in body["rows"]]

    page = client.post(
        f"/core/views/{SETTINGS.ref_type_view}/get-data",
        json={"page_offset": 2, "page_limit": 2, "calculate_row_counting": True},
        auth=AUTH,
    )
    assert page.json()["row_count"] == len(store.ref_types) and page.json()["rows"] == body["rows"][2:4]
    assert client.post("/core/views/Departments/get-data", json={}, auth=AUTH).status_code == 404


def test_invalid_card_is_skipped_with_log(
    contract: CardServiceContract, corpus_dir: Path, caplog: pytest.LogCaptureFixture
) -> None:
    broken_dir = corpus_dir.parent / "broken"
    generate_corpus(broken_dir)
    card_path = export_dir(broken_dir) / CARDS_DIR / f"{ORDER_144}.json"
    card_path.write_text(json.dumps({"id": str(ORDER_144), "sections": "не словарь"}), encoding="utf-8")
    with caplog.at_level(logging.ERROR, logger="mocks.card_service.store"):
        store = load_store(broken_dir, contract)
    assert store.invalid_cards == 1 and ORDER_144 not in store.cards and len(store.cards) == 12
    assert "CardData" in caplog.text
