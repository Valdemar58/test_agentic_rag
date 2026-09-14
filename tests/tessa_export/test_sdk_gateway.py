"""Реальный шлюз на SDK Тессы против respx-заглушки сервера (нужны внешние пути)."""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any
from urllib.parse import quote
from uuid import UUID

import httpx
import pytest
import respx

from contracts.card_service import CardServiceContract
from contracts.external_paths import ExternalPaths
from tessa_export.config import ExportConfig
from tessa_export.gateway_sdk import SdkGateway
from tessa_export.models import CardAccessError, CardNotFoundError, GatewayError

pytestmark = pytest.mark.contract

BASE_URL = "https://tessa.test"
CANCELLED_STATUS_ID = "de9d3b6d-532b-4cb8-aa7b-e055e8986e48"


@pytest.fixture
def gateway(contract: CardServiceContract) -> Iterator[SdkGateway]:
    paths = ExternalPaths()
    assert paths.tessa_sdk_path and paths.card_service_path
    config = ExportConfig.model_validate(
        {
            "tessa": {"base_url": BASE_URL, "max_retries": 0},
            "external": {
                "tessa_sdk_path": str(paths.tessa_sdk_path),
                "card_service_path": str(paths.card_service_path),
            },
        }
    )
    instance = SdkGateway(config, "DOMAIN\\user", "secret")
    yield instance
    instance.close()


@pytest.fixture
def tessa(card_example_raw: dict[str, Any]) -> Iterator[respx.MockRouter]:
    with respx.mock(base_url=BASE_URL, assert_all_called=False) as router:
        router.post("/service/login", name="login").mock(
            return_value=httpx.Response(200, json='<session id="1" />')
        )
        router.post("/api/v1/cards/get", name="cards_get").mock(
            return_value=httpx.Response(200, json=card_example_raw)
        )
        yield router


def test_get_card_returns_snapshot_and_raw(
    gateway: SdkGateway, tessa: respx.MockRouter, card_example_raw: dict[str, Any]
) -> None:
    card_id = UUID(card_example_raw["Card"]["ID::uid"])
    snapshot = gateway.get_card(card_id)

    request_body = json.loads(tessa["cards_get"].calls.last.request.content)
    assert request_body["CardID::uid"] == str(card_id)
    assert request_body["GetMode::int"] == 1
    assert tessa["cards_get"].calls.last.request.headers["tessa-session"] == '<session id="1" />'

    assert snapshot.card_id == card_id
    assert snapshot.type_name == "OrderMKC"
    assert snapshot.common_text("StatusID") == CANCELLED_STATUS_ID
    assert snapshot.common_text("DepartmentName")
    assert snapshot.outgoing[0].ref_type_name == "в отмену"
    assert snapshot.outgoing[0].ref_type_reverse_name == "отменено"
    assert snapshot.incoming[0].doc_id
    assert snapshot.incoming[0].ref_type_name is None
    assert len(snapshot.files) == 5
    virtual = [item for item in snapshot.files if item.is_virtual]
    assert len(virtual) == 1 and virtual[0].type_name == "KrVirtualFileType"
    assert snapshot.raw == card_example_raw
    assert snapshot.card_data_json["type_name"] == "OrderMKC"
    assert snapshot.card_data_json["sections"]["OutgoingRefDocs"]["rows"][0]["RefTypeName"] == "в отмену"


def test_download_file_uses_sdk_and_keeps_cyrillic_name(gateway: SdkGateway, tessa: respx.MockRouter) -> None:
    card_id = UUID("3a2d502f-d44e-4e8d-8c1e-d17fa72c9c3b")
    snapshot = gateway.get_card(card_id)
    file = next(item for item in snapshot.files if item.extension == "pdf")
    name = quote(file.name)
    tessa.post("/api/v1/cards/get-file-content", name="file_content").mock(
        return_value=httpx.Response(
            200,
            content=b"%PDF-1.4 demo",
            headers={
                "content-type": "application/pdf",
                "content-disposition": f"attachment; filename=\"x.pdf\"; filename*=UTF-8''{name}",
            },
        )
    )
    downloaded = gateway.download_file(card_id, file)
    assert downloaded.content == b"%PDF-1.4 demo"
    assert downloaded.file_name == file.name
    assert downloaded.content_type == "application/pdf"
    body = json.loads(tessa["file_content"].calls.last.request.content)
    assert body["FileID::uid"] == str(file.row_id)
    assert body["VersionRowID::uid"] == str(file.version_row_id)


def test_error_mapping(gateway: SdkGateway, tessa: respx.MockRouter) -> None:
    card_id = UUID("3a2d502f-d44e-4e8d-8c1e-d17fa72c9c3b")
    tessa.post("/api/v1/cards/get").mock(return_value=httpx.Response(404, json={"Items": None}))
    with pytest.raises(CardNotFoundError):
        gateway.get_card(card_id)
    tessa.post("/api/v1/cards/get").mock(return_value=httpx.Response(403, json={"Items": None}))
    with pytest.raises(CardAccessError):
        gateway.get_card(card_id)
    tessa.post("/api/v1/cards/get").mock(
        return_value=httpx.Response(
            200,
            json={
                "Card": None,
                "ValidationResult": {
                    "Items": [
                        {
                            "Key::uid": "00000000-0000-0000-0000-000000000000",
                            "Type::int": 2,
                            "Message": "Ошибка расширения",
                        }
                    ]
                },
            },
        )
    )
    with pytest.raises(GatewayError) as exc_info:
        gateway.get_card(card_id)
    assert "Ошибка расширения" in str(exc_info.value)
    tessa.post("/api/v1/cards/get").mock(
        return_value=httpx.Response(200, json={"Card": None, "ValidationResult": {"Items": None}})
    )
    with pytest.raises(CardNotFoundError):
        gateway.get_card(card_id)
