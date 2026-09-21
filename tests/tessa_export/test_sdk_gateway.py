"""Реальный шлюз на SDK Тессы против respx-заглушки сервера (нужны внешние пути)."""

from __future__ import annotations

import json
import ssl
from collections.abc import Iterator
from typing import Any
from urllib.parse import quote
from uuid import UUID

import httpx
import pytest
import respx

from contracts.card_service import CardServiceContract
from contracts.external_paths import ExternalPaths
from tessa_export.config import ExportConfig, ViewParameter, ViewValue
from tessa_export.gateway_sdk import SdkGateway
from tessa_export.models import CardAccessError, CardNotFoundError, GatewayConnectionError, GatewayError

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


def test_tls_verification_is_off_by_default_and_error_gives_hint(
    gateway: SdkGateway, tessa: respx.MockRouter
) -> None:
    # verify=False доходит и до клиента логина внутри SDK, и до клиента запросов
    clients: list[Any] = [gateway._auth._login_client, gateway._session]
    for client in clients:
        assert client._transport._pool._ssl_context.verify_mode == ssl.CERT_NONE

    tessa["login"].mock(
        side_effect=httpx.ConnectError(
            "[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: "
            "unable to get local issuer certificate (_ssl.c:1032)"
        )
    )
    with pytest.raises(GatewayConnectionError) as exc_info:
        gateway.check_connection()
    message = str(exc_info.value)
    assert "CERTIFICATE_VERIFY_FAILED" in message
    assert "tessa.verify_tls: false" in message and "tessa.ca_bundle" in message

    tessa["login"].mock(side_effect=httpx.ConnectError("[Errno -2] Name or service not known"))
    with pytest.raises(GatewayConnectionError) as exc_info:
        gateway.check_connection()
    assert "verify_tls" not in str(exc_info.value)


def test_views_listing_and_paged_rows(gateway: SdkGateway, tessa: respx.MockRouter) -> None:
    tessa.get("/api/v1/views", name="views").mock(
        return_value=httpx.Response(
            200,
            json={
                "Views": [
                    {
                        "Alias": "Orders",
                        "Caption": "Приказы",
                        "Columns": [{"Alias": "DocID"}, {"Alias": "StateID"}],
                        "Parameters": [{"Alias": "DocType"}],
                    }
                ]
            },
        )
    )
    doc_id = "3a2d502f-d44e-4e8d-8c1e-d17fa72c9c3b"
    tessa.post("/api/v1/views/get-data", name="get_data").mock(
        return_value=httpx.Response(
            200,
            json={"RowCount::int": 1, "Columns": ["DocID", "StateID"], "Rows": [[doc_id, 6]]},
        )
    )

    views = gateway.list_views()
    assert [(view.alias, view.caption, view.columns, view.parameters) for view in views] == [
        ("Orders", "Приказы", ["DocID", "StateID"], ["DocType"])
    ]

    page = gateway.view_page(
        "Orders",
        [ViewParameter(name="DocType", values=[ViewValue(value="order", text="Приказ")])],
        sorting=("DocDate", True),
        page_offset=2,
        page_limit=50,
        with_count=True,
    )
    assert page.columns == ["DocID", "StateID"]
    assert page.rows == [{"DocID": doc_id, "StateID": 6}]

    assert page.row_count == 1  # запрошен подсчёт строк: сверяем полноту чтения представления

    body = json.loads(tessa["get_data"].calls.last.request.content)
    assert body["ViewAlias"] == "Orders"
    assert body["CalculateRowCounting"] is True
    assert body["SortingColumns"] == [{"Alias": "DocDate", "Descending": True}]
    names = {item["Name"]: item for item in body["Parameters"]}
    assert names["DocType"]["CriteriaValues"][0]["CriteriaName"] == "Equality"
    assert names["PageOffset"]["CriteriaValues"][0]["Values"][0]["Value::int"] == 2
    assert names["PageLimit"]["CriteriaValues"][0]["Values"][0]["Value::int"] == 50


def test_views_errors_are_mapped(gateway: SdkGateway, tessa: respx.MockRouter) -> None:
    tessa.post("/api/v1/views/get-data").mock(return_value=httpx.Response(403, json={"Items": None}))
    with pytest.raises(CardAccessError, match="представление «Orders»"):
        gateway.view_page("Orders")
    tessa.get("/api/v1/views").mock(return_value=httpx.Response(500, json={"Items": None}))
    with pytest.raises(GatewayError, match="перечень представлений"):
        gateway.list_views()


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
