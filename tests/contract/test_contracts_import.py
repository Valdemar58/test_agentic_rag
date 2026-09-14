"""Контракт карточек импортируется по внешним путям (§8.0 ТЗ, основа AC-2.4)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from contracts.card_service import CardServiceContract, card_data_from_tessa_response
from contracts.external_paths import ExternalPaths, resolve_external_paths

pytestmark = pytest.mark.contract

CANCELLED_STATUS_ID = "de9d3b6d-532b-4cb8-aa7b-e055e8986e48"


def test_missing_paths_report_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TESSA_SDK_PATH", raising=False)
    monkeypatch.delenv("CARD_SERVICE_PATH", raising=False)
    status = resolve_external_paths(ExternalPaths(_env_file=None))
    assert not status.available
    assert status.reason is not None
    assert "TESSA_SDK_PATH" in status.reason
    assert "CARD_SERVICE_PATH" in status.reason


def test_wrong_paths_report_missing_packages(tmp_path: Path) -> None:
    status = resolve_external_paths(
        ExternalPaths(tessa_sdk_path=tmp_path, card_service_path=tmp_path, _env_file=None)
    )
    assert not status.available
    assert status.reason is not None
    assert "tessa_client" in status.reason
    assert "robot_skills" in status.reason


def test_card_data_schema_loaded(contract: CardServiceContract) -> None:
    assert issubclass(contract.card_data, BaseModel)
    fields = contract.card_data.model_fields
    for name in ("id", "type_id", "type_name", "type_caption", "sections", "files", "permissions"):
        assert name in fields, f"в CardData нет поля {name}"


def test_card_data_from_anonymized_real_example(
    contract: CardServiceContract, card_example_raw: dict[str, Any]
) -> None:
    card = card_data_from_tessa_response(card_example_raw, contract)
    dumped = card.model_dump()

    assert dumped["type_name"] == "OrderMKC"
    sections = dumped["sections"]
    assert "OutgoingRefDocs" in sections
    assert "IncomingRefDocs" in sections

    common = sections["DocumentCommonInfo"]["fields"]
    assert common["StatusID"] == CANCELLED_STATUS_ID
    assert common["StatusNameStatus"]
    assert common["DepartmentName"]
    assert common["DocTypeTitle"]

    outgoing = sections["OutgoingRefDocs"]["rows"]
    assert outgoing and outgoing[0]["RefTypeName"] == "в отмену"
    incoming = sections["IncomingRefDocs"]["rows"]
    assert incoming and incoming[0]["DocID"]

    files = dumped["files"]
    assert files and any(item["is_virtual"] for item in files)
