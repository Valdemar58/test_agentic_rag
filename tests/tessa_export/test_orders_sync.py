"""Синхронизация приказов: накопительный экспорт, пропуск выгруженных, отбор по состоянию."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest
import structlog

from tessa_export import cli
from tessa_export.fake import FakeGateway, link, make_file, make_snapshot, stable_uuid
from tessa_export.sample_files import minimal_docx_bytes

COLUMNS = ["DocID", "DocDescription", "StateID"]
# FIRST и SECOND — действующие приказы (состояние 6), DRAFT — несогласованный проект (состояние 1)
FIRST, SECOND, DRAFT = (stable_uuid("orders", name) for name in ("first", "second", "draft"))
NAMES = {FIRST: "101", SECOND: "102", DRAFT: "103"}
REGISTERED, IN_APPROVAL = 6, 1


@pytest.fixture(autouse=True)
def _environment(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("TESSA_USERNAME", "DOMAIN\\user")
    monkeypatch.setenv("TESSA_PASSWORD", "secret")
    yield
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()
    structlog.reset_defaults()


def _config(tmp_path: Path, output: Path) -> Path:
    (tmp_path / "seed.yaml").write_text(f"- {FIRST}\n", encoding="utf-8")
    path = tmp_path / "config.yaml"
    path.write_text(
        "tessa:\n  base_url: https://tessa.local\n"
        "external:\n  tessa_sdk_path: /nonexistent\n  card_service_path: /nonexistent\n"
        f"seed_file: seed.yaml\noutput_dir: {output.as_posix()}\n"
        "orders:\n"
        "  view_alias: Orders\n"
        "  page_limit: 50\n"
        "  match:\n    DocDescription: ['приказ']\n",
        encoding="utf-8",
    )
    return path


def _order(card_id: UUID, *, state: int = REGISTERED) -> tuple[Any, dict[str, bytes]]:
    name = NAMES[card_id]
    file_name = f"Приказ {name}.docx"
    snapshot = make_snapshot(
        card_id,
        number=name,
        doc_type_title="Приказ",
        state_id=state,
        state_name="Зарегистрировано" if state == REGISTERED else "На согласовании",
        files=[make_file(card_id, file_name)],
        outgoing=[link(stable_uuid("orders", "outside"))],
    )
    return snapshot, {file_name: minimal_docx_bytes([f"Приказ № {name}", "1. Утвердить."])}


def _gateway(rows: list[dict[str, Any]], cards: list[UUID]) -> FakeGateway:
    gateway = FakeGateway()
    gateway.add_view("Orders", COLUMNS, rows, caption="Приказы")
    for card_id in cards:
        snapshot, contents = _order(card_id, state=IN_APPROVAL if card_id == DRAFT else REGISTERED)
        gateway.add(snapshot, contents)
    return gateway


def _row(card_id: UUID, state: int) -> dict[str, Any]:
    return {"DocID": str(card_id), "DocDescription": f"Приказ № {NAMES[card_id]}", "StateID": state}


def _manifest(output: Path) -> dict[str, Any]:
    data: dict[str, Any] = json.loads((output / "export" / "manifest.json").read_text(encoding="utf-8"))
    return data


def test_orders_sync_skips_exported_and_keeps_only_registered_state(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "out"
    config = _config(tmp_path, output)

    first_gateway = _gateway([_row(FIRST, REGISTERED)], [FIRST])
    assert cli.main(["orders", "--config", str(config)], gateway_factory=lambda *_: first_gateway) == 0
    manifest = _manifest(output)
    assert [document["card_id"] for document in manifest["documents"]] == [str(FIRST)]
    assert manifest["source"].startswith("приказы Тессы")
    assert manifest["traversal"]["max_depth"] == 0
    first_file = (output / "export" / "files" / str(FIRST) / "Приказ 101.docx").read_bytes()

    # второй прогон: тот же приказ + новый действующий + несогласованный проект
    rows = [_row(FIRST, REGISTERED), _row(SECOND, REGISTERED), _row(DRAFT, IN_APPROVAL)]
    second_gateway = _gateway(rows, [FIRST, SECOND, DRAFT])
    assert cli.main(["orders", "--config", str(config)], gateway_factory=lambda *_: second_gateway) == 0

    # выгруженный приказ не перезапрашивался, проект отсеян по строке представления
    assert second_gateway.get_calls == [SECOND]
    assert second_gateway.download_calls == [(SECOND, _order(SECOND)[0].files[0].row_id)]

    manifest = _manifest(output)
    assert {document["card_id"] for document in manifest["documents"]} == {str(FIRST), str(SECOND)}
    assert manifest["stats"]["documents"] == 2
    assert manifest["stats"]["files_downloaded"] == 2
    assert sorted(manifest["seed_ids"]) == sorted([str(FIRST), str(SECOND)])
    # файлы первого прогона на месте и не перезаписаны
    assert (output / "export" / "files" / str(FIRST) / "Приказ 101.docx").read_bytes() == first_file
    assert (output / "export" / "cards" / f"{FIRST}.json").is_file()
    # связь на документ вне сета осталась висячим ребром, а не потерялась
    graph = json.loads((output / "export" / "links_graph.json").read_text(encoding="utf-8"))
    assert {edge["from_id"] for edge in graph["dangling_edges"]} == {str(FIRST), str(SECOND)}

    out = capsys.readouterr().out
    assert "Уже выгружено ранее: 1" in out
    assert "отсеяно: по состоянию 1" in out


def test_orders_sync_checks_state_on_the_card_when_the_view_has_no_column(tmp_path: Path) -> None:
    """Колонки состояния в представлении может не быть — тогда решает поле карточки."""
    output = tmp_path / "out"
    config = _config(tmp_path, output)
    config.write_text(
        config.read_text(encoding="utf-8").replace(
            "  page_limit: 50\n", "  page_limit: 50\n  state_column: null\n"
        ),
        encoding="utf-8",
    )
    gateway = FakeGateway()
    gateway.add_view(
        "Orders",
        ["DocID", "DocDescription"],
        [
            {"DocID": str(DRAFT), "DocDescription": "Приказ № 103"},
            {"DocID": str(FIRST), "DocDescription": "Приказ № 101"},
        ],
    )
    for card_id in (DRAFT, FIRST):
        snapshot, contents = _order(card_id, state=IN_APPROVAL if card_id == DRAFT else REGISTERED)
        gateway.add(snapshot, contents)

    assert cli.main(["orders", "--config", str(config)], gateway_factory=lambda *_: gateway) == 0
    manifest = _manifest(output)
    assert [document["card_id"] for document in manifest["documents"]] == [str(FIRST)]
    assert manifest["excluded"][0]["card_id"] == str(DRAFT)
    assert "не из списка выгрузки: 6 Зарегистрировано" in manifest["excluded"][0]["reason"]
    assert not (output / "export" / "files" / str(DRAFT)).exists()


def test_orders_dry_run_does_not_touch_tessa_cards(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "out"
    config = _config(tmp_path, output)
    gateway = _gateway([_row(FIRST, REGISTERED), _row(SECOND, REGISTERED)], [FIRST, SECOND])

    code = cli.main(
        ["orders", "--config", str(config), "--dry-run", "--limit", "1"],
        gateway_factory=lambda *_: gateway,
    )
    assert code == 0
    assert gateway.get_calls == []
    assert not (output / "export").exists()
    out = capsys.readouterr().out
    assert "к выгрузке в этом прогоне: 1" in out
    assert str(FIRST) in out


def test_orders_without_view_alias_is_a_config_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "out"
    config = _config(tmp_path, output)
    config.write_text(
        config.read_text(encoding="utf-8").replace("  view_alias: Orders\n", ""), encoding="utf-8"
    )
    gateway = _gateway([], [])
    assert (
        cli.main(["orders", "--config", str(config)], gateway_factory=lambda *_: gateway) == cli.EXIT_CONFIG
    )
    assert "orders.view_alias" in capsys.readouterr().out
    assert gateway.closed
