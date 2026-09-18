"""CLI и оркестрация экспорта: самопроверка, коды выхода, понятные сообщения об ошибках, архив."""

from __future__ import annotations

import json
import logging
import zipfile
from collections.abc import Iterator
from pathlib import Path

import pytest
import structlog

from tessa_export import cli
from tessa_export.config import ExportConfig
from tessa_export.fake import FakeGateway, build_demo_scenario, make_file, make_snapshot, stable_uuid
from tessa_export.models import CardAccessError, GatewayConnectionError, TessaViewGateway

A = stable_uuid("card", "A")


@pytest.fixture(autouse=True)
def _reset_logging() -> Iterator[None]:
    yield
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()
    structlog.reset_defaults()


def _write_config(tmp_path: Path, seed_ids: list[str] | None = None) -> Path:
    seed_path = tmp_path / "seed.yaml"
    seed_path.write_text("\n".join(f"- {item}" for item in seed_ids or [str(A)]) + "\n", encoding="utf-8")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "tessa:\n  base_url: https://tessa.local\n"
        "external:\n  tessa_sdk_path: /nonexistent/sdk\n  card_service_path: /nonexistent/svc\n"
        f"seed_file: seed.yaml\noutput_dir: {tmp_path.as_posix()}/out\n",
        encoding="utf-8",
    )
    return config_path


def test_self_test_produces_full_result(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = cli.main(["self-test", "--output", str(tmp_path / "selftest")])
    assert code == cli.EXIT_OK
    out = capsys.readouterr().out
    assert "ИТОГ: сет ПРИГОДЕН" in out
    export = tmp_path / "selftest" / "export"
    manifest = json.loads((export / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["synthetic"] is True
    assert manifest["stats"]["documents"] >= 5
    assert (export / "links_graph.json").is_file()
    report = (export / "validation_report.md").read_text(encoding="utf-8")
    assert "Данные синтетические" in report and "ИТОГ: ПРИГОДЕН" in report
    assert (tmp_path / "selftest" / "tessa_export.log").is_file()
    with zipfile.ZipFile(tmp_path / "selftest" / "tessa_export_selftest.zip") as archive:
        names = archive.namelist()
    assert "manifest.json" in names and "links_graph.json" in names and "validation_report.md" in names
    assert any(name.startswith("cards/") for name in names)
    assert any(name.startswith("cards_raw/") for name in names)
    assert any(name.startswith(f"files/{A}/") for name in names)
    assert all("\\" not in name for name in names)


def test_run_with_fake_gateway_and_rerun_cleans_previous(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TESSA_USERNAME", "DOMAIN\\user")
    monkeypatch.setenv("TESSA_PASSWORD", "secret")
    config_path = _write_config(tmp_path)
    gateway, _ = build_demo_scenario()
    seen: list[tuple[str, str]] = []

    def factory(config: ExportConfig, username: str, password: str) -> TessaViewGateway:
        seen.append((username, password))
        return gateway

    code = cli.main(["run", "--config", str(config_path)], gateway_factory=factory)
    assert code == cli.EXIT_OK
    assert seen == [("DOMAIN\\user", "secret")]
    assert gateway.closed
    export = tmp_path / "out" / "export"
    stale = export / "stale.txt"
    stale.write_text("old", encoding="utf-8")
    gateway2, _ = build_demo_scenario()
    code = cli.main(["run", "--config", str(config_path)], gateway_factory=lambda *_: gateway2)
    assert code == cli.EXIT_OK
    assert not stale.exists()
    manifest = json.loads((export / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["synthetic"] is False


def test_invalid_set_gives_exit_3(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("TESSA_USERNAME", "u")
    monkeypatch.setenv("TESSA_PASSWORD", "p")
    config_path = _write_config(tmp_path)
    gateway = FakeGateway()
    gateway.add(make_snapshot(A, files=[make_file(A, "битый.pdf")]), {"битый.pdf": b"not a pdf"})
    code = cli.main(["run", "--config", str(config_path)], gateway_factory=lambda *_: gateway)
    assert code == cli.EXIT_INVALID_SET
    assert "НЕ ПРИГОДЕН" in capsys.readouterr().out


def test_config_errors_are_readable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    code = cli.main(["run", "--config", str(tmp_path / "missing.yaml")])
    assert code == cli.EXIT_CONFIG
    assert "не найден" in capsys.readouterr().out
    config_path = _write_config(tmp_path)
    monkeypatch.delenv("TESSA_USERNAME", raising=False)
    monkeypatch.delenv("TESSA_PASSWORD", raising=False)
    code = cli.main(["run", "--config", str(config_path)])
    assert code == cli.EXIT_CONFIG
    assert "TESSA_USERNAME" in capsys.readouterr().out


def test_external_code_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("TESSA_USERNAME", "u")
    monkeypatch.setenv("TESSA_PASSWORD", "p")
    config_path = _write_config(tmp_path)
    code = cli.main(["run", "--config", str(config_path)])  # реальный шлюз, путей к SDK нет
    assert code == cli.EXIT_CONFIG
    out = capsys.readouterr().out
    assert "tessa_client" in out and "external.tessa_sdk_path" in out


def test_access_and_network_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("TESSA_USERNAME", "u")
    monkeypatch.setenv("TESSA_PASSWORD", "p")
    config_path = _write_config(tmp_path)
    denied = FakeGateway()
    denied.connection_error = CardAccessError("неверный пароль")
    assert (
        cli.main(["run", "--config", str(config_path)], gateway_factory=lambda *_: denied) == cli.EXIT_CONFIG
    )
    assert "ОШИБКА ДОСТУПА" in capsys.readouterr().out
    offline = FakeGateway()
    offline.connection_error = GatewayConnectionError("нет маршрута")
    assert (
        cli.main(["run", "--config", str(config_path)], gateway_factory=lambda *_: offline) == cli.EXIT_CONFIG
    )
    assert "ОШИБКА СЕТИ" in capsys.readouterr().out


def test_no_cards_at_all_is_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("TESSA_USERNAME", "u")
    monkeypatch.setenv("TESSA_PASSWORD", "p")
    config_path = _write_config(tmp_path)
    empty = FakeGateway()  # seed-карточки в нём нет
    code = cli.main(["run", "--config", str(config_path)], gateway_factory=lambda *_: empty)
    assert code == cli.EXIT_FAILURE
    assert "ЭКСПОРТ ПРЕРВАН" in capsys.readouterr().out
    assert (tmp_path / "out" / "tessa_export.log").read_text(encoding="utf-8")


def test_version_command(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["version"]) == cli.EXIT_OK
    assert capsys.readouterr().out.strip()
