"""Офлайн-фильтр архива (команда filter): тот же обход, манифест и архив без обращения к Тессе."""

from __future__ import annotations

import json
import logging
import zipfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import structlog

from tessa_export import cli
from tessa_export.fake import build_demo_scenario, stable_uuid
from tessa_export.gateway_archive import ArchiveGateway, ArchiveSource
from tessa_export.models import CardNotFoundError, GatewayError

A, B, C = (stable_uuid("card", name) for name in "ABC")


@pytest.fixture(autouse=True)
def _reset_logging(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("TESSA_USERNAME", "DOMAIN\\user")
    monkeypatch.setenv("TESSA_PASSWORD", "secret")
    yield
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()
    structlog.reset_defaults()


def _write_config(directory: Path, extra: str = "") -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    _, seed = build_demo_scenario()
    (directory / "seed.yaml").write_text(
        "cards:\n" + "".join(f"  - id: {item}\n" for item in seed), encoding="utf-8"
    )
    config = directory / "config.yaml"
    config.write_text(
        "tessa:\n  base_url: https://tessa.local\n"
        "external:\n  tessa_sdk_path: /x\n  card_service_path: /y\n"
        "seed_file: seed.yaml\noutput_dir: out\n"
        "traversal:\n  max_depth: 2\n  max_docs: 200\n" + extra,
        encoding="utf-8",
    )
    return config


def _run_online(root: Path) -> Path:
    config = _write_config(root / "online")
    gateway, _ = build_demo_scenario()
    assert cli.main(["run", "--config", str(config)], gateway_factory=lambda *_: gateway) == cli.EXIT_OK
    return root / "online" / "out"


def _manifest(output: Path) -> dict[str, Any]:
    data: dict[str, Any] = json.loads((output / "export" / "manifest.json").read_text(encoding="utf-8"))
    return data


def _comparable(manifest: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in manifest.items() if key not in {"created_at", "source"}}


def test_filter_without_rules_reproduces_online_export(tmp_path: Path) -> None:
    online = _run_online(tmp_path)
    for name, source in (("from_dir", online / "export"), ("from_zip", online / "tessa_export.zip")):
        config = _write_config(tmp_path / name)
        output = tmp_path / name / "out"
        code = cli.main(["filter", "--config", str(config), "--source", str(source), "--output", str(output)])
        assert code == cli.EXIT_OK
        filtered = _manifest(output)
        assert filtered["source"].startswith("офлайн-фильтр архива")
        # документы, файлы, хэши, связи и статистика совпадают с онлайн-экспортом один в один
        assert _comparable(filtered) == _comparable(_manifest(online))
        assert (output / "export" / "documents_review.csv").is_file()
        with zipfile.ZipFile(output / "tessa_export.zip") as archive:
            assert f"cards/{A}.json" in archive.namelist()
        report = (output / "export" / "validation_report.md").read_text(encoding="utf-8")
        assert "ИТОГ: ПРИГОДЕН" in report and "офлайн-фильтр архива" in report


def test_filter_applies_exclusion_rules_from_config(tmp_path: Path) -> None:
    online = _run_online(tmp_path)
    config = _write_config(
        tmp_path / "filtered",
        extra='exclude_rules:\n  - reason: "без положений"\n    doc_type_titles: ["Положение"]\n',
    )
    output = tmp_path / "filtered" / "out"
    code = cli.main(
        ["filter", "--config", str(config), "--source", str(online / "export"), "--output", str(output)]
    )
    assert code == cli.EXIT_OK
    manifest = _manifest(output)
    ids = {document["card_id"] for document in manifest["documents"]}
    assert str(A) in ids and str(B) in ids and str(C) not in ids
    assert [item["card_id"] for item in manifest["excluded"]] == [str(C)]
    assert manifest["excluded"][0]["reason"] == "без положений"
    assert not (output / "export" / "cards" / f"{C}.json").exists()
    assert not (output / "export" / "files" / str(C)).exists()
    with zipfile.ZipFile(output / "tessa_export.zip") as archive:
        names = set(archive.namelist())
    assert f"cards/{A}.json" in names and f"cards/{C}.json" not in names
    assert not any(name.startswith(f"files/{C}/") for name in names)
    assert len(manifest["documents"]) == len(_manifest(online)["documents"]) - 1


def test_archive_gateway_and_cli_errors(tmp_path: Path) -> None:
    online = _run_online(tmp_path)
    gateway = ArchiveGateway(ArchiveSource(online / "export"))
    assert gateway.documents_in_source == len(_manifest(online)["documents"])
    with pytest.raises(CardNotFoundError):
        gateway.get_card(stable_uuid("card", "missing"))
    gateway.close()
    with pytest.raises(GatewayError):
        ArchiveSource(tmp_path / "nope")
    # источник внутри каталога результата затирается первым же шагом — отказ до начала работы
    config = _write_config(tmp_path / "clobber")
    code = cli.main(
        [
            "filter",
            "--config",
            str(config),
            "--source",
            str(tmp_path / "clobber" / "out" / "export"),
            "--output",
            str(tmp_path / "clobber" / "out"),
        ]
    )
    assert code == cli.EXIT_CONFIG
