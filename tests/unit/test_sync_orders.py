"""Сквозной скрипт синхронизации приказов: порядок шагов, каталог, остановка после ошибки."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import sync_orders
from tessa_export.storage import EXPORT_DIR_NAME, MANIFEST_NAME


@pytest.fixture
def steps(monkeypatch: pytest.MonkeyPatch) -> dict[str, list[Any]]:
    """Подменяет оба шага: проверяем оркестрацию, а не экспорт и инжест по отдельности."""
    calls: dict[str, list[Any]] = {"export": [], "ingest": []}

    def export(config_path: Path, corpus_root: Path, args: Any) -> int:
        calls["export"].append((config_path, corpus_root, args.limit, args.dry_run))
        return 0

    def ingest(corpus_root: Path, args: Any) -> int:
        calls["ingest"].append((corpus_root, args.no_gpu_switch, args.switch_corpus))
        return 0

    monkeypatch.setattr(sync_orders, "_export", export)
    monkeypatch.setattr(sync_orders, "_ingest", ingest)
    return calls


def _corpus(root: Path) -> Path:
    export_dir = root / EXPORT_DIR_NAME
    export_dir.mkdir(parents=True, exist_ok=True)
    (export_dir / MANIFEST_NAME).write_text(json.dumps({"documents": []}), encoding="utf-8")
    return root


def _export_config(tmp_path: Path) -> Path:
    path = tmp_path / "export.yaml"
    path.write_text("tessa:\n  base_url: https://tessa.local\n", encoding="utf-8")
    return path


def test_export_then_ingest_use_the_same_directory(tmp_path: Path, steps: dict[str, list[Any]]) -> None:
    corpus = _corpus(tmp_path / "orders")
    config = _export_config(tmp_path)

    code = sync_orders.main(["--export-config", str(config), "--corpus", str(corpus), "--limit", "5"])

    assert code == 0
    assert steps["export"] == [(config, corpus, 5, False)]
    assert steps["ingest"] == [(corpus, False, False)]


def test_dry_run_stops_before_ingest(tmp_path: Path, steps: dict[str, list[Any]]) -> None:
    corpus = _corpus(tmp_path / "orders")
    code = sync_orders.main(
        ["--export-config", str(_export_config(tmp_path)), "--corpus", str(corpus), "--dry-run"]
    )
    assert code == 0
    assert steps["export"][0][3] is True
    assert steps["ingest"] == []


def test_failed_export_does_not_start_ingest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    started: list[Path] = []

    def ingest(corpus_root: Path, args: Any) -> int:
        started.append(corpus_root)
        return 0

    monkeypatch.setattr(sync_orders, "_export", lambda *_: 2)
    monkeypatch.setattr(sync_orders, "_ingest", ingest)

    code = sync_orders.main(
        ["--export-config", str(_export_config(tmp_path)), "--corpus", str(tmp_path / "orders")]
    )

    assert code == 2
    assert started == []
    assert "инжест не запускается" in capsys.readouterr().out


def test_missing_export_config_and_empty_corpus_are_reported(
    tmp_path: Path, steps: dict[str, list[Any]], capsys: pytest.CaptureFixture[str]
) -> None:
    code = sync_orders.main(["--export-config", str(tmp_path / "nope.yaml")])
    assert code == sync_orders.EXIT_CONFIG
    assert "config.example.yaml" in capsys.readouterr().out
    assert steps["export"] == []

    code = sync_orders.main(["--ingest-only", "--corpus", str(tmp_path / "empty")])
    assert code == sync_orders.EXIT_CONFIG
    assert "нет выгруженных приказов" in capsys.readouterr().out
    assert steps["ingest"] == []


def test_ingest_only_skips_export(tmp_path: Path, steps: dict[str, list[Any]]) -> None:
    corpus = _corpus(tmp_path / "orders")
    code = sync_orders.main(["--ingest-only", "--corpus", str(corpus), "--no-gpu-switch"])
    assert code == 0
    assert steps["export"] == []
    assert steps["ingest"] == [(corpus, True, False)]
