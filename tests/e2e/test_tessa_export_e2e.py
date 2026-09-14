"""AC-EXP.1 / AC-EXP.2: seed → рекурсивный обход с циклом и связью глубже max_depth →
валидация → архив; проверяется содержимое архива, а не внутренние структуры."""

from __future__ import annotations

import json
import logging
import zipfile
from collections.abc import Iterator
from pathlib import Path

import pytest
import structlog

from tessa_export import cli
from tessa_export.fake import build_demo_scenario, stable_uuid

pytestmark = pytest.mark.e2e

A, B, C, D, E, G = (stable_uuid("card", name) for name in "ABCDEG")


@pytest.fixture(autouse=True)
def _reset_logging() -> Iterator[None]:
    yield
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()
    structlog.reset_defaults()


def test_export_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TESSA_USERNAME", "DOMAIN\\user")
    monkeypatch.setenv("TESSA_PASSWORD", "secret")
    gateway, seed = build_demo_scenario()
    (tmp_path / "seed.yaml").write_text(
        "cards:\n" + "".join(f"  - id: {item}\n" for item in seed), encoding="utf-8"
    )
    (tmp_path / "config.yaml").write_text(
        "tessa:\n  base_url: https://tessa.local\n"
        "external:\n  tessa_sdk_path: /x\n  card_service_path: /y\n"
        "seed_file: seed.yaml\noutput_dir: out\n"
        "traversal:\n  max_depth: 2\n  max_docs: 200\n",
        encoding="utf-8",
    )

    code = cli.main(["run", "--config", str(tmp_path / "config.yaml")], gateway_factory=lambda *_: gateway)
    assert code == cli.EXIT_OK

    archive_path = tmp_path / "out" / "tessa_export.zip"
    with zipfile.ZipFile(archive_path) as archive:
        names = set(archive.namelist())
        manifest = json.loads(archive.read("manifest.json"))
        graph = json.loads(archive.read("links_graph.json"))
        report = archive.read("validation_report.md").decode("utf-8")

    # структура архива (§8.1.6)
    for required in ("manifest.json", "links_graph.json", "validation_report.md"):
        assert required in names
    doc_ids = {document["card_id"] for document in manifest["documents"]}
    for card_id in doc_ids:
        assert f"cards/{card_id}.json" in names
        assert f"cards_raw/{card_id}.json" in names

    # цикл A ↔ B: оба документа ровно по одному разу, обе связи в графе
    ids = [document["card_id"] for document in manifest["documents"]]
    assert ids.count(str(A)) == 1 and ids.count(str(B)) == 1
    edges = {(edge["from_id"], edge["to_id"]): edge for edge in graph["edges"]}
    assert edges[(str(A), str(B))]["relation_type"] == "в отмену"
    assert edges[(str(B), str(A))]["relation_type"] == "в отмену"
    assert gateway.get_calls.count(A) == 1

    # связь глубже max_depth: D не в сете, но перечислен в непройденных с причиной
    assert str(D) not in doc_ids
    skipped = [item for item in manifest["skipped_links"] if item["to_card_id"] == str(D)]
    assert skipped and skipped[0]["reason"] == "max_depth"
    assert any(edge["to_id"] == str(D) for edge in graph["dangling_edges"])

    # документ с двумя путями входа — один раз в сете, оба пути в манифесте
    memo = next(document for document in manifest["documents"] if document["card_id"] == str(E))
    assert {path["via_card_id"] for path in memo["entry_paths"]} == {str(A), str(G)}

    # файлы: скачанные лежат в архиве, пропущенные помечены причиной
    for document in manifest["documents"]:
        for file in document["files"]:
            if file["downloaded"]:
                assert file["path"] in names and file["sha256"]
            else:
                assert file["skipped_reason"] in {"virtual", "format"}

    # отчёт однозначен (AC-EXP.2)
    assert "ИТОГ: ПРИГОДЕН" in report
    assert "## Проверки 8.3" in report and "## Покрытие 8.2" in report
    assert "в отмену → отменено" in report
