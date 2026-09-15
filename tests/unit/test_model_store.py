"""Хранилище весов: загрузка по ревизии идемпотентна, смена ревизии очищает каталог (без сети)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from common.config import ModelSource
from common.model_store import REVISION_FILE, directory_size, ensure_model, model_status

REVISION_A = "a" * 40
REVISION_B = "b" * 40


class FakeDownloader:
    """Пишет файл весов вместо обращения к Hugging Face и запоминает вызовы."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def __call__(
        self, *, repo_id: str, revision: str, local_dir: Path, ignore_patterns: Sequence[str]
    ) -> None:
        self.calls.append(
            {"repo_id": repo_id, "revision": revision, "ignore_patterns": list(ignore_patterns)}
        )
        (local_dir / "model.safetensors").write_bytes(b"\0" * 1024)
        cache = local_dir / ".cache" / "huggingface"
        cache.mkdir(parents=True, exist_ok=True)
        (cache / "meta").write_bytes(b"\0" * 4096)


def _source(revision: str = REVISION_A, ignore: list[str] | None = None) -> ModelSource:
    return ModelSource(
        repo_id="BAAI/bge-m3", revision=revision, local_name="bge-m3", ignore_patterns=ignore or []
    )


def test_second_run_downloads_nothing(tmp_path: Path) -> None:
    downloader = FakeDownloader()
    source = _source(ignore=["onnx/*"])
    status, downloaded = ensure_model(source, tmp_path, downloader=downloader)
    assert downloaded and status.present
    assert (tmp_path / "bge-m3" / REVISION_FILE).read_text(encoding="utf-8").strip() == REVISION_A
    assert downloader.calls == [
        {"repo_id": "BAAI/bge-m3", "revision": REVISION_A, "ignore_patterns": ["onnx/*"]}
    ]
    # служебный кэш загрузчика не считается размером весов
    assert status.size_bytes == 1024 and directory_size(status.path) == 1024

    again, downloaded_again = ensure_model(source, tmp_path, downloader=downloader)
    assert not downloaded_again and again.present
    assert len(downloader.calls) == 1


def test_changed_revision_clears_directory_and_redownloads(tmp_path: Path) -> None:
    downloader = FakeDownloader()
    ensure_model(_source(REVISION_A), tmp_path, downloader=downloader)
    stale = tmp_path / "bge-m3" / "stale.bin"
    stale.write_bytes(b"old")

    status, downloaded = ensure_model(_source(REVISION_B), tmp_path, downloader=downloader)
    assert downloaded and status.present and status.revision == REVISION_B
    assert not stale.exists()
    assert len(downloader.calls) == 2


def test_interrupted_download_without_marker_is_resumed_in_place(tmp_path: Path) -> None:
    partial_dir = tmp_path / "bge-m3"
    partial_dir.mkdir()
    partial = partial_dir / "model.safetensors.incomplete"
    partial.write_bytes(b"part")
    assert not model_status(_source(), tmp_path).present

    downloader = FakeDownloader()
    status, downloaded = ensure_model(_source(), tmp_path, downloader=downloader)
    assert downloaded and status.present
    assert partial.exists()  # каталог не очищался: докачка на месте


def test_status_of_missing_model(tmp_path: Path) -> None:
    status = model_status(_source(), tmp_path)
    assert not status.present and status.size_bytes == 0 and status.size_gib == 0
