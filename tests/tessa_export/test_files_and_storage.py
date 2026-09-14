"""Сохранение карточек и скачивание файлов (§8.1.1, §8.3): форматы, виртуальные файлы,
безопасные имена, sha256, устойчивость к ошибкам."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from uuid import uuid4

from tessa_export.fake import FakeGateway, build_demo_scenario, make_file, make_snapshot, stable_uuid
from tessa_export.files import SKIP_ERROR, SKIP_FORMAT, SKIP_VIRTUAL, download_card_files, skip_reason
from tessa_export.models import GatewayError
from tessa_export.storage import safe_file_name, save_card, write_json

ALLOWED = {"pdf", "docx", "xlsx", "pptx", "png", "jpg", "jpeg", "tif", "tiff", "bmp", "gif"}
A = stable_uuid("card", "A")


def test_save_card_writes_card_data_and_raw(tmp_path: Path) -> None:
    snapshot = make_snapshot(uuid4(), subject="Тема с кириллицей")
    card_path, raw_path = save_card(tmp_path, snapshot)
    assert card_path == tmp_path / "cards" / f"{snapshot.card_id}.json"
    assert raw_path == tmp_path / "cards_raw" / f"{snapshot.card_id}.json"
    card = json.loads(card_path.read_text(encoding="utf-8"))
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    assert card == snapshot.card_data_json
    assert raw == snapshot.raw
    assert "Тема с кириллицей" in card_path.read_text(encoding="utf-8")


def test_write_json_serializes_uuid_and_dates(tmp_path: Path) -> None:
    from datetime import UTC, datetime

    path = tmp_path / "nested" / "x.json"
    write_json(path, {"id": uuid4(), "when": datetime(2026, 1, 1, tzinfo=UTC), "p": Path("a/b")})
    data = json.loads(path.read_text(encoding="utf-8"))
    assert len(data["id"]) == 36
    assert data["when"].startswith("2026-01-01")
    assert data["p"] in {"a/b", "a\\b"}


def test_skip_reasons() -> None:
    card_id = uuid4()
    assert skip_reason(make_file(card_id, "x.pdf"), ALLOWED) is None
    assert skip_reason(make_file(card_id, "x.PDF"), ALLOWED) is None
    assert skip_reason(make_file(card_id, "x.sig"), ALLOWED) == (SKIP_FORMAT, "sig")
    assert skip_reason(make_file(card_id, "x.doc"), ALLOWED) == (SKIP_FORMAT, "doc")
    assert skip_reason(make_file(card_id, "noext"), ALLOWED) == (SKIP_FORMAT, "без расширения")
    virtual = make_file(card_id, "Лист.html", is_virtual=True)
    assert skip_reason(virtual, ALLOWED) == (SKIP_VIRTUAL, "KrVirtualFileType")


def test_download_demo_card_files(tmp_path: Path) -> None:
    gateway, _ = build_demo_scenario()
    snapshot = gateway.get_card(A)
    records = download_card_files(gateway, A, snapshot.files, ALLOWED, tmp_path)
    by_name = {record.file.name: record for record in records}
    assert len(records) == len(snapshot.files)
    downloaded = [record for record in records if record.downloaded]
    assert {record.file.extension for record in downloaded} == {"docx", "pdf", "jpg"}
    assert by_name["Подпись.sig"].skipped_reason == SKIP_FORMAT
    assert by_name["Лист согласования.html"].skipped_reason == SKIP_VIRTUAL
    for record in downloaded:
        assert record.relative_path is not None
        stored = tmp_path / record.relative_path
        assert stored.is_file()
        assert record.sha256 == hashlib.sha256(stored.read_bytes()).hexdigest()
        assert record.size == stored.stat().st_size
        assert record.relative_path.startswith(f"files/{A}/")
    # кириллическое имя сохранено
    assert (tmp_path / "files" / str(A) / "Приказ 144 оригинал.docx").is_file()
    # виртуальный файл не скачивался
    assert all(
        row_id != by_name["Лист согласования.html"].file.row_id for _, row_id in gateway.download_calls
    )


def test_download_errors_do_not_stop_other_files(tmp_path: Path) -> None:
    gateway = FakeGateway()
    card_id = uuid4()
    good = make_file(card_id, "ok.pdf")
    bad = make_file(card_id, "bad.pdf")
    empty = make_file(card_id, "empty.pdf")
    gateway.add(make_snapshot(card_id, files=[bad, good, empty]), {"ok.pdf": b"%PDF ok", "empty.pdf": b""})
    gateway.file_errors[(card_id, bad.row_id)] = GatewayError("403")
    records = download_card_files(gateway, card_id, [bad, good, empty], ALLOWED, tmp_path)
    by_name = {record.file.name: record for record in records}
    assert by_name["bad.pdf"].skipped_reason == SKIP_ERROR and "403" in by_name["bad.pdf"].skipped_detail
    assert by_name["empty.pdf"].skipped_reason == SKIP_ERROR
    assert by_name["ok.pdf"].downloaded


def test_safe_file_name_sanitizes_and_resolves_collisions() -> None:
    used: set[str] = set()
    assert safe_file_name("../../etc/passwd.pdf", used, "id1") == "passwd.pdf"
    assert safe_file_name("C:\\temp\\Приказ: 1?.docx", used, "id2") == "Приказ_ 1_.docx"
    assert safe_file_name("", used, "id3") == "id3"
    assert safe_file_name("PASSWD.pdf", used, "id4") == "PASSWD__id4.pdf"
    long_name = "a" * 300 + ".pdf"
    assert len(safe_file_name(long_name, used, "id5")) <= 200


def test_safe_file_name_limits_utf8_bytes_not_characters() -> None:
    # реальный случай из контура: 150 символов кириллицы = ~290 байт UTF-8 > лимита ext4 в 255 байт
    used: set[str] = set()
    name = "1 б от 13.01.2025 О внесении изменений в приказы по направлению деятельности " * 3 + ".pdf"
    first = safe_file_name(name, used, "id1")
    assert first.endswith(".pdf") and first.startswith("1 б от 13.01.2025")
    assert len(first.encode("utf-8")) <= 200
    # коллизия с тем же длинным именем: суффикс добавляется, итог всё ещё в лимите файловой системы
    second = safe_file_name(name, used, "0e2a1b6b-4d4d-4c0f-9d0e-2a3a5e6b7c8d")
    assert second != first and second.endswith(".pdf")
    assert len(second.encode("utf-8")) <= 255
    # обрезка не ломает многобайтовый символ на границе
    assert "�" not in first


def test_disk_write_error_does_not_stop_export(tmp_path: Path) -> None:
    gateway = FakeGateway()
    card_id = uuid4()
    file = make_file(card_id, "ok.pdf")
    gateway.add(make_snapshot(card_id, files=[file]), {"ok.pdf": b"%PDF ok"})
    # на месте каталога карточки лежит файл → mkdir/запись дают OSError
    (tmp_path / "files").mkdir()
    (tmp_path / "files" / str(card_id)).write_bytes(b"not a directory")
    records = download_card_files(gateway, card_id, [file], ALLOWED, tmp_path)
    assert records[0].skipped_reason == SKIP_ERROR
    assert "не записан на диск" in records[0].skipped_detail
    assert not records[0].downloaded


def test_name_collision_within_card_keeps_both_files(tmp_path: Path) -> None:
    gateway = FakeGateway()
    card_id = uuid4()
    first = make_file(card_id, "Документ.pdf")
    second = make_file(card_id, "документ.pdf")
    gateway.add(make_snapshot(card_id, files=[first, second]))
    gateway.contents[(card_id, first.row_id)] = b"%PDF one"
    gateway.contents[(card_id, second.row_id)] = b"%PDF two"
    records = download_card_files(gateway, card_id, [first, second], ALLOWED, tmp_path)
    paths = {record.relative_path for record in records}
    assert len(paths) == 2
    assert all(path is not None and (tmp_path / path).is_file() for path in paths)
