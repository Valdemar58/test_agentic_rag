"""Реестр файлов инжеста (4.1): прогон, upsert по (card_id, file_row_id), хэши, удаление."""

from __future__ import annotations

import asyncio
import uuid

import pytest

from db.session import build_engine, build_sessionmaker
from ingest.registry import FileRecordInput, FileRegistry, RunCounters

pytestmark = pytest.mark.integration

CARD = uuid.uuid4()
ROW_A, ROW_B = uuid.uuid4(), uuid.uuid4()


def _record(
    row_id: uuid.UUID,
    sha256: str,
    status: str = "indexed",
    *,
    reason: str | None = None,
    file_role: str | None = None,
    parse_route: str | None = None,
    chunk_count: int = 0,
    metadata_sha256: str | None = None,
) -> FileRecordInput:
    return FileRecordInput(
        card_id=CARD,
        file_row_id=row_id,
        relative_path=f"files/{CARD}/{row_id}.docx",
        sha256=sha256,
        size_bytes=10,
        extension="docx",
        status=status,
        reason=reason,
        file_role=file_role,
        parse_route=parse_route,
        chunk_count=chunk_count,
        metadata_sha256=metadata_sha256,
    )


async def _scenario(url: str) -> None:
    engine = build_engine(url)
    try:
        registry = FileRegistry(build_sessionmaker(engine))
        run_id = await registry.start_run("data/corpus", synthetic=True)
        run = await registry.get_run(run_id)
        assert run is not None and run.outcome == "running" and run.synthetic

        await registry.record(
            run_id,
            [
                _record(ROW_A, "a" * 64, chunk_count=3, file_role="main", parse_route="native"),
                _record(ROW_B, "b" * 64, status="error", reason="нет карточки"),
            ],
        )
        rows = await registry.load([CARD])
        assert set(rows) == {(CARD, ROW_A), (CARD, ROW_B)}
        assert rows[(CARD, ROW_A)].chunk_count == 3 and rows[(CARD, ROW_A)].last_run_id == run_id
        assert rows[(CARD, ROW_B)].status == "error" and rows[(CARD, ROW_B)].reason == "нет карточки"

        # повторная запись того же файла с новым хэшем обновляет запись, а не создаёт вторую
        second_run = await registry.start_run("data/corpus", synthetic=True)
        await registry.record(second_run, [_record(ROW_A, "c" * 64, chunk_count=5, metadata_sha256="m" * 64)])
        rows = await registry.load()
        assert len(rows) == 2
        updated = rows[(CARD, ROW_A)]
        assert updated.sha256 == "c" * 64 and updated.chunk_count == 5 and updated.last_run_id == second_run
        assert updated.metadata_sha256 == "m" * 64

        assert await registry.remove([(CARD, ROW_B), (uuid.uuid4(), uuid.uuid4())]) == 1
        assert set(await registry.load()) == {(CARD, ROW_A)}

        counters = RunCounters(files_total=2, files_indexed=1, files_failed=1, chunks_total=5)
        await registry.finish_run(second_run, "partial", counters, error_text="1 файл с ошибкой")
        finished = await registry.get_run(second_run)
        assert finished is not None and finished.outcome == "partial" and finished.finished_at is not None
        assert finished.files_failed == 1 and finished.chunks_total == 5
        with pytest.raises(ValueError, match="итог прогона"):
            await registry.finish_run(second_run, "done", counters)
        with pytest.raises(LookupError):
            await registry.finish_run(10**6, "success", counters)
    finally:
        await engine.dispose()


def test_registry_upserts_by_card_and_file(migrated_database_url: str) -> None:
    asyncio.run(_scenario(migrated_database_url))


def test_record_input_rejects_unknown_status() -> None:
    with pytest.raises(ValueError, match="статус файла"):
        _record(ROW_A, "a" * 64, status="done")
