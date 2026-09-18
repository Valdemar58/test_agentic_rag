"""Реестр файлов инжеста в PostgreSQL (FR-3, FR-9): один файл карточки — одна запись с хэшем.

Только ORM: прогон `ingest_run` и записи `indexed_file` по ключу (card_id, file_row_id).
Реестр — основа инкрементальности (задача 4.7): по sha256 решается, менялся ли файл
с прошлого прогона, а исчезнувшие из корпуса файлы удаляются вместе с чанками.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Protocol
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from db.models import FILE_STATUSES, RUN_OUTCOMES, IndexedFile, IngestRun
from db.session import session_scope

FileKey = tuple[UUID, UUID]


@dataclass(frozen=True)
class RunCounters:
    files_total: int = 0
    files_indexed: int = 0
    files_skipped: int = 0
    files_failed: int = 0
    files_unchanged: int = 0
    chunks_total: int = 0


@dataclass(frozen=True)
class FileRecordInput:
    """Что известно о файле после обработки: хэш, размер, итог (indexed/skipped/error) и причина."""

    card_id: UUID
    file_row_id: UUID
    relative_path: str
    sha256: str
    size_bytes: int
    extension: str
    status: str
    reason: str | None = None
    file_role: str | None = None
    parse_route: str | None = None
    chunk_count: int = 0
    doc_status: str | None = None
    metadata_sha256: str | None = None

    def __post_init__(self) -> None:
        if self.status not in FILE_STATUSES:
            raise ValueError(f"статус файла {self.status!r} не из {FILE_STATUSES}")


class IndexedFileLike(Protocol):
    """Что читает инкрементальный прогон из записи реестра (ORM `IndexedFile` подходит структурно)."""

    @property
    def sha256(self) -> str: ...

    @property
    def status(self) -> str: ...

    @property
    def metadata_sha256(self) -> str | None: ...

    @property
    def chunk_count(self) -> int: ...

    @property
    def file_role(self) -> str | None: ...

    @property
    def parse_route(self) -> str | None: ...

    @property
    def doc_status(self) -> str | None: ...


class Registry(Protocol):
    """Интерфейс реестра для прогона: PostgreSQL в проде, память в тестах."""

    async def last_corpus_path(self) -> str | None: ...

    async def start_run(self, corpus_path: str, *, synthetic: bool) -> int: ...

    async def finish_run(
        self, run_id: int, outcome: str, counters: RunCounters, *, error_text: str | None = None
    ) -> None: ...

    async def load(self, card_ids: Iterable[UUID] | None = None) -> Mapping[FileKey, IndexedFileLike]: ...

    async def record(self, run_id: int, items: Iterable[FileRecordInput]) -> None: ...

    async def remove(self, keys: Iterable[FileKey]) -> int: ...


class FileRegistry:
    def __init__(self, factory: async_sessionmaker[AsyncSession]) -> None:
        self._factory = factory

    async def last_corpus_path(self) -> str | None:
        """Каталог корпуса последнего прогона: по нему видно, что индекс собирали из другого архива."""
        statement = select(IngestRun.corpus_path).order_by(IngestRun.id.desc()).limit(1)
        async with session_scope(self._factory) as session:
            return (await session.execute(statement)).scalars().first()

    async def start_run(self, corpus_path: str, *, synthetic: bool) -> int:
        async with session_scope(self._factory) as session:
            run = IngestRun(corpus_path=corpus_path, synthetic=synthetic, outcome="running")
            session.add(run)
            await session.flush()
            return run.id

    async def finish_run(
        self, run_id: int, outcome: str, counters: RunCounters, *, error_text: str | None = None
    ) -> None:
        if outcome not in RUN_OUTCOMES:
            raise ValueError(f"итог прогона {outcome!r} не из {RUN_OUTCOMES}")
        async with session_scope(self._factory) as session:
            run = await session.get(IngestRun, run_id)
            if run is None:
                raise LookupError(f"прогон инжеста {run_id} не найден")
            run.outcome = outcome
            run.files_total = counters.files_total
            run.files_indexed = counters.files_indexed
            run.files_skipped = counters.files_skipped
            run.files_failed = counters.files_failed
            run.files_unchanged = counters.files_unchanged
            run.chunks_total = counters.chunks_total
            run.error_text = error_text
            run.finished_at = dt.datetime.now(dt.UTC)

    async def get_run(self, run_id: int) -> IngestRun | None:
        async with session_scope(self._factory) as session:
            return await session.get(IngestRun, run_id)

    async def load(self, card_ids: Iterable[UUID] | None = None) -> dict[FileKey, IndexedFile]:
        """Записи реестра по ключу (card_id, file_row_id); card_ids=None — весь реестр."""
        statement = select(IndexedFile)
        if card_ids is not None:
            statement = statement.where(IndexedFile.card_id.in_(list(card_ids)))
        async with session_scope(self._factory) as session:
            rows = (await session.execute(statement)).scalars().all()
        return {(row.card_id, row.file_row_id): row for row in rows}

    async def record(self, run_id: int, items: Iterable[FileRecordInput]) -> None:
        """Вставляет или обновляет записи по ключу (card_id, file_row_id) в одной транзакции."""
        records = list(items)
        if not records:
            return
        card_ids = {item.card_id for item in records}
        now = dt.datetime.now(dt.UTC)
        async with session_scope(self._factory) as session:
            statement = select(IndexedFile).where(IndexedFile.card_id.in_(list(card_ids)))
            existing = {
                (row.card_id, row.file_row_id): row
                for row in (await session.execute(statement)).scalars().all()
            }
            for item in records:
                row = existing.get((item.card_id, item.file_row_id))
                if row is None:
                    row = IndexedFile(card_id=item.card_id, file_row_id=item.file_row_id)
                    existing[(item.card_id, item.file_row_id)] = row
                    session.add(row)
                row.relative_path = item.relative_path
                row.sha256 = item.sha256
                row.size_bytes = item.size_bytes
                row.extension = item.extension
                row.file_role = item.file_role
                row.parse_route = item.parse_route
                row.status = item.status
                row.reason = item.reason
                row.chunk_count = item.chunk_count
                row.doc_status = item.doc_status
                row.metadata_sha256 = item.metadata_sha256
                row.last_run_id = run_id
                row.indexed_at = now

    async def remove(self, keys: Iterable[FileKey]) -> int:
        """Удаляет записи файлов, исчезнувших из корпуса; возвращает число удалённых."""
        wanted = set(keys)
        if not wanted:
            return 0
        removed = 0
        async with session_scope(self._factory) as session:
            statement = select(IndexedFile).where(IndexedFile.card_id.in_({key[0] for key in wanted}))
            for row in (await session.execute(statement)).scalars().all():
                if (row.card_id, row.file_row_id) in wanted:
                    await session.delete(row)
                    removed += 1
        return removed
