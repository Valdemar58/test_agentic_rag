"""AC-9.1: чистый контейнер PostgreSQL → `alembic upgrade head` → схема есть; autogenerate пуст.

Требует Docker (testcontainers поднимает свежий PostgreSQL); без Docker тест скипается.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Callable
from pathlib import Path

import pytest
from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from sqlalchemy import Connection, inspect, select
from sqlalchemy.ext.asyncio import create_async_engine

from common.config import ROOT
from db.base import Base
from db.models import IndexedFile, IngestRun
from db.session import build_engine, build_sessionmaker, session_scope

pytestmark = pytest.mark.integration

EXPECTED_TABLES = {"app_user", "conversation", "message", "element", "feedback", "ingest_run", "indexed_file"}


async def _table_names(url: str) -> set[str]:
    engine = create_async_engine(url)
    try:
        async with engine.connect() as connection:
            return set(await connection.run_sync(lambda sync: inspect(sync).get_table_names()))
    finally:
        await engine.dispose()


async def _autogenerate_diff(url: str) -> list[object]:
    engine = create_async_engine(url)
    try:
        async with engine.connect() as connection:

            def compare(sync_connection: Connection) -> list[object]:
                context = MigrationContext.configure(sync_connection, opts={"compare_type": True})
                return list(compare_metadata(context, Base.metadata))

            return await connection.run_sync(compare)
    finally:
        await engine.dispose()


async def _roundtrip_rows(url: str) -> int:
    engine = build_engine(url)
    try:
        factory = build_sessionmaker(engine)
        async with session_scope(factory) as session:
            run = IngestRun(corpus_path="data/corpus", synthetic=True)
            session.add(run)
            await session.flush()
            session.add(
                IndexedFile(
                    card_id=uuid.uuid4(),
                    file_row_id=uuid.uuid4(),
                    relative_path="files/x/doc.docx",
                    sha256="0" * 64,
                    extension="docx",
                    status="indexed",
                    last_run_id=run.id,
                )
            )
        async with session_scope(factory) as session:
            return len((await session.execute(select(IndexedFile))).scalars().all())
    finally:
        await engine.dispose()


def test_clean_database_upgrade_head_creates_schema_and_diff_is_empty(
    fresh_database_url: str, make_alembic_config: Callable[[str], Config]
) -> None:
    database_url = fresh_database_url
    config = make_alembic_config(database_url)
    assert asyncio.run(_table_names(database_url)) == set()

    command.upgrade(config, "head")
    tables = asyncio.run(_table_names(database_url))
    assert EXPECTED_TABLES <= tables and "alembic_version" in tables

    # autogenerate не видит расхождений между моделями и схемой после миграций
    assert asyncio.run(_autogenerate_diff(database_url)) == []
    # ORM работает поверх созданной схемы
    assert asyncio.run(_roundtrip_rows(database_url)) == 1

    command.downgrade(config, "base")
    assert asyncio.run(_table_names(database_url)) == {"alembic_version"}


def test_migration_files_follow_naming_convention() -> None:
    versions = sorted(path.name for path in (ROOT / "alembic" / "versions").glob("*.py"))
    assert versions and versions[0] == "0001_initial.py"
    assert all(Path(name).stem.split("_", 1)[0].isdigit() for name in versions)
