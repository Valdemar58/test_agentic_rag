"""Общее для интеграционных тестов: свежий PostgreSQL в testcontainers (без Docker — скип)."""

from __future__ import annotations

from collections.abc import Callable, Iterator

import pytest
from alembic import command
from alembic.config import Config

from common.config import ROOT

POSTGRES_IMAGE = "postgres:17.11-alpine3.23"


def fresh_postgres_url() -> Iterator[str]:
    """Поднимает контейнер PostgreSQL и отдаёт URL с драйвером asyncpg; останавливает по выходу."""
    try:
        from testcontainers.community.postgres import PostgresContainer
    except ImportError:  # pragma: no cover
        pytest.skip("testcontainers не установлен")
    try:
        container = PostgresContainer(POSTGRES_IMAGE, driver="asyncpg")
        container.start()
    except Exception as exc:  # noqa: BLE001 — нет Docker или образа: скип, а не падение
        pytest.skip(f"Docker недоступен для testcontainers: {exc}")
    try:
        yield container.get_connection_url()
    finally:
        container.stop()


def alembic_config(url: str) -> Config:
    config = Config(str(ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(ROOT / "alembic"))
    config.set_main_option("sqlalchemy.url", url)
    config.attributes["configure_logger"] = False
    return config


@pytest.fixture(scope="module")
def fresh_database_url() -> Iterator[str]:
    """Чистый контейнер без схемы — для теста миграций."""
    yield from fresh_postgres_url()


@pytest.fixture(scope="module")
def migrated_database_url() -> Iterator[str]:
    """Чистый контейнер с применёнными миграциями — для тестов, которым нужна схема."""
    for url in fresh_postgres_url():
        command.upgrade(alembic_config(url), "head")
        yield url


@pytest.fixture
def make_alembic_config() -> Callable[[str], Config]:
    return alembic_config
