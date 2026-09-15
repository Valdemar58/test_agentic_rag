"""Alembic async env: URL из настроек окружения (APP_DB_*), метаданные — из src/db/models.py."""

from __future__ import annotations

import asyncio
from logging.config import fileConfig

from alembic import context
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

from common.settings import load_settings
from db import models  # noqa: F401 регистрация ORM-моделей в metadata
from db.base import Base

config = context.config

# Приоритет: -x url=… → sqlalchemy.url в ini/программной конфигурации → настройки окружения
url_override = context.get_x_argument(as_dictionary=True).get("url")
if url_override:
    config.set_main_option("sqlalchemy.url", url_override)
elif not config.get_main_option("sqlalchemy.url"):
    config.set_main_option("sqlalchemy.url", load_settings().database_url)

# Логирование из ini только при запуске из CLI: программный вызов (тесты, приложение)
# выставляет attributes["configure_logger"] = False и сохраняет свои логгеры.
if config.config_file_name is not None and config.attributes.get("configure_logger", True):
    fileConfig(config.config_file_name, disable_existing_loggers=False)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    context.configure(
        url=config.get_main_option("sqlalchemy.url"),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata, compare_type=True)
    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online() -> None:
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}), prefix="sqlalchemy."
    )
    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())
