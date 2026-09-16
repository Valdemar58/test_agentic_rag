"""UI Chainlit (этап 7).

  uv run python -m ui [--host HOST] [--port PORT] [--headless] [--migrate] [--config PATH]

Поднимает Chainlit с приложением `ui/app.py`; хост и порт — из секции `ui` конфига. `--migrate`
применяет миграции Alembic к прикладной БД перед стартом (для контейнера: чистая БД → рабочее
состояние без ручных шагов, FR-9). Нужны PostgreSQL, MCP-сервер и vLLM профиля runtime.
Код выхода: 0 — штатная остановка; 2 — конфиг.
"""

from __future__ import annotations

import argparse
import logging
import os
import secrets
import sys
from pathlib import Path

from alembic import command
from alembic.config import Config

from common.config import ROOT, ConfigError, load_app_config
from common.logs import configure_logging
from common.settings import load_settings

EXIT_OK = 0
EXIT_CONFIG = 2
APP_PATH = Path(__file__).with_name("app.py")
AUTH_SECRET_ENV = "CHAINLIT_AUTH_SECRET"
logger = logging.getLogger("ui")


def migrate(database_url: str) -> None:
    config = Config(str(ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(ROOT / "alembic"))
    config.set_main_option("sqlalchemy.url", database_url)
    config.attributes["configure_logger"] = False
    command.upgrade(config, "head")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ui", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, help="путь к app.yaml (по умолчанию configs/app.yaml)")
    parser.add_argument("--host", help="адрес (по умолчанию ui.host)")
    parser.add_argument("--port", type=int, help="порт (по умолчанию ui.port)")
    parser.add_argument("--headless", action="store_true", help="не открывать браузер")
    parser.add_argument("--migrate", action="store_true", help="применить миграции Alembic перед стартом")
    args = parser.parse_args(argv)
    try:
        config = load_app_config(args.config)
    except ConfigError as exc:
        print(f"ОШИБКА КОНФИГУРАЦИИ: {exc}")
        return EXIT_CONFIG
    configure_logging(config.logging.level)
    if args.config:
        os.environ["APP_CONFIG_PATH"] = str(args.config)
    if args.migrate:
        logger.info("Применяю миграции прикладной БД")
        migrate(load_settings().database_url)
    # Chainlit читает корень приложения (.chainlit/config.toml, chainlit.md) и адрес из окружения
    os.environ.setdefault("CHAINLIT_APP_ROOT", str(ROOT))
    os.environ["CHAINLIT_HOST"] = args.host or config.ui.host
    os.environ["CHAINLIT_PORT"] = str(args.port or config.ui.port)
    if not os.environ.get(AUTH_SECRET_ENV):
        os.environ[AUTH_SECRET_ENV] = secrets.token_urlsafe(48)
        logger.warning(
            "%s не задан: сгенерирован на этот запуск, после перезапуска потребуется войти заново",
            AUTH_SECRET_ENV,
        )
    from chainlit.cli import run_chainlit
    from chainlit.config import config as chainlit_config

    chainlit_config.run.headless = args.headless
    run_chainlit(str(APP_PATH))
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
