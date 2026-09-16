"""Логирование скриптов по секции `logging` конфига (уровень из `configs/app.yaml`)."""

from __future__ import annotations

import logging

LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
DATE_FORMAT = "%H:%M:%S"
# pypdf на каждую кривую таблицу шрифтов пишет десятки WARNING (на реальном корпусе ~2 000 строк);
# для детекции текстового слоя они не важны, поэтому библиотека говорит только об ошибках.
# httpx и клиент MCP пишут INFO на каждый HTTP-запрос — в консоли агента это шум между шагами.
QUIET_LOGGERS = {
    "pypdf": logging.ERROR,
    "httpx": logging.WARNING,
    "mcp.client.streamable_http": logging.WARNING,
}


def configure_logging(level: str) -> None:
    logging.basicConfig(level=level, format=LOG_FORMAT, datefmt=DATE_FORMAT, force=True)
    for name, quiet_level in QUIET_LOGGERS.items():
        logging.getLogger(name).setLevel(quiet_level)
