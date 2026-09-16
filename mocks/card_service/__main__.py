"""Мок сервиса карточек (§9 ТЗ).

  uv run python -m mocks.card_service [--corpus PATH] [--host HOST] [--port PORT] [--config PATH]

Поднимает FastAPI с маршрутами и схемами реального сервиса карточек на данных архива экспорта
(по умолчанию paths.corpus_dir из конфига). Нужны TESSA_SDK_PATH и CARD_SERVICE_PATH (.env).
Код выхода: 0 — штатная остановка; 2 — конфиг, внешний код или архив.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import uvicorn

from common.config import ConfigError, load_app_config
from common.logs import configure_logging
from contracts.card_service import ContractsUnavailableError, load_contract
from ingest.corpus import CorpusError
from mocks.card_service.app import create_app, load_store

EXIT_OK = 0
EXIT_CONFIG = 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="mocks.card_service", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, help="путь к app.yaml (по умолчанию configs/app.yaml)")
    parser.add_argument("--corpus", type=Path, help="каталог архива экспорта (по умолчанию paths.corpus_dir)")
    parser.add_argument("--host", help="адрес (по умолчанию mock_card_service.host)")
    parser.add_argument("--port", type=int, help="порт (по умолчанию mock_card_service.port)")
    args = parser.parse_args(argv)
    try:
        config = load_app_config(args.config)
    except ConfigError as exc:
        print(f"ОШИБКА КОНФИГУРАЦИИ: {exc}")
        return EXIT_CONFIG
    configure_logging(config.logging.level)
    try:
        contract = load_contract()
    except ContractsUnavailableError as exc:
        print(f"ОШИБКА: схемы сервиса карточек недоступны ({exc}); задайте пути в .env")
        return EXIT_CONFIG
    try:
        store = load_store(args.corpus or config.paths.corpus_dir_absolute, contract)
    except CorpusError as exc:
        print(f"ОШИБКА АРХИВА: {exc}")
        return EXIT_CONFIG
    for line in store.summary_lines():
        print(line)
    settings = config.mock_card_service
    app = create_app(store, settings)
    uvicorn.run(app, host=args.host or settings.host, port=args.port or settings.port, log_level="info")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
