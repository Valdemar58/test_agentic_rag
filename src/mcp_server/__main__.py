"""MCP-сервер инструментов агента (FR-2, AC-2.3).

  uv run python -m mcp_server [--host HOST] [--port PORT] [--path PATH] [--config PATH] [--no-warm-up]

Поднимает FastMCP (streamable-http) с инструментами hybrid_search, get_document_card,
get_related_documents, get_document_content, glossary_lookup. Зависимости: Qdrant с индексом,
веса bge-m3 и bge-reranker-v2-m3 в models/ (CPU), сервис карточек по CARD_SERVICE_URL.
Код выхода: 0 — штатная остановка; 2 — конфиг.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Literal

from common.config import ConfigError, load_app_config
from common.logs import configure_logging
from common.settings import load_settings
from mcp_server.server import build_server, build_services

EXIT_OK = 0
EXIT_CONFIG = 2
TRANSPORT: Literal["http"] = "http"  # streamable-http в терминах FastMCP 3
logger = logging.getLogger("mcp_server")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="mcp_server", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, help="путь к app.yaml (по умолчанию configs/app.yaml)")
    parser.add_argument("--host", help="адрес (по умолчанию mcp.host)")
    parser.add_argument("--port", type=int, help="порт (по умолчанию mcp.port)")
    parser.add_argument("--path", help="путь endpoint (по умолчанию mcp.path)")
    parser.add_argument("--no-warm-up", action="store_true", help="не загружать модели до первого запроса")
    args = parser.parse_args(argv)
    try:
        config = load_app_config(args.config)
    except ConfigError as exc:
        print(f"ОШИБКА КОНФИГУРАЦИИ: {exc}")
        return EXIT_CONFIG
    configure_logging(config.logging.level)
    settings = load_settings()
    services = build_services(config, settings)
    if not args.no_warm_up:
        logger.info("Прогрев моделей (bge-m3 и reranker на CPU)…")
        services.warm_up()
    server = build_server(config, services)
    server.run(
        transport=TRANSPORT,
        host=args.host or config.mcp.host,
        port=args.port or config.mcp.port,
        path=args.path or config.mcp.path,
        show_banner=False,
    )
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
