"""Smoke-проверка стенда (этап 3 §11).

  uv run python scripts/smoke.py             все проверки: БД и миграции, Qdrant, LLM
  uv run python scripts/smoke.py --skip-llm  без профиля runtime (только база)

Проверки LLM: ответ по-русски, вызов инструмента hybrid_search, время первого токена (NFR-2).
Код выхода: 0 — все проверки прошли; 1 — есть проваленные или недоступные сервисы; 2 — конфиг.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from common.config import ConfigError, load_app_config
from common.settings import load_settings
from common.smoke import run_all

EXIT_OK = 0
EXIT_FAILURE = 1
EXIT_CONFIG = 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="smoke", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, help="путь к app.yaml (по умолчанию configs/app.yaml)")
    parser.add_argument(
        "--skip-llm", action="store_true", help="не проверять LLM (профиль runtime не поднят)"
    )
    args = parser.parse_args(argv)
    try:
        config = load_app_config(args.config)
    except ConfigError as exc:
        print(f"ОШИБКА КОНФИГУРАЦИИ: {exc}")
        return EXIT_CONFIG
    settings = load_settings()
    print(f"LLM: {settings.resolve_llm_base_url(config)}  Qdrant: {settings.resolve_qdrant_url()}")
    results = run_all(settings, config, skip_llm=args.skip_llm)
    for result in results:
        mark = "OK  " if result.ok else "FAIL"
        print(f"{mark} {result.name:<26} {result.seconds:6.2f} с  {result.detail}")
    failed = [result.name for result in results if not result.ok]
    if failed:
        print(f"ИТОГ: провалено {len(failed)} из {len(results)}: {', '.join(failed)}")
        return EXIT_FAILURE
    print(f"ИТОГ: все {len(results)} проверок прошли.")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
