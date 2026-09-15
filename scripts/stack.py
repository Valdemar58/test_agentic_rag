"""Управление стендом: docker compose с бюджетом GPU и взаимоисключением профилей.

  uv run python scripts/stack.py up runtime [--observability] [--wait]   Qwen3-8B + база
  uv run python scripts/stack.py up ingest [--switch] [--wait]           dots.mocr + база
  uv run python scripts/stack.py up base                                 только qdrant и postgres
  uv run python scripts/stack.py down [--volumes]                        остановить всё
  uv run python scripts/stack.py status                                  состояние контейнеров
  uv run python scripts/stack.py env                                     переменные для compose
  uv run python scripts/stack.py compose -- config                      любая команда compose

Профили runtime и ingest взаимоисключены: `up` одного при работающем другом завершается
ошибкой (код 2); `--switch` сначала останавливает соперника. Доля памяти GPU считается из
бюджета 12 ГиБ в configs/app.yaml и фактического объёма карты (см. `env`).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from common.config import ConfigError, load_app_config
from common.stack import Stack, StackError

EXIT_OK = 0
EXIT_FAILURE = 1
EXIT_CONFIG = 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="stack", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, help="путь к app.yaml (по умолчанию configs/app.yaml)")
    parser.add_argument("--gpu-total-mib", type=int, help="объём памяти GPU в МиБ вместо опроса nvidia-smi")
    subparsers = parser.add_subparsers(dest="command", required=True)

    up = subparsers.add_parser("up", help="поднять базу и профиль")
    up.add_argument("target", choices=["base", "runtime", "ingest"])
    up.add_argument("--observability", action="store_true", help="дополнительно поднять Langfuse")
    up.add_argument("--switch", action="store_true", help="остановить соперничающий GPU-профиль")
    up.add_argument("--wait", action="store_true", help="ждать healthy всех сервисов")

    down = subparsers.add_parser("down", help="остановить все профили")
    down.add_argument("--volumes", action="store_true", help="удалить и данные (Qdrant, PostgreSQL)")

    subparsers.add_parser("status", help="состояние контейнеров всех профилей")
    subparsers.add_parser("env", help="напечатать переменные окружения для docker compose")
    compose = subparsers.add_parser("compose", help="произвольная команда docker compose")
    compose.add_argument("args", nargs=argparse.REMAINDER, help="аргументы после --")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_app_config(args.config)
    except ConfigError as exc:
        print(f"ОШИБКА КОНФИГУРАЦИИ: {exc}")
        return EXIT_CONFIG
    stack = Stack(config, gpu_total_mib=args.gpu_total_mib)
    try:
        if args.command == "env":
            for key, value in stack.environment().items():
                print(f"{key}={value}")
            return EXIT_OK
        if args.command == "up":
            stack.up(args.target, observability=args.observability, switch=args.switch, wait=args.wait)
            print(f"Стенд поднят: {args.target}" + (" + observability" if args.observability else ""))
            return EXIT_OK
        if args.command == "down":
            if args.volumes:
                print("ВНИМАНИЕ: удаляются данные Qdrant, PostgreSQL и Langfuse.")
            stack.down(volumes=args.volumes)
            return EXIT_OK
        if args.command == "status":
            stack.status()
            return EXIT_OK
        passthrough = [item for item in args.args if item != "--"]
        stack.passthrough(passthrough)
        return EXIT_OK
    except StackError as exc:
        print(f"ОШИБКА СТЕНДА: {exc}")
        return EXIT_CONFIG if "взаимоисключены" in str(exc) or "nvidia-smi" in str(exc) else EXIT_FAILURE


if __name__ == "__main__":
    sys.exit(main())
