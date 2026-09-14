"""Командная строка экспорт-скрипта.

  tessa-export run --config config.yaml        экспорт из Тессы по конфигу
  tessa-export self-test [--output DIR]        проверка контейнера на синтетических данных
  tessa-export version

Коды выхода: 0 — успех и сет пригоден; 3 — экспорт выполнен, но валидация дала FAIL
(см. validation_report.md); 2 — ошибка конфигурации/окружения/учётных данных; 1 — сбой во время
работы (подробности в tessa_export.log).
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Callable
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from uuid import UUID

import structlog

from tessa_export.config import ConfigError, ExportConfig, load_config, load_seed
from tessa_export.external import ExternalCodeError
from tessa_export.fake import build_demo_scenario
from tessa_export.models import CardAccessError, GatewayConnectionError, GatewayError, TessaGateway
from tessa_export.runner import ExportError, RunSummary, run_export
from tessa_export.storage import LOG_NAME

EXIT_OK = 0
EXIT_FAILURE = 1
EXIT_CONFIG = 2
EXIT_INVALID_SET = 3

GatewayFactory = Callable[[ExportConfig, str, str], TessaGateway]

logger = logging.getLogger("tessa_export")


def _tool_version() -> str:
    try:
        return version("tessa-export")
    except PackageNotFoundError:
        return "dev"


def setup_logging(output_dir: Path, level: str) -> Path:
    """Консоль + файл в output_dir; логи SDK (structlog) направляются в тот же файл."""
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / LOG_NAME
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
    root.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(getattr(logging, level.upper(), logging.INFO))
    console.setFormatter(logging.Formatter("%(levelname)-7s %(message)s"))
    root.addHandler(file_handler)
    root.addHandler(console)
    # Контекст SDK (module=…, url=…) рендерится в строку сообщения: передача его как extra
    # конфликтует с атрибутами LogRecord (например, module).
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.KeyValueRenderer(key_order=["event"]),
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=False,
    )
    return log_path


def _default_gateway(config: ExportConfig, username: str, password: str) -> TessaGateway:
    from tessa_export.gateway_sdk import SdkGateway

    return SdkGateway(config, username, password)


def _print_summary(summary: RunSummary) -> None:
    print()
    print(f"Документов в сете: {summary.documents}, файлов скачано: {summary.files_downloaded}")
    print(f"Отчёт валидации:   {summary.report_path}")
    print(f"Таблица отбора:    {summary.review_path}")
    print(f"Архив:             {summary.archive_path}")
    if summary.overall == "PASS":
        print("ИТОГ: сет ПРИГОДЕН. Передайте архив исполнителю.")
    else:
        print("ИТОГ: сет НЕ ПРИГОДЕН. Откройте validation_report.md, раздел «Проверки 8.3».")


def _run_and_report(
    config: ExportConfig, seed_ids: list[UUID], gateway: TessaGateway, *, synthetic: bool
) -> int:
    try:
        summary = run_export(config, seed_ids, gateway, synthetic=synthetic)
    except CardAccessError as exc:
        logger.error("Нет доступа: %s", exc)
        print(f"ОШИБКА ДОСТУПА: {exc}\nПроверьте логин/пароль в переменных окружения и права учётной записи.")
        return EXIT_CONFIG
    except GatewayConnectionError as exc:
        logger.error("Сеть: %s", exc)
        print(f"ОШИБКА СЕТИ: {exc}\nПроверьте tessa.base_url в конфиге и доступность сервера из контейнера.")
        return EXIT_CONFIG
    except (ExportError, GatewayError) as exc:
        logger.exception("Экспорт прерван")
        print(f"ЭКСПОРТ ПРЕРВАН: {exc}\nПодробности в {config.output_dir / LOG_NAME}.")
        return EXIT_FAILURE
    finally:
        gateway.close()
    _print_summary(summary)
    return EXIT_OK if summary.overall == "PASS" else EXIT_INVALID_SET


def command_run(args: argparse.Namespace, gateway_factory: GatewayFactory) -> int:
    try:
        config = load_config(Path(args.config))
        if args.output:
            config.output_dir = Path(args.output)
        seed = load_seed(config.seed_file)
        username, password = config.resolve_credentials()
    except ConfigError as exc:
        print(f"ОШИБКА КОНФИГУРАЦИИ: {exc}")
        return EXIT_CONFIG
    log_path = setup_logging(config.output_dir, config.log_level)
    logger.info("tessa-export %s, конфиг %s, лог %s", _tool_version(), args.config, log_path)
    try:
        gateway = gateway_factory(config, username, password)
    except ExternalCodeError as exc:
        logger.error("Внешний код: %s", exc)
        print(f"ОШИБКА ОКРУЖЕНИЯ: {exc}")
        return EXIT_CONFIG
    return _run_and_report(config, [card.id for card in seed], gateway, synthetic=False)


def command_self_test(args: argparse.Namespace) -> int:
    output_dir = Path(args.output)
    config = ExportConfig.model_validate(
        {
            "tessa": {"base_url": "http://selftest.invalid"},
            "external": {"tessa_sdk_path": "/nonexistent", "card_service_path": "/nonexistent"},
            "output_dir": str(output_dir),
            "archive_name": "tessa_export_selftest.zip",
        }
    )
    log_path = setup_logging(config.output_dir, config.log_level)
    logger.info("tessa-export %s: самопроверка на синтетических данных, лог %s", _tool_version(), log_path)
    gateway, seed = build_demo_scenario()
    return _run_and_report(config, seed, gateway, synthetic=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tessa-export", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="экспорт из Тессы по конфигу")
    run.add_argument("--config", required=True, help="путь к config.yaml")
    run.add_argument("--output", help="каталог результата (переопределяет output_dir из конфига)")
    self_test = subparsers.add_parser("self-test", help="проверка контейнера на синтетических данных")
    self_test.add_argument("--output", default="selftest_output", help="каталог результата самопроверки")
    subparsers.add_parser("version", help="версия инструмента")
    return parser


def main(argv: list[str] | None = None, gateway_factory: GatewayFactory = _default_gateway) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "version":
        print(_tool_version())
        return EXIT_OK
    if args.command == "self-test":
        return command_self_test(args)
    return command_run(args, gateway_factory)


if __name__ == "__main__":
    sys.exit(main())
