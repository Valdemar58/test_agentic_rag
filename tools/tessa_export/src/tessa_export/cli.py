"""Командная строка экспорт-скрипта.

  tessa-export run --config config.yaml        экспорт из Тессы по конфигу
  tessa-export filter --config config.yaml --source АРХИВ --output DIR
                                               применить правила исключения и лимиты обхода из
                                               конфига к уже полученному архиву, без Тессы
  tessa-export orders --config config.yaml [--output DIR] [--limit N] [--dry-run] [--archive]
                                               синхронизация приказов (вне ТЗ, запрос заказчика):
                                               перечень берётся из представления (секция orders
                                               конфига), уже выгруженные карточки пропускаются,
                                               грузятся только состояния orders.include_state_ids
  tessa-export views --config config.yaml [--alias ALIAS] [--rows N]
                                               какие представления доступны учётной записи; с
                                               --alias — колонки, параметры и первые строки
                                               (так подбирается orders.view_alias в конфиге)
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

from tessa_export.config import (
    ConfigError,
    ExportConfig,
    TraversalSettings,
    apply_env_files,
    load_config,
    load_seed,
)
from tessa_export.external import ExternalCodeError
from tessa_export.fake import build_demo_scenario
from tessa_export.models import (
    CardAccessError,
    GatewayConnectionError,
    GatewayError,
    TessaGateway,
    TessaViewGateway,
)
from tessa_export.orders import collect_orders, state_rules
from tessa_export.runner import (
    ExportError,
    RunSummary,
    load_previous,
    run_export,
    run_incremental_export,
)
from tessa_export.storage import LOG_NAME, export_dir

EXIT_OK = 0
EXIT_FAILURE = 1
EXIT_CONFIG = 2
EXIT_INVALID_SET = 3

VIEW_VALUE_CHARS = 40

GatewayFactory = Callable[[ExportConfig, str, str], TessaViewGateway]

logger = logging.getLogger("tessa_export")


def _load_env(config_path: str) -> list[Path]:
    """`.env` в каталоге запуска и рядом с конфигом; переменные процесса важнее (Docker)."""
    return apply_env_files(Path.cwd(), Path(config_path).resolve().parent)


def _log_env_files(paths: list[Path]) -> None:
    for path in paths:
        logger.info("Переменные окружения дополнены из %s", path)


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


def _default_gateway(config: ExportConfig, username: str, password: str) -> TessaViewGateway:
    from tessa_export.gateway_sdk import SdkGateway

    return SdkGateway(config, username, password)


def _print_summary(summary: RunSummary) -> None:
    print()
    print(f"Документов в сете: {summary.documents}, файлов скачано: {summary.files_downloaded}")
    if summary.documents_added or summary.documents_known:
        print(
            f"Из них добавлено этим прогоном: {summary.documents_added}; "
            f"пропущено как уже выгруженные: {summary.documents_known}"
        )
    print(f"Отчёт валидации:   {summary.report_path}")
    print(f"Таблица отбора:    {summary.review_path}")
    print(f"Каталог:           {summary.export_root}")
    if summary.archive_path is not None:
        print(f"Архив:             {summary.archive_path}")
    if summary.overall == "PASS":
        print("ИТОГ: сет ПРИГОДЕН.")
    else:
        print("ИТОГ: сет НЕ ПРИГОДЕН. Откройте validation_report.md, раздел «Проверки 8.3».")


def _guarded_run(config: ExportConfig, gateway: TessaGateway, action: Callable[[], RunSummary]) -> int:
    """Единая обработка ошибок Тессы и печать итога для всех режимов экспорта."""
    try:
        summary = action()
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


def _run_and_report(
    config: ExportConfig,
    seed_ids: list[UUID],
    gateway: TessaGateway,
    *,
    synthetic: bool,
    source: str = "tessa",
) -> int:
    return _guarded_run(
        config, gateway, lambda: run_export(config, seed_ids, gateway, synthetic=synthetic, source=source)
    )


def command_run(args: argparse.Namespace, gateway_factory: GatewayFactory) -> int:
    try:
        config = load_config(Path(args.config))
        env_files = _load_env(args.config)
        if args.output:
            config.output_dir = Path(args.output)
        seed = load_seed(config.seed_file)
        username, password = config.resolve_credentials()
    except ConfigError as exc:
        print(f"ОШИБКА КОНФИГУРАЦИИ: {exc}")
        return EXIT_CONFIG
    log_path = setup_logging(config.output_dir, config.log_level)
    logger.info("tessa-export %s, конфиг %s, лог %s", _tool_version(), args.config, log_path)
    _log_env_files(env_files)
    try:
        gateway = gateway_factory(config, username, password)
    except ExternalCodeError as exc:
        logger.error("Внешний код: %s", exc)
        print(f"ОШИБКА ОКРУЖЕНИЯ: {exc}")
        return EXIT_CONFIG
    return _run_and_report(config, [card.id for card in seed], gateway, synthetic=False)


def command_filter(args: argparse.Namespace) -> int:
    """Офлайн-фильтр: тот же обход по правилам конфига, но карточки и файлы из готового архива."""
    from tessa_export.gateway_archive import ArchiveGateway, ArchiveSource

    try:
        config = load_config(Path(args.config))
        config.output_dir = Path(args.output)
        seed = load_seed(config.seed_file)
    except ConfigError as exc:
        print(f"ОШИБКА КОНФИГУРАЦИИ: {exc}")
        return EXIT_CONFIG
    source_path = Path(args.source)
    resolved = source_path.resolve()
    export_root = export_dir(config.output_dir).resolve()
    archive_target = (config.output_dir / config.archive_name).resolve()
    if resolved in (export_root, archive_target) or export_root in resolved.parents:
        print("ОШИБКА КОНФИГУРАЦИИ: источник лежит там, куда пишется результат; укажите другой --output")
        return EXIT_CONFIG
    log_path = setup_logging(config.output_dir, config.log_level)
    logger.info(
        "tessa-export %s: офлайн-фильтр архива %s по конфигу %s, лог %s",
        _tool_version(),
        source_path,
        args.config,
        log_path,
    )
    try:
        gateway = ArchiveGateway(ArchiveSource(source_path))
    except GatewayError as exc:
        logger.error("Источник: %s", exc)
        print(f"ОШИБКА КОНФИГУРАЦИИ: {exc}")
        return EXIT_CONFIG
    logger.info("В источнике %d документов", gateway.documents_in_source)
    return _run_and_report(
        config,
        [card.id for card in seed],
        gateway,
        synthetic=False,
        source=f"офлайн-фильтр архива {source_path.name}",
    )


def command_orders(args: argparse.Namespace, gateway_factory: GatewayFactory) -> int:
    """Синхронизация приказов: перечень из представления → экспорт только новых карточек."""
    try:
        config = load_config(Path(args.config))
        env_files = _load_env(args.config)
        if args.output:
            config.output_dir = Path(args.output)
        if args.limit:
            config.orders.max_documents = args.limit
        username, password = config.resolve_credentials()
    except ConfigError as exc:
        print(f"ОШИБКА КОНФИГУРАЦИИ: {exc}")
        return EXIT_CONFIG
    log_path = setup_logging(config.output_dir, config.log_level)
    logger.info(
        "tessa-export %s: синхронизация приказов, конфиг %s, лог %s", _tool_version(), args.config, log_path
    )
    _log_env_files(env_files)
    try:
        gateway = gateway_factory(config, username, password)
    except ExternalCodeError as exc:
        print(f"ОШИБКА ОКРУЖЕНИЯ: {exc}")
        return EXIT_CONFIG

    try:
        gateway.check_connection()
        listing = collect_orders(gateway, config.orders)
    except ConfigError as exc:
        gateway.close()
        print(f"ОШИБКА КОНФИГУРАЦИИ: {exc}")
        return EXIT_CONFIG
    except CardAccessError as exc:
        gateway.close()
        print(f"ОШИБКА ДОСТУПА: {exc}")
        return EXIT_CONFIG
    except GatewayConnectionError as exc:
        gateway.close()
        print(f"ОШИБКА СЕТИ: {exc}")
        return EXIT_CONFIG
    except GatewayError as exc:
        gateway.close()
        print(f"ОШИБКА ТЕССЫ: {exc}")
        return EXIT_FAILURE
    for line in listing.summary_lines():
        print(line)

    previous = load_previous(export_dir(config.output_dir))
    known = previous.card_ids if previous else set()
    fresh = [card_id for card_id in listing.card_ids if card_id not in known]
    print(f"Уже выгружено ранее: {len(listing.rows) - len(fresh)}; к выгрузке в этом прогоне: {len(fresh)}")
    if args.dry_run:
        known_ids = set(fresh)
        for row in listing.rows[: args.rows]:
            print(f"  [{'новый' if row.card_id in known_ids else 'есть'}] {row.card_id} {row.label}")
        gateway.close()
        return EXIT_OK
    if not fresh:
        print("Новых приказов нет: каталог экспорта уже содержит всё, что отдало представление.")
        gateway.close()
        return EXIT_OK

    orders = config.orders
    traversal = TraversalSettings(
        max_depth=orders.max_depth,
        max_docs=max(orders.max_documents, len(fresh), 1),
        directions=config.traversal.directions,
    )
    return _guarded_run(
        config,
        gateway,
        lambda: run_incremental_export(
            config,
            listing.card_ids,
            gateway,
            traversal=traversal,
            extra_rules=state_rules(orders),
            source=f"приказы Тессы, представление «{orders.view_alias}»",
            archive=args.archive,
        ),
    )


def _short(value: object) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= VIEW_VALUE_CHARS else text[: VIEW_VALUE_CHARS - 1] + "…"


def _print_views(gateway: TessaViewGateway) -> None:
    views = sorted(gateway.list_views(), key=lambda item: item.alias)
    print(f"Представлений доступно: {len(views)}")
    for view in views:
        caption = f" — {view.caption}" if view.caption else ""
        print(f"  {view.alias}{caption}")
        print(f"      колонки: {', '.join(view.columns) or '—'}")
        print(f"      параметры: {', '.join(view.parameters) or '—'}")
    print("\nНужный алиас впишите в конфиг: orders.view_alias")


def _print_view_rows(gateway: TessaViewGateway, alias: str, rows_limit: int) -> None:
    page = gateway.view_page(alias, page_offset=1, page_limit=rows_limit)
    print(f"Представление «{alias}»: колонок {len(page.columns)}, строк на первой странице {len(page.rows)}")
    print(f"  колонки: {', '.join(page.columns) or '—'}")
    for number, row in enumerate(page.rows, start=1):
        values = ", ".join(f"{name}={_short(row.get(name))}" for name in page.columns if row.get(name))
        print(f"  [{number}] {values}")


def command_views(args: argparse.Namespace, gateway_factory: GatewayFactory) -> int:
    """Разведка представлений на машине заказчика: из какого брать перечень приказов."""
    try:
        config = load_config(Path(args.config))
        env_files = _load_env(args.config)
        username, password = config.resolve_credentials()
    except ConfigError as exc:
        print(f"ОШИБКА КОНФИГУРАЦИИ: {exc}")
        return EXIT_CONFIG
    setup_logging(config.output_dir, config.log_level)
    _log_env_files(env_files)
    try:
        gateway = gateway_factory(config, username, password)
    except ExternalCodeError as exc:
        print(f"ОШИБКА ОКРУЖЕНИЯ: {exc}")
        return EXIT_CONFIG
    try:
        gateway.check_connection()
        if args.alias:
            _print_view_rows(gateway, args.alias, args.rows)
        else:
            _print_views(gateway)
    except CardAccessError as exc:
        print(f"ОШИБКА ДОСТУПА: {exc}")
        return EXIT_CONFIG
    except GatewayConnectionError as exc:
        print(f"ОШИБКА СЕТИ: {exc}")
        return EXIT_CONFIG
    except GatewayError as exc:
        print(f"ОШИБКА ТЕССЫ: {exc}")
        return EXIT_FAILURE
    finally:
        gateway.close()
    return EXIT_OK


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
    filter_cmd = subparsers.add_parser(
        "filter", help="применить правила исключения из конфига к уже полученному архиву, без Тессы"
    )
    filter_cmd.add_argument("--config", required=True, help="путь к config.yaml")
    filter_cmd.add_argument(
        "--source", required=True, help="tessa_export.zip или распакованный каталог export/"
    )
    filter_cmd.add_argument("--output", required=True, help="каталог результата (не тот, где лежит источник)")
    orders = subparsers.add_parser(
        "orders", help="синхронизация приказов: выгрузить новые, уже выгруженные пропустить"
    )
    orders.add_argument("--config", required=True, help="путь к config.yaml")
    orders.add_argument("--output", help="каталог результата (переопределяет output_dir из конфига)")
    orders.add_argument("--limit", type=int, help="взять не больше N приказов (для пробного прогона)")
    orders.add_argument("--rows", type=int, default=20, help="сколько строк показать при --dry-run")
    orders.add_argument("--dry-run", action="store_true", help="только показать план, ничего не выгружать")
    orders.add_argument("--archive", action="store_true", help="собрать zip-архив после выгрузки")
    views = subparsers.add_parser("views", help="какие представления доступны учётной записи")
    views.add_argument("--config", required=True, help="путь к config.yaml")
    views.add_argument("--alias", help="показать колонки и первые строки этого представления")
    views.add_argument("--rows", type=int, default=5, help="сколько строк показать с --alias")
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
    if args.command == "filter":
        return command_filter(args)
    if args.command == "views":
        return command_views(args, gateway_factory)
    if args.command == "orders":
        return command_orders(args, gateway_factory)
    return command_run(args, gateway_factory)


if __name__ == "__main__":
    sys.exit(main())
