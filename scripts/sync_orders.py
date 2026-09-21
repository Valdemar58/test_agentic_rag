"""Синхронизация приказов Тессы: выгрузка новых приказов и инжест по тому же каталогу.

Запрос заказчика 2026-09-18 (вне ТЗ): одной командой получить в индексе все приказы, при повторном
прогоне не трогать уже обработанные, грузить только действующие — состояние «Зарегистрировано»
(`$KrStates_Doc_Registered`, 6), чтобы не попадались проекты и несогласованные документы.

  uv run python scripts/sync_orders.py [--corpus PATH] [--limit N] [--dry-run]
      шаг 1 — `tessa-export orders`: перечень приказов из представления Тессы, выгружаются только
      карточки, которых ещё нет в каталоге (сверка по manifest.json);
      шаг 2 — `run_ingest run` по этому же каталогу: разбираются только новые и изменённые файлы
      (сверка по sha256 в реестре PostgreSQL), в конце собирается глоссарий.

  --dry-run     только перечень приказов и план, без выгрузки и без инжеста
  --export-only / --ingest-only    выполнить один шаг
  --limit N     взять не больше N приказов (пробный прогон)

Перед первым запуском: в `tools/tessa_export/config.yaml` заполнить секцию `orders`
(`tessa-export views --config …` покажет, какое представление подходит), в окружении —
`TESSA_USERNAME` и `TESSA_PASSWORD`.

Проверки §8.3 (код 3 экспорта) инжест не останавливают: они оценивают качество корпуса, а не
пригодность архива к индексации — ошибки отдельных файлов инжест фиксирует сам.

Код выхода: 0 — успех; 1 — шаг завершился с ошибкой или частично; 2 — конфиг, окружение, стенд.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from common.config import AppConfig, ConfigError, load_app_config
from tessa_export.storage import EXPORT_DIR_NAME, MANIFEST_NAME

EXIT_OK = 0
EXIT_ISSUES = 1
EXIT_CONFIG = 2
# Код 3 экспорта: архив собран, но проверки §8.3 дали FAIL. Это критерии качества голден-корпуса
# (состав видов, доля сканов, открылись ли все файлы), а не пригодность архива к индексации:
# инжест всё равно разбирает файлы по одному и сам сообщает о каждой ошибке.
EXIT_EXPORT_INVALID_SET = 3

NO_EXPORT_CONFIG = (
    "не найден конфиг экспорт-скрипта: {path}\n"
    "Скопируйте tools/tessa_export/config.example.yaml в этот файл, укажите tessa.base_url,"
    " пути SDK и сервиса карточек и заполните секцию orders"
)
INVALID_SET_NOTE = (
    "Проверки §8.3 дали FAIL (подробности в validation_report.md) — это про качество корпуса,"
    " а не про пригодность к индексации. Продолжаю инжест."
)


def _export(config_path: Path, corpus_root: Path, args: argparse.Namespace) -> int:
    from tessa_export import cli as export_cli

    argv = ["orders", "--config", str(config_path), "--output", str(corpus_root)]
    if args.limit:
        argv += ["--limit", str(args.limit)]
    if args.dry_run:
        argv.append("--dry-run")
    print(f"Шаг 1/2: выгрузка приказов из Тессы в {corpus_root}")
    return export_cli.main(argv)


def _ingest(corpus_root: Path, args: argparse.Namespace) -> int:
    import run_ingest

    argv = ["run", "--corpus", str(corpus_root), "--restore-runtime"]
    if args.no_gpu_switch:
        argv.append("--no-gpu-switch")
    if args.switch_corpus:
        argv.append("--switch-corpus")
    print(f"\nШаг 2/2: инжест каталога {corpus_root}")
    return run_ingest.main(argv)


def _corpus_root(config: AppConfig, args: argparse.Namespace) -> Path:
    return Path(args.corpus) if args.corpus else config.paths.orders_dir_absolute


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="sync_orders", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, help="путь к app.yaml (по умолчанию configs/app.yaml)")
    parser.add_argument("--export-config", type=Path, help="конфиг экспорт-скрипта")
    parser.add_argument("--corpus", help="каталог синхронизации (по умолчанию paths.orders_dir)")
    parser.add_argument("--limit", type=int, help="взять не больше N приказов")
    parser.add_argument("--dry-run", action="store_true", help="только показать план выгрузки")
    parser.add_argument("--export-only", action="store_true", help="только выгрузка из Тессы")
    parser.add_argument("--ingest-only", action="store_true", help="только инжест уже выгруженного")
    parser.add_argument("--no-gpu-switch", action="store_true", help="не управлять контейнерами стенда")
    parser.add_argument(
        "--switch-corpus", action="store_true", help="разрешить смену каталога корпуса в индексе"
    )
    args = parser.parse_args(argv)
    if args.export_only and args.ingest_only:
        print("ОШИБКА: --export-only и --ingest-only взаимоисключены")
        return EXIT_CONFIG
    try:
        config = load_app_config(args.config)
    except ConfigError as exc:
        print(f"ОШИБКА КОНФИГУРАЦИИ: {exc}")
        return EXIT_CONFIG

    corpus_root = _corpus_root(config, args)
    export_config = Path(args.export_config) if args.export_config else config.paths.export_config_absolute
    if not args.ingest_only:
        if not export_config.is_file():
            print(f"ОШИБКА КОНФИГУРАЦИИ: {NO_EXPORT_CONFIG.format(path=export_config)}")
            return EXIT_CONFIG
        code = _export(export_config, corpus_root, args)
        if code == EXIT_EXPORT_INVALID_SET:
            print(f"\n{INVALID_SET_NOTE}")
        elif code != EXIT_OK:
            print("\nВыгрузка не завершилась успешно — инжест не запускается.")
            return code
    if args.dry_run or args.export_only:
        return EXIT_OK
    if not (corpus_root / EXPORT_DIR_NAME / MANIFEST_NAME).is_file():
        print(f"ОШИБКА: в {corpus_root} нет выгруженных приказов — сначала выполните шаг выгрузки")
        return EXIT_CONFIG
    return _ingest(corpus_root, args)


if __name__ == "__main__":
    sys.exit(main())
