"""Инжест корпуса (этап 4).

  uv run python scripts/run_ingest.py inspect [--corpus PATH]   разбор архива экспорта без индексации:
                                                                состав, хэши, файлы без карточки

Подкоманда `run` (оркестрация профилей GPU, разбор, эмбеддинги, Qdrant) добавляется в задаче 4.8.
Код выхода: 0 — все файлы допущены; 1 — часть файлов не допущена (перечислены с причиной);
2 — конфиг или архив не читается.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from common.config import ConfigError, load_app_config
from common.logs import configure_logging
from ingest.corpus import CorpusError, load_corpus

EXIT_OK = 0
EXIT_ISSUES = 1
EXIT_CONFIG = 2


def inspect_corpus(root: Path) -> int:
    try:
        corpus = load_corpus(root)
    except CorpusError as exc:
        print(f"ОШИБКА АРХИВА: {exc}")
        return EXIT_CONFIG
    for line in corpus.summary_lines():
        print(line)
    if corpus.issues:
        print("Не допущены в инжест:")
        for issue in corpus.issues:
            print(f"  {issue.title}: {issue.path}" + (f" — {issue.detail}" if issue.detail else ""))
        return EXIT_ISSUES
    return EXIT_OK


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="run_ingest", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, help="путь к app.yaml (по умолчанию configs/app.yaml)")
    subparsers = parser.add_subparsers(dest="command", required=True)
    inspect_parser = subparsers.add_parser("inspect", help="разобрать архив экспорта без индексации")
    inspect_parser.add_argument(
        "--corpus", type=Path, help="каталог архива экспорта (по умолчанию paths.corpus_dir из конфига)"
    )
    args = parser.parse_args(argv)
    try:
        config = load_app_config(args.config)
    except ConfigError as exc:
        print(f"ОШИБКА КОНФИГУРАЦИИ: {exc}")
        return EXIT_CONFIG
    configure_logging(config.logging.level)
    if args.command == "inspect":
        return inspect_corpus(args.corpus or config.paths.corpus_dir_absolute)
    parser.error(f"неизвестная команда {args.command}")
    return EXIT_CONFIG


if __name__ == "__main__":
    sys.exit(main())
