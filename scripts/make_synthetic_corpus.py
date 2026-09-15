"""Генерация синтетического корпуса в формате экспорта Тессы (§9 ТЗ). ДАННЫЕ СИНТЕТИЧЕСКИЕ.

  uv run python scripts/make_synthetic_corpus.py [--output data/corpus] [--force]

Результат: `<output>/export/` (cards, cards_raw, files, manifest.json, links_graph.json,
validation_report.md), архив synthetic_corpus.zip и README_SYNTHETIC.md. По умолчанию каталог —
`paths.corpus_dir` из configs/app.yaml. Реальный экспорт в этом каталоге не перезаписывается
без --force. Коды выхода: 0 — корпус собран и валиден; 1 — валидация FAIL; 2 — конфиг/отказ.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from common.config import ConfigError, load_app_config
from synthetic.corpus import RealCorpusPresentError, generate_corpus

EXIT_OK = 0
EXIT_FAILURE = 1
EXIT_CONFIG = 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="make_synthetic_corpus",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", type=Path, help="путь к app.yaml (по умолчанию configs/app.yaml)")
    parser.add_argument("--output", type=Path, help="каталог корпуса (по умолчанию paths.corpus_dir)")
    parser.add_argument("--force", action="store_true", help="перезаписать даже реальный экспорт")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    try:
        config = load_app_config(args.config)
    except ConfigError as exc:
        print(f"ОШИБКА КОНФИГУРАЦИИ: {exc}")
        return EXIT_CONFIG
    output_dir = args.output or config.paths.corpus_dir_absolute
    try:
        summary = generate_corpus(output_dir, force=args.force)
    except RealCorpusPresentError as exc:
        print(f"ОТКАЗ: {exc}")
        return EXIT_CONFIG
    print()
    print("ДАННЫЕ СИНТЕТИЧЕСКИЕ: организация, документы и связи вымышлены; метрики на них не считаются.")
    print(
        f"Документов: {summary.documents}, файлов: {summary.files_downloaded}, валидация: {summary.overall}"
    )
    print(f"Каталог: {summary.export_root}")
    print(f"Отчёт:   {summary.report_path}")
    return EXIT_OK if summary.overall == "PASS" else EXIT_FAILURE


if __name__ == "__main__":
    sys.exit(main())
