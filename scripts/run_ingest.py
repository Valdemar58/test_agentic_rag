"""Инжест корпуса (этап 4).

  uv run python scripts/run_ingest.py inspect [--corpus PATH] [--files]
      разбор архива экспорта без индексации: состав, хэши, файлы без карточки, роли файлов
      по правилу О5 и маршрут разбора (нативный Docling / dots.mocr); --files — построчно

Подкоманда `run` (оркестрация профилей GPU, разбор, эмбеддинги, Qdrant) добавляется в задаче 4.8.
Код выхода: 0 — все файлы допущены; 1 — часть файлов не допущена (перечислены с причиной);
2 — конфиг или архив не читается.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

from common.config import AppConfig, ConfigError, load_app_config
from common.logs import configure_logging
from ingest.corpus import CorpusError, load_corpus
from ingest.files import plan_corpus_files
from ingest.router import choose_route

EXIT_OK = 0
EXIT_ISSUES = 1
EXIT_CONFIG = 2


def inspect_corpus(root: Path, config: AppConfig, *, show_files: bool) -> int:
    try:
        corpus = load_corpus(root)
    except CorpusError as exc:
        print(f"ОШИБКА АРХИВА: {exc}")
        return EXIT_CONFIG
    for line in corpus.summary_lines():
        print(line)

    plans = plan_corpus_files(corpus, config.ingest)
    roles: Counter[str] = Counter()
    skips: Counter[str] = Counter()
    routes: Counter[str] = Counter()
    for card_id, card_plans in plans.items():
        for plan in card_plans:
            if not plan.indexed:
                skips[(plan.skip_reason or "").split(" «")[0].split(" при ")[0]] += 1
                if show_files:
                    print(f"  пропуск  {card_id} {plan.file.name} — {plan.skip_reason}")
                continue
            decision = choose_route(plan.file, config.ingest)
            roles[plan.role or ""] += 1
            routes[decision.route] += 1
            if show_files:
                print(f"  {decision.route:<6} {plan.role:<10} {card_id} {plan.file.name} — {decision.reason}")
    indexed = sum(roles.values())
    by_role = f"main {roles['main']}, appendix {roles['appendix']}, supplement {roles['supplement']}"
    print(f"К индексации: {indexed} файлов ({by_role})")
    print(f"Маршруты: нативный Docling {routes['native']}, dots.mocr {routes['vlm']}")
    print(f"Пропущено правилом файлов: {sum(skips.values())}")
    for reason, count in skips.most_common():
        print(f"  {count:>4}  {reason}")

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
    inspect_parser.add_argument("--files", action="store_true", help="показать решение по каждому файлу")
    args = parser.parse_args(argv)
    try:
        config = load_app_config(args.config)
    except ConfigError as exc:
        print(f"ОШИБКА КОНФИГУРАЦИИ: {exc}")
        return EXIT_CONFIG
    configure_logging(config.logging.level)
    if args.command == "inspect":
        return inspect_corpus(args.corpus or config.paths.corpus_dir_absolute, config, show_files=args.files)
    parser.error(f"неизвестная команда {args.command}")
    return EXIT_CONFIG


if __name__ == "__main__":
    sys.exit(main())
