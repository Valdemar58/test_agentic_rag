"""Разовая загрузка весов моделей по pinned-ревизиям из configs/app.yaml (NFR-3, NFR-5).

  uv run python scripts/download_models.py                 скачать всё, чего нет
  uv run python scripts/download_models.py --only qwen dots только выбранные
      (ключи: qwen, dots, embedding, reranker, docling_layout, docling_tables)
  uv run python scripts/download_models.py --check          ничего не качать, показать состояние

Повторный запуск ничего не качает: у каждой модели в `models/<имя>/.revision` записан commit HF.
Рантайм и инжест читают веса из этого каталога офлайн (HF_HUB_OFFLINE=1 в docker-compose.yml).
Коды выхода: 0 — все модели на месте; 1 — загрузка не удалась или при --check чего-то нет;
2 — ошибка конфигурации.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from common.config import MODEL_SOURCE_KEYS, ConfigError, ModelSource, load_app_config
from common.model_store import ModelStatus, ensure_model, model_status

EXIT_OK = 0
EXIT_FAILURE = 1
EXIT_CONFIG = 2
MODEL_KEYS = MODEL_SOURCE_KEYS


def _format(status: ModelStatus, note: str) -> str:
    return (
        f"{status.name:<22} {status.repo_id:<30} {status.revision[:12]}  {status.size_gib:6.2f} ГиБ  {note}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="download_models", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, help="путь к app.yaml (по умолчанию configs/app.yaml)")
    parser.add_argument("--only", nargs="+", choices=MODEL_KEYS, help="какие модели обработать")
    parser.add_argument("--check", action="store_true", help="только показать состояние, не качать")
    args = parser.parse_args(argv)
    try:
        config = load_app_config(args.config)
    except ConfigError as exc:
        print(f"ОШИБКА КОНФИГУРАЦИИ: {exc}")
        return EXIT_CONFIG

    models_dir = config.models.dir_absolute
    selected: list[ModelSource] = [getattr(config.models, key) for key in (args.only or MODEL_KEYS)]
    print(f"Каталог весов: {models_dir}")
    missing = 0
    for source in selected:
        if args.check:
            status = model_status(source, models_dir)
            missing += not status.present
            print(_format(status, "на месте" if status.present else "ОТСУТСТВУЕТ"))
            continue
        print(f"{source.local_name}: проверка…", flush=True)
        try:
            status, downloaded = ensure_model(source, models_dir)
        except Exception as exc:  # noqa: BLE001 — любая сетевая ошибка HF показывается пользователю
            print(_format(model_status(source, models_dir), f"ОШИБКА ЗАГРУЗКИ: {exc}"))
            missing += 1
            continue
        print(_format(status, "скачано" if downloaded else "уже на месте, пропущено"))
    if missing:
        print(f"Не хватает моделей: {missing}. Повторите запуск при доступной сети.")
        return EXIT_FAILURE
    print("Все модели на месте; рантайм может работать офлайн.")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
