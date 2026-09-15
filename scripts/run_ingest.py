"""Инжест корпуса (этап 4).

  uv run python scripts/run_ingest.py inspect [--corpus PATH] [--files]
      разбор архива экспорта без индексации: состав, хэши, файлы без карточки, роли файлов
      по правилу О5 и маршрут разбора (нативный Docling / dots.mocr); --files — построчно

  uv run python scripts/run_ingest.py run [--corpus PATH] [--force] [--no-gpu-switch] [--restore-runtime]
      полный инжест с оркестрацией GPU (§2 ТЗ): останавливает vllm-qwen, поднимает vllm-dots, если
      есть сканы; фаза 1 — разбор всех новых/изменённых файлов; затем vllm-dots останавливается и
      фаза 2 считает эмбеддинги bge-m3 на свободном GPU и пишет в Qdrant; реестр — PostgreSQL.
      --force переразбирает всё; --no-gpu-switch не трогает контейнеры (профили поднимает оператор);
      --restore-runtime в конце поднимает профиль runtime обратно.

Код выхода: 0 — успех; 1 — часть файлов с ошибкой или не допущена; 2 — конфиг, архив или стенд.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from collections import Counter
from pathlib import Path

from qdrant_client import QdrantClient

from common.config import AppConfig, ConfigError, load_app_config
from common.logs import configure_logging
from common.settings import Settings, load_settings
from common.stack import GPU_SERVICE, INGEST, RUNTIME, Stack, StackError
from db.session import build_engine, build_sessionmaker
from ingest.corpus import Corpus, CorpusError, load_corpus
from ingest.embeddings import build_embedder
from ingest.files import plan_corpus_files
from ingest.index import ChunkIndex
from ingest.parsing import DocumentParser
from ingest.pipeline import IngestPipeline
from ingest.registry import FileRegistry
from ingest.report import write_report
from ingest.router import choose_route
from ingest.run import IngestRunner
from ingest.tokens import HfTokenCounter

EXIT_OK = 0
EXIT_ISSUES = 1
EXIT_CONFIG = 2
logger = logging.getLogger("ingest.cli")


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


class GpuOrchestrator:
    """Профили стенда вокруг двух фаз инжеста: VLM только на разборе, GPU под эмбеддинги после."""

    def __init__(self, stack: Stack | None) -> None:
        self._stack = stack

    def before_parse(self, *, needs_vlm: bool) -> None:
        if self._stack is None:
            return
        running = self._stack.running_services()
        if needs_vlm:
            logger.info("Оркестрация: поднимаю профиль ingest (vllm-dots), профиль runtime останавливается")
            self._stack.up(INGEST, switch=True, wait=True)
            return
        if GPU_SERVICE[RUNTIME] in running:
            logger.info("Оркестрация: сканов нет, но GPU нужен под эмбеддинги — останавливаю vllm-qwen")
            self._stack.stop_service(GPU_SERVICE[RUNTIME])
        elif GPU_SERVICE[INGEST] in running:
            logger.info("Оркестрация: сканов нет — останавливаю vllm-dots, GPU свободен")
            self._stack.stop_service(GPU_SERVICE[INGEST])

    def before_index(self) -> None:
        if self._stack is None:
            return
        if GPU_SERVICE[INGEST] in self._stack.running_services():
            logger.info("Оркестрация: разбор завершён — останавливаю vllm-dots, GPU отдаётся эмбеддингам")
            self._stack.stop_service(GPU_SERVICE[INGEST])

    def after_run(self, *, restore_runtime: bool) -> None:
        if self._stack is None or not restore_runtime:
            return
        logger.info("Оркестрация: поднимаю профиль runtime обратно")
        self._stack.up(RUNTIME, switch=True, wait=True)


async def run_ingest(config: AppConfig, settings: Settings, corpus: Corpus, args: argparse.Namespace) -> int:
    engine = build_engine(settings.database_url)
    client = QdrantClient(url=settings.resolve_qdrant_url(), timeout=int(config.qdrant.timeout_s))
    try:
        registry = FileRegistry(build_sessionmaker(engine))
        index = ChunkIndex(client, config.qdrant, config.embedding.dense_dim)
        embedding_dir = config.models.local_path(config.models.embedding)
        pipeline = IngestPipeline(
            config,
            parser=DocumentParser(config, vlm_base_url=settings.resolve_vlm_base_url(config)),
            embedder=build_embedder(embedding_dir, config.embedding, ingest=True),
            index=index,
            counter=HfTokenCounter(embedding_dir),
        )
        runner = IngestRunner(
            config, corpus=corpus, pipeline=pipeline, index=index, registry=registry, force=args.force
        )
        work = await runner.preflight()
        print(f"План: {work.summary()}")
        orchestrator = GpuOrchestrator(None if args.no_gpu_switch else Stack(config))
        orchestrator.before_parse(needs_vlm=work.needs_vlm)
        report = await runner.run(before_index=orchestrator.before_index)
        orchestrator.after_run(restore_runtime=args.restore_runtime)
    finally:
        client.close()
        await engine.dispose()
    json_path, md_path = write_report(report, config.paths.work_dir_absolute)
    for line in report.summary_lines():
        print(line)
    print(f"Отчёт: {md_path} и {json_path}")
    return EXIT_OK if report.failed == 0 and not report.issues else EXIT_ISSUES


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
    run_parser = subparsers.add_parser("run", help="полный инжест с оркестрацией GPU")
    run_parser.add_argument("--corpus", type=Path, help="каталог архива экспорта")
    run_parser.add_argument("--force", action="store_true", help="переразобрать все файлы, игнорируя реестр")
    run_parser.add_argument("--no-gpu-switch", action="store_true", help="не управлять контейнерами стенда")
    run_parser.add_argument("--restore-runtime", action="store_true", help="в конце поднять профиль runtime")
    args = parser.parse_args(argv)
    try:
        config = load_app_config(args.config)
    except ConfigError as exc:
        print(f"ОШИБКА КОНФИГУРАЦИИ: {exc}")
        return EXIT_CONFIG
    configure_logging(config.logging.level)
    corpus_root = args.corpus or config.paths.corpus_dir_absolute
    if args.command == "inspect":
        return inspect_corpus(corpus_root, config, show_files=args.files)
    try:
        corpus = load_corpus(corpus_root)
    except CorpusError as exc:
        print(f"ОШИБКА АРХИВА: {exc}")
        return EXIT_CONFIG
    if corpus.synthetic:
        print("ДАННЫЕ СИНТЕТИЧЕСКИЕ: индекс строится по синтетическому корпусу, метрики M1–M8 не считаются")
    try:
        return asyncio.run(run_ingest(config, load_settings(), corpus, args))
    except StackError as exc:
        print(f"ОШИБКА СТЕНДА: {exc}")
        return EXIT_CONFIG


if __name__ == "__main__":
    sys.exit(main())
