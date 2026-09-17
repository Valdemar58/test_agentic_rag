"""Прогон голден-сета через агента и отчёт по метрикам (§10.2).

  uv run python -m eval.run [--golden PATH] [--only id,category] [--limit N] [--no-judge] [--tag NAME]

Каждый вопрос идёт в агента отдельной сессией — как новый диалог пользователя; у сценариев
`clarification` второй вопрос задаётся в той же сессии, чтобы проверить работу с контекстом (FR-6).
По ответу считаются детерминированные признаки (ожидаемые документы в источниках, потерянные ссылки,
явный отказ, статусы процитированных документов), затем ответ оценивает LLM-судья (`eval/judge.py`).
Отчёт пишется в `eval.reports_dir` двумя файлами — markdown и json; json прошлого прогона даёт diff.

Нужны поднятый профиль runtime (vLLM + MCP-сервер) и индекс реального корпуса. Ошибка одного вопроса
не роняет прогон: она попадает в отчёт, и метрики по этому вопросу не считаются.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from collections.abc import Awaitable, Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path

from agent.runner import AgentRunner, AgentSession, Answer, AnswerDelta, AnswerReady
from agent.service import build_runner
from common.config import AppConfig, ConfigError, load_app_config
from common.logs import configure_logging
from common.settings import load_settings
from eval.golden_set import GoldenQuestion, GoldenSet, GoldenSetError, load_golden_set
from eval.judge import AnswerJudge, JudgeVerdict, build_judge_llm
from eval.metrics import (
    EvalMetrics,
    QuestionOutcome,
    compute_metrics,
    diff_runs,
    latest_report,
    render_markdown,
    run_payload,
)

EXIT_OK = 0
EXIT_ISSUES = 1
EXIT_CONFIG = 2
logger = logging.getLogger("eval.run")

Ask = Callable[[GoldenQuestion], Awaitable[QuestionOutcome]]
Judge = Callable[[GoldenQuestion, QuestionOutcome], Awaitable[tuple[JudgeVerdict, JudgeVerdict | None]]]
INGEST_REPORT_GLOB = "ingest_*.json"
REPORT_PREFIX = "eval"


def select(golden: GoldenSet, *, only: Sequence[str] = (), limit: int | None = None) -> list[GoldenQuestion]:
    """Вопросы прогона: все, либо по id и категориям из `--only`, не больше `--limit`."""
    wanted = {item.strip().casefold() for item in only if item.strip()}
    questions = [
        question
        for question in golden.questions
        if not wanted or question.id.casefold() in wanted or question.category in wanted
    ]
    return questions[:limit] if limit else questions


def ingest_success_share(work_dir: Path) -> float | None:
    """Доля успешно проиндексированных файлов (M8) из самого свежего отчёта инжеста."""
    reports = latest_report((work_dir / "reports").glob(INGEST_REPORT_GLOB))
    if reports is None:
        return None
    try:
        payload = json.loads(Path(reports).read_text(encoding="utf-8"))
        return float(payload["success_share"])
    except (OSError, ValueError, KeyError) as exc:
        logger.warning("Отчёт инжеста %s не прочитан: %s", reports, exc)
        return None


class AgentAsk:
    """Вопрос голден-сета → ответ агента с замерами; уточнение задаётся в той же сессии (FR-6)."""

    def __init__(self, runner: AgentRunner, config: AppConfig) -> None:
        self._runner = runner
        self._config = config

    async def __call__(self, question: GoldenQuestion) -> QuestionOutcome:
        outcome = QuestionOutcome(
            id=question.id,
            category=question.category,
            derived=question.derived,
            question=question.question,
            answer="",
            expected_doc_ids=list(question.expected_doc_ids),
            follow_up=question.follow_up,
        )
        session = AgentSession(self._config)
        try:
            answer, first_signal, first_token = await self._ask(question.question, session)
        except Exception as exc:  # noqa: BLE001 — один вопрос не должен ронять прогон
            logger.exception("Вопрос %s не отработан", question.id)
            return outcome.model_copy(update={"error": f"{type(exc).__name__}: {exc}"})
        update: dict[str, object] = {
            "answer": answer.text,
            "sources": list(answer.sources),
            "refused": answer.refused,
            "unresolved": list(answer.unresolved_markers),
            "tool_calls": len(answer.tool_calls),
            "first_signal_s": first_signal,
            "first_token_s": first_token,
            "seconds": answer.seconds,
            "trace_id": answer.trace_id,
        }
        if question.follow_up:
            try:
                follow_up, _, _ = await self._ask(question.follow_up, session)
                # источники второго хода тоже засчитываются: ожидаемый документ может найтись там
                update |= {
                    "follow_up_answer": follow_up.text,
                    "follow_up_seconds": follow_up.seconds,
                    "sources": [*answer.sources, *follow_up.sources],
                }
            except Exception as exc:  # noqa: BLE001 — уточнение не должно ронять прогон
                logger.exception("Уточнение вопроса %s не отработано", question.id)
                update["error"] = f"уточнение: {type(exc).__name__}: {exc}"
        return outcome.model_copy(update=update)

    async def _ask(self, text: str, session: AgentSession) -> tuple[Answer, float, float]:
        started = time.perf_counter()
        first_signal = 0.0
        first_token = 0.0
        answer = None
        async for event in self._runner.run(text, session):
            if not first_signal:
                first_signal = time.perf_counter() - started
            if not first_token and isinstance(event, AnswerDelta):
                first_token = time.perf_counter() - started
            if isinstance(event, AnswerReady):
                answer = event.answer
        if answer is None:
            raise RuntimeError("агент не вернул ответ")
        return answer, first_signal, first_token


class JudgeAsk:
    """Оценка ответа и уточнения судьёй."""

    def __init__(self, judge: AnswerJudge) -> None:
        self._judge = judge

    async def __call__(
        self, question: GoldenQuestion, outcome: QuestionOutcome
    ) -> tuple[JudgeVerdict, JudgeVerdict | None]:
        verdict = await self._judge.judge(
            question.question, question.expected_answer, outcome.answer, outcome.sources
        )
        follow_up = None
        if question.follow_up and outcome.follow_up_answer:
            follow_up = await self._judge.judge(
                question.follow_up,
                question.follow_up_expected or "",
                outcome.follow_up_answer,
                outcome.sources,
            )
        return verdict, follow_up


async def run_eval(
    questions: Sequence[GoldenQuestion],
    *,
    ask: Ask,
    judge: Judge | None = None,
    progress: Callable[[QuestionOutcome], None] | None = None,
) -> list[QuestionOutcome]:
    """Прогон вопросов по порядку: ответ агента, затем вердикт судьи (если он включён)."""
    outcomes: list[QuestionOutcome] = []
    for question in questions:
        outcome = await ask(question)
        if judge is not None and outcome.error is None and outcome.answer:
            verdict, follow_up = await judge(question, outcome)
            outcome = outcome.model_copy(update={"judge": verdict, "follow_up_judge": follow_up})
        outcomes.append(outcome)
        if progress is not None:
            progress(outcome)
    return outcomes


def write_reports(
    outcomes: Sequence[QuestionOutcome],
    metrics: EvalMetrics,
    *,
    config: AppConfig,
    golden: GoldenSet,
    tag: str | None = None,
) -> tuple[Path, Path, list[str]]:
    """Пишет json и markdown в `eval.reports_dir`; возвращает пути и diff с прошлым прогоном."""
    reports_dir = config.eval.reports_dir_absolute
    reports_dir.mkdir(parents=True, exist_ok=True)
    previous_path = latest_report(reports_dir.glob(f"{REPORT_PREFIX}_*.json"))
    previous = None
    if previous_path is not None:
        try:
            previous = json.loads(Path(previous_path).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            logger.warning("Прошлый отчёт %s не прочитан: %s", previous_path, exc)
    payload = run_payload(outcomes, metrics, corpus=golden.corpus, synthetic=golden.synthetic)
    diff = diff_runs(previous, payload)
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S") + (f"_{tag}" if tag else "")
    json_path = reports_dir / f"{REPORT_PREFIX}_{stamp}.json"
    md_path = reports_dir / f"{REPORT_PREFIX}_{stamp}.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    md_path.write_text(
        render_markdown(outcomes, metrics, corpus=golden.corpus, synthetic=golden.synthetic, diff=diff),
        encoding="utf-8",
    )
    return json_path, md_path, diff


async def _main(args: argparse.Namespace) -> int:
    try:
        config = load_app_config(args.config)
    except ConfigError as exc:
        print(f"ОШИБКА КОНФИГУРАЦИИ: {exc}")
        return EXIT_CONFIG
    configure_logging(config.logging.level)
    try:
        golden = load_golden_set(args.golden or config.eval.golden_set_absolute)
    except GoldenSetError as exc:
        print(f"ОШИБКА ГОЛДЕН-СЕТА: {exc}")
        return EXIT_CONFIG
    gaps = golden.gaps()
    if gaps:
        print("ВНИМАНИЕ, состав сета не по §10.1: " + "; ".join(gaps))
    if golden.synthetic:
        print("ДАННЫЕ СИНТЕТИЧЕСКИЕ: метрики M1–M8 не считаются достигнутыми")
    questions = select(golden, only=args.only or (), limit=args.limit)
    if not questions:
        print("Вопросы не выбраны: проверьте --only")
        return EXIT_CONFIG

    settings = load_settings()
    try:
        runner = await build_runner(config, settings)
    except Exception as exc:  # noqa: BLE001 — недоступный стенд: сообщение вместо трассировки
        print(f"ОШИБКА СТЕНДА: MCP-сервер {settings.resolve_mcp_url(config)} недоступен: {exc}")
        return EXIT_CONFIG
    judge: Judge | None = None
    if not args.no_judge:
        judge = JudgeAsk(AnswerJudge(build_judge_llm(config, settings), config.eval.judge))

    print(f"Прогон {len(questions)} вопросов по корпусу {golden.corpus}")
    started = time.perf_counter()
    index = 0

    def progress(outcome: QuestionOutcome) -> None:
        nonlocal index
        index += 1
        score = outcome.judge.correctness if outcome.judge and outcome.judge.parsed else "—"
        state = outcome.error or f"документы {'найдены' if outcome.hit else 'не найдены'}, судья {score}"
        print(f"[{index}/{len(questions)}] {outcome.id}: {state} ({outcome.seconds:.0f} с)", flush=True)

    try:
        outcomes = await run_eval(questions, ask=AgentAsk(runner, config), judge=judge, progress=progress)
    finally:
        await runner.aclose()
    metrics = compute_metrics(
        outcomes,
        first_signal_budget_s=config.eval.first_signal_budget_s,
        ingest_share=ingest_success_share(config.paths.work_dir_absolute),
    )
    json_path, md_path, diff = write_reports(outcomes, metrics, config=config, golden=golden, tag=args.tag)
    print(f"\nПрогон занял {time.perf_counter() - started:.0f} с")
    for metric in metrics.metrics:
        print(f"  {metric.id:<14} {metric.rendered():>7}  цель {metric.target:<8} {metric.reached}")
    for line in diff:
        print(f"  изменение: {line}")
    print(f"Отчёт: {md_path} и {json_path}")
    failed = [outcome.id for outcome in outcomes if outcome.error]
    if failed:
        print(f"С ошибкой прогона: {', '.join(failed)}")
    return EXIT_ISSUES if failed else EXIT_OK


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="eval.run", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, help="путь к app.yaml")
    parser.add_argument("--golden", type=Path, help="путь к голден-сету (по умолчанию eval.golden_set)")
    parser.add_argument("--only", nargs="*", help="id вопросов или категории")
    parser.add_argument("--limit", type=int, help="взять первые N вопросов")
    parser.add_argument("--no-judge", action="store_true", help="без LLM-судьи (только детерминированное)")
    parser.add_argument("--tag", help="метка в имени файла отчёта")
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    return asyncio.run(_main(args))


if __name__ == "__main__":
    sys.exit(main())
