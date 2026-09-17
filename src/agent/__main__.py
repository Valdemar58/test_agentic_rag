"""Агент в консоли (демонстрация этапа 6).

  uv run python -m agent "вопрос"            # один вопрос
  uv run python -m agent                     # диалог: вопросы построчно, пустая строка — выход

Нужны поднятые vLLM (профиль runtime) и MCP-сервер (MCP_URL или порт из конфига). Показывает шаги
агента, заметки после поиска, стрим ответа и время. Код выхода: 0; 2 — конфиг или недоступный сервис.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from agent.runner import (
    AgentRunner,
    AgentSession,
    AnswerDelta,
    AnswerReady,
    AnswerRestarted,
    AnswerVerified,
    CacheUsed,
    GlossaryUsed,
    LoopNotes,
    LoopText,
    QueryRewritten,
    ToolFinished,
    ToolStarted,
    VerifyStarted,
)
from agent.service import build_runner
from common.config import ConfigError, load_app_config
from common.logs import configure_logging
from common.settings import load_settings

EXIT_OK = 0
EXIT_CONFIG = 2
PROMPT = "Вопрос> "


async def _ask(runner: AgentRunner, session: AgentSession, question: str) -> None:
    async for event in runner.run(question, session):
        if isinstance(event, GlossaryUsed):
            for item in event.expansions:
                print(f"· Глоссарий: {item.term} — {item.definition}", flush=True)
        elif isinstance(event, QueryRewritten):
            if event.changed:
                print(f"≈ Запрос с учётом диалога: {event.query}", flush=True)
            for index, query in enumerate(event.queries, start=1):
                print(f"   {index}. {query}", flush=True)
            if not event.needs_search:
                print("· Поиск по документам не нужен", flush=True)
        elif isinstance(event, CacheUsed):
            print(f"· Использую ранее найденное: {', '.join(event.document_aliases)}", flush=True)
        elif isinstance(event, ToolStarted):
            print(f"→ {event.status}", flush=True)
        elif isinstance(event, ToolFinished):
            mark = "✓" if event.ok else "✗"
            print(f"   {mark} {event.summary} ({event.seconds:.1f} с)", flush=True)
        elif isinstance(event, LoopText):
            print(f"· {event.text}", flush=True)
        elif isinstance(event, LoopNotes):
            print(f"Заметки: {event.text}\n\nОтвет (составляется)…", flush=True)
        elif isinstance(event, AnswerDelta):
            # стрим содержит маркеры [S#]/[D#]; в консоли показываем готовый текст с номерами ссылок
            print(".", end="", flush=True)
        elif isinstance(event, VerifyStarted):
            print("\n· Проверяю ответ по фрагментам…", flush=True)
        elif isinstance(event, AnswerRestarted):
            print("· Черновик не подтверждён свидетельствами, составляю ответ заново…", flush=True)
        elif isinstance(event, AnswerVerified):
            if not event.parsed:
                print("   ✗ проверка не выполнена, показан черновик", flush=True)
            elif not event.problems:
                print("   ✓ замечаний нет", flush=True)
            else:
                state = "исправлено" if event.corrected else "текст оставлен"
                print(f"   ! замечаний {len(event.problems)}, {state}:", flush=True)
                for problem in event.problems:
                    print(f"     — {problem.claim}: {problem.reason}", flush=True)
        elif isinstance(event, AnswerReady):
            answer = event.answer
            print(f"\n\n{answer.text}", flush=True)
            if answer.unresolved_markers:
                print(f"(удалены ссылки без источника: {', '.join(answer.unresolved_markers)})", flush=True)
            print(
                f"\n[{answer.seconds:.1f} с: поиск {answer.loop_seconds:.1f} с, "
                f"ответ {answer.answer_seconds:.1f} с, проверка {answer.verify_seconds:.1f} с; "
                f"вызовов инструментов {len(answer.tool_calls)}"
                + ("; бюджет исчерпан" if answer.budget_exhausted else "")
                + ("; контекст исчерпан" if answer.context_exhausted else "")
                + ("; отказ" if answer.refused else "")
                + (f"; трейс {answer.trace_id}" if answer.trace_id else "")
                + "]",
                flush=True,
            )


async def _main(args: argparse.Namespace) -> int:
    try:
        config = load_app_config(args.config)
    except ConfigError as exc:
        print(f"ОШИБКА КОНФИГУРАЦИИ: {exc}")
        return EXIT_CONFIG
    configure_logging(config.logging.level)
    settings = load_settings()
    try:
        runner = await build_runner(config, settings)
    except Exception as exc:  # noqa: BLE001 — недоступный MCP или vLLM: сообщение вместо трассировки
        print(f"ОШИБКА СТЕНДА: MCP-сервер {settings.resolve_mcp_url(config)} недоступен: {exc}")
        return EXIT_CONFIG
    session = AgentSession(config)
    try:
        if args.question:
            await _ask(runner, session, args.question)
            return EXIT_OK
        print("Диалог с агентом; пустая строка — выход.")
        while True:
            try:
                question = input(PROMPT).strip()
            except EOFError:
                break
            if not question:
                break
            await _ask(runner, session, question)
        return EXIT_OK
    finally:
        await runner.aclose()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="agent", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("question", nargs="?", help="вопрос; без него — диалог построчно")
    parser.add_argument("--config", type=Path, help="путь к app.yaml (по умолчанию configs/app.yaml)")
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    return asyncio.run(_main(args))


if __name__ == "__main__":
    sys.exit(main())
