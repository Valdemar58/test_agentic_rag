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
    LoopNotes,
    LoopText,
    ToolFinished,
    ToolStarted,
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
        if isinstance(event, ToolStarted):
            print(f"→ {event.status}", flush=True)
        elif isinstance(event, ToolFinished):
            mark = "✓" if event.ok else "✗"
            print(f"   {mark} {event.summary} ({event.seconds:.1f} с)", flush=True)
        elif isinstance(event, LoopText):
            print(f"· {event.text}", flush=True)
        elif isinstance(event, LoopNotes):
            print(f"Заметки: {event.text}\n\nОтвет:", flush=True)
        elif isinstance(event, AnswerDelta):
            print(event.text, end="", flush=True)
        elif isinstance(event, AnswerReady):
            answer = event.answer
            print(
                f"\n\n[{answer.seconds:.1f} с: поиск {answer.loop_seconds:.1f} с, "
                f"ответ {answer.answer_seconds:.1f} с; вызовов инструментов {len(answer.tool_calls)}"
                + ("; бюджет исчерпан" if answer.budget_exhausted else "")
                + ("; отказ" if answer.refused else "")
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
