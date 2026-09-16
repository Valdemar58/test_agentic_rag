"""AC-7.1 / M6 (7.4): первый видимый сигнал в потоке UI укладывается в бюджет `eval.first_signal_budget_s`.

Живой стенд. Первый сигнал — первое событие потока `ui.flow.run_question` (шаг «Разбираю вопрос»);
браузер добавляет только доставку по websocket. Маркеры integration + gpu: нужны vLLM профиля runtime
и MCP-сервер, без них тест скипается.
"""

from __future__ import annotations

import time

import httpx
import pytest

from agent.runner import AgentSession
from agent.service import build_runner
from common.config import load_app_config
from common.settings import load_settings
from ui.flow import Final, StepStarted, run_question

pytestmark = [pytest.mark.integration, pytest.mark.gpu]

QUESTION = "Какие инструкции по охране труда утверждены приказами?"


async def test_first_signal_within_budget_on_live_stand() -> None:
    config = load_app_config()
    settings = load_settings()
    base_url = settings.resolve_llm_base_url(config)
    try:
        httpx.get(f"{base_url}/models", timeout=5).raise_for_status()
    except httpx.HTTPError as exc:
        pytest.skip(f"vLLM {base_url} недоступен: {exc}")
    try:
        runner = await build_runner(config, settings)
    except Exception as exc:  # noqa: BLE001 — стенд не поднят: скип, а не падение
        pytest.skip(f"MCP-сервер {settings.resolve_mcp_url(config)} недоступен: {exc}")

    budget = config.eval.first_signal_budget_s
    started = time.perf_counter()
    first_event_s: float | None = None
    final: Final | None = None
    try:
        async for event in run_question(runner, AgentSession(config), QUESTION):
            if first_event_s is None:
                first_event_s = time.perf_counter() - started
                assert isinstance(event, StepStarted)
            if isinstance(event, Final):
                final = event
    finally:
        await runner.aclose()

    assert first_event_s is not None and final is not None
    assert first_event_s <= budget, f"первый сигнал через {first_event_s:.2f} с при бюджете {budget} с"
    assert final.first_signal_s <= budget
    print(
        f"\nпервый сигнал {first_event_s:.3f} с (в потоке {final.first_signal_s:.3f} с), "
        f"ответ целиком {final.seconds:.1f} с, вызовов инструментов {len(final.answer.tool_calls)}"
    )
