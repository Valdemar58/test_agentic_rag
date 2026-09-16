"""FR-8 / AC-8.1 на живом Langfuse: трейс вопроса содержит цепочку переписывание → цикл → инструменты → ответ.

LLM и инструменты — фейки (сценарий и MCP-сервер в памяти), трейсинг — настоящий клиент Langfuse
профиля observability. Langfuse 4 в режиме events_only не отдаёт трейсы через публичный API, поэтому
записанные наблюдения читаются из ClickHouse стенда (`docker exec … clickhouse-client`).
"""

from __future__ import annotations

import os
import subprocess
import time

import httpx
import pytest

from agent.rendering import TOOL_SEARCH
from agent.tracing import LangfuseTracing
from common.config import load_app_config
from common.settings import Settings
from tests.unit.agent_fakes import tool_step
from tests.unit.test_agent_runner import QUERY, Harness
from tests.unit.test_agent_runner import harness as harness_fixture  # noqa: F401 — фикстура pytest «harness»

pytestmark = pytest.mark.integration

CLICKHOUSE_CONTAINER = "agentic-rag-clickhouse-1"
CLICKHOUSE_PASSWORD_ENV = "LANGFUSE_CLICKHOUSE_PASSWORD"  # как в docker-compose.yml
CLICKHOUSE_DEFAULT_PASSWORD = "clickhouse"
EXPECTED_STEPS = {"question", "rewrite", "tool_loop", "answer"}
WAIT_S = 60
POLL_S = 3


@pytest.fixture(scope="module")
def tracing() -> LangfuseTracing:
    settings = Settings()
    if not settings.langfuse_enabled or not settings.langfuse_public_key:
        pytest.skip("LANGFUSE_ENABLED и ключи не заданы: трейсинг выключен")
    try:
        httpx.get(f"{settings.langfuse_url}/api/public/health", timeout=5).raise_for_status()
    except httpx.HTTPError as exc:
        pytest.skip(f"Langfuse {settings.langfuse_url} недоступен: {exc}")
    return LangfuseTracing.from_settings(settings, load_app_config().langfuse)


def _events(trace_id: str, password: str) -> list[tuple[str, str, str]]:
    query = (
        "SELECT name, type, session_id FROM events_full "
        f"WHERE trace_id = '{trace_id}' ORDER BY start_time FORMAT TSV"
    )
    completed = subprocess.run(
        [
            "docker",
            "exec",
            CLICKHOUSE_CONTAINER,
            "clickhouse-client",
            "--password",
            password,
            "--query",
            query,
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    if completed.returncode != 0:
        pytest.skip(f"ClickHouse стенда недоступен: {completed.stderr.strip()[:200]}")
    rows = [line.split("\t") for line in completed.stdout.splitlines() if line.strip()]
    return [(row[0], row[1], row[2]) for row in rows if len(row) == 3]


async def test_trace_chain_is_recorded_in_langfuse(harness: Harness, tracing: LangfuseTracing) -> None:
    runner = harness.runner(
        [tool_step(TOOL_SEARCH, query=QUERY), "Заметки: D1."],
        ["Отчёт сдаётся до пятого числа [S1]."],
        rewrite_steps=['{"query": "срок сдачи отчёта по охране труда", "needs_search": true}'],
        tracing=tracing,
    )
    session = harness.session()
    answer = await runner.ask("Когда сдаётся отчёт?", session)
    assert answer.trace_id
    tracing.flush()

    password = os.environ.get(CLICKHOUSE_PASSWORD_ENV, CLICKHOUSE_DEFAULT_PASSWORD)
    deadline = time.monotonic() + WAIT_S
    events: list[tuple[str, str, str]] = []
    while time.monotonic() < deadline:
        events = _events(answer.trace_id, password)
        recorded = {name for name, _, _ in events}
        if EXPECTED_STEPS <= recorded and any(kind == "TOOL" for _, kind, _ in events):
            break
        time.sleep(POLL_S)
    names = [name for name, _, _ in events]
    kinds = {kind for _, kind, _ in events}
    assert EXPECTED_STEPS <= set(names), names
    assert names.index("rewrite") < names.index("tool_loop") < names.index("answer")
    assert "TOOL" in kinds and "GENERATION" in kinds, kinds
    assert {session_id for _, _, session_id in events} == {session.id}
