"""Smoke-проверки стенда (этап 3 §11): БД с миграциями, Qdrant, LLM по-русски и tool calling.

Каждая проверка возвращает `CheckResult`; недоступность сервиса (нет соединения) — отдельное
исключение `ServiceUnavailable`, чтобы CLI и тесты отличали «стенд не поднят» от «сервис
отвечает неправильно».
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import httpx
from alembic.config import Config as AlembicConfig
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import Connection
from sqlalchemy.ext.asyncio import create_async_engine

from common.config import ROOT, AppConfig
from common.settings import Settings

CYRILLIC = re.compile(r"[А-Яа-яЁё]")
SEARCH_TOOL_NAME = "hybrid_search"
SEARCH_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": SEARCH_TOOL_NAME,
        "description": "Гибридный поиск по корпоративным документам: приказы, положения, договоры.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Поисковый запрос на русском языке"},
                "top_k": {"type": "integer", "description": "Сколько фрагментов вернуть"},
            },
            "required": ["query"],
        },
    },
}
SYSTEM_PROMPT = "Ты помощник по корпоративным документам. Отвечай только по-русски и кратко."
RUSSIAN_QUESTION = "Назови столицу России одним словом."
TOOL_QUESTION = "Найди действующие приказы об охране труда и назначении ответственных."


class ServiceUnavailable(Exception):
    """Сервис не отвечает: стенд не поднят или недоступен по сети."""


@dataclass(frozen=True)
class CheckResult:
    name: str
    ok: bool
    detail: str
    seconds: float


def _timed(start: float) -> float:
    return round(time.perf_counter() - start, 2)


# ---------- PostgreSQL ----------


def _revision_state(connection: Connection) -> str | None:
    return MigrationContext.configure(connection).get_current_revision()


async def _current_revision(database_url: str) -> str | None:
    engine = create_async_engine(database_url)
    try:
        async with engine.connect() as connection:
            return await connection.run_sync(_revision_state)
    finally:
        await engine.dispose()


def migrations_head() -> str | None:
    config = AlembicConfig(str(ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(ROOT / "alembic"))
    return ScriptDirectory.from_config(config).get_current_head()


def check_postgres(settings: Settings) -> CheckResult:
    """Соединение с прикладной БД и совпадение ревизии Alembic с head репозитория."""
    start = time.perf_counter()
    try:
        current = asyncio.run(_current_revision(settings.database_url))
    except OSError as exc:
        raise ServiceUnavailable(f"PostgreSQL {settings.app_db_host}:{settings.app_db_port}: {exc}") from exc
    head = migrations_head()
    ok = current is not None and current == head
    detail = f"ревизия БД {current or 'нет'}, head {head}" + (
        "" if ok else " — выполните alembic upgrade head"
    )
    return CheckResult("PostgreSQL и миграции", ok, detail, _timed(start))


# ---------- Qdrant ----------


def check_qdrant(settings: Settings, *, timeout_s: float) -> CheckResult:
    """Qdrant отвечает на /readyz и отдаёт версию."""
    start = time.perf_counter()
    url = settings.resolve_qdrant_url().rstrip("/")
    try:
        with httpx.Client(timeout=timeout_s) as client:
            ready = client.get(f"{url}/readyz")
            info = client.get(url).json()
            collections = client.get(f"{url}/collections").json()
    except httpx.HTTPError as exc:
        raise ServiceUnavailable(f"Qdrant {url}: {exc}") from exc
    names = [item["name"] for item in collections.get("result", {}).get("collections", [])]
    ok = ready.status_code == httpx.codes.OK
    detail = f"версия {info.get('version', '?')}, коллекций {len(names)}"
    return CheckResult("Qdrant", ok, detail, _timed(start))


# ---------- LLM ----------


def _chat_payload(config: AppConfig, messages: list[dict[str, Any]], **extra: Any) -> dict[str, Any]:
    options = config.agent.llm_options("tool_loop")
    return {
        "model": config.vllm.qwen.served_model_name,
        "messages": messages,
        "temperature": options.temperature,
        "top_p": options.top_p,
        "top_k": options.top_k,
        "max_tokens": options.max_tokens,
        "chat_template_kwargs": options.chat_template_kwargs(),
        **extra,
    }


def _post_chat(client: httpx.Client, base_url: str, payload: dict[str, Any]) -> dict[str, Any]:
    try:
        response = client.post(f"{base_url.rstrip('/')}/chat/completions", json=payload)
    except httpx.HTTPError as exc:
        raise ServiceUnavailable(f"LLM {base_url}: {exc}") from exc
    if response.status_code != httpx.codes.OK:
        raise ServiceUnavailable(f"LLM {base_url}: HTTP {response.status_code} {response.text[:200]}")
    data: dict[str, Any] = response.json()
    return data


def check_llm_russian(settings: Settings, config: AppConfig) -> CheckResult:
    """LLM отвечает по-русски на простой вопрос (NFR-1)."""
    start = time.perf_counter()
    base_url = settings.resolve_llm_base_url(config)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": RUSSIAN_QUESTION}]
    with httpx.Client(timeout=config.agent.llm.timeout_s) as client:
        data = _post_chat(client, base_url, _chat_payload(config, messages))
    text = str(data["choices"][0]["message"].get("content") or "").strip()
    ok = bool(CYRILLIC.search(text))
    usage = data.get("usage", {})
    detail = f"ответ «{text[:60]}», токенов {usage.get('completion_tokens', '?')}"
    return CheckResult("LLM отвечает по-русски", ok, detail, _timed(start))


def check_llm_tool_call(settings: Settings, config: AppConfig) -> CheckResult:
    """LLM выбирает инструмент hybrid_search и передаёт запрос в аргументах (tool calling)."""
    start = time.perf_counter()
    base_url = settings.resolve_llm_base_url(config)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": TOOL_QUESTION}]
    with httpx.Client(timeout=config.agent.llm.timeout_s) as client:
        data = _post_chat(
            client, base_url, _chat_payload(config, messages, tools=[SEARCH_TOOL], tool_choice="auto")
        )
    message = data["choices"][0]["message"]
    calls = message.get("tool_calls") or []
    if not calls:
        return CheckResult("LLM вызывает инструмент", False, "tool_calls пуст", _timed(start))
    function = calls[0]["function"]
    try:
        arguments = json.loads(function.get("arguments") or "{}")
    except json.JSONDecodeError:
        arguments = {}
    query = str(arguments.get("query", ""))
    ok = function.get("name") == SEARCH_TOOL_NAME and bool(query)
    detail = f"{function.get('name')}({json.dumps(arguments, ensure_ascii=False)[:80]})"
    return CheckResult("LLM вызывает инструмент", ok, detail, _timed(start))


def check_llm_first_token(settings: Settings, config: AppConfig) -> CheckResult:
    """Время до первого токена при стриминге против бюджета NFR-2 (первый сигнал ≤ 5 с)."""
    start = time.perf_counter()
    base_url = settings.resolve_llm_base_url(config)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": RUSSIAN_QUESTION}]
    payload = _chat_payload(config, messages, stream=True)
    first_token: float | None = None
    try:
        with (
            httpx.Client(timeout=config.agent.llm.timeout_s) as client,
            client.stream("POST", f"{base_url.rstrip('/')}/chat/completions", json=payload) as response,
        ):
            if response.status_code != httpx.codes.OK:
                raise ServiceUnavailable(f"LLM {base_url}: HTTP {response.status_code}")
            for line in response.iter_lines():
                if not line.startswith("data:") or line.endswith("[DONE]"):
                    continue
                chunk = json.loads(line[len("data:") :])
                delta = chunk["choices"][0].get("delta", {})
                if delta.get("content"):
                    first_token = time.perf_counter() - start
                    break
    except httpx.HTTPError as exc:
        raise ServiceUnavailable(f"LLM {base_url}: {exc}") from exc
    budget = config.eval.first_signal_budget_s
    if first_token is None:
        return CheckResult("Первый токен", False, "поток без содержимого", _timed(start))
    return CheckResult(
        "Первый токен", first_token <= budget, f"{first_token:.2f} с при бюджете {budget} с", _timed(start)
    )


def run_all(settings: Settings, config: AppConfig, *, skip_llm: bool = False) -> list[CheckResult]:
    """Все проверки по порядку; недоступный сервис — результат с ok=False и причиной."""
    checks: list[tuple[str, Callable[[], CheckResult]]] = [
        ("PostgreSQL и миграции", lambda: check_postgres(settings)),
        ("Qdrant", lambda: check_qdrant(settings, timeout_s=config.qdrant.timeout_s)),
    ]
    if not skip_llm:
        checks += [
            ("LLM отвечает по-русски", lambda: check_llm_russian(settings, config)),
            ("LLM вызывает инструмент", lambda: check_llm_tool_call(settings, config)),
            ("Первый токен", lambda: check_llm_first_token(settings, config)),
        ]
    results: list[CheckResult] = []
    for name, check in checks:
        start = time.perf_counter()
        try:
            results.append(check())
        except ServiceUnavailable as exc:
            results.append(CheckResult(name, False, f"НЕДОСТУПЕН: {exc}", _timed(start)))
    return results
