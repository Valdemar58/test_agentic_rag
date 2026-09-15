"""Smoke стенда (этап 3 §11): проверки против поднятого docker compose; без стенда — скип.

Сервис недоступен → скип с причиной; сервис доступен, но отвечает неверно → провал.
"""

from __future__ import annotations

import pytest

from common.config import AppConfig, load_app_config
from common.settings import Settings
from common.smoke import (
    CheckResult,
    ServiceUnavailable,
    check_llm_first_token,
    check_llm_russian,
    check_llm_tool_call,
    check_postgres,
    check_qdrant,
)

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def config() -> AppConfig:
    return load_app_config()


@pytest.fixture(scope="module")
def settings() -> Settings:
    return Settings()


def _assert_ok(result: CheckResult) -> None:
    assert result.ok, f"{result.name}: {result.detail}"


def test_postgres_migrations_applied(settings: Settings) -> None:
    try:
        result = check_postgres(settings)
    except ServiceUnavailable as exc:
        pytest.skip(str(exc))
    _assert_ok(result)


def test_qdrant_ready(settings: Settings, config: AppConfig) -> None:
    try:
        result = check_qdrant(settings, timeout_s=config.qdrant.timeout_s)
    except ServiceUnavailable as exc:
        pytest.skip(str(exc))
    _assert_ok(result)


@pytest.mark.gpu
def test_llm_answers_in_russian_and_calls_tool(settings: Settings, config: AppConfig) -> None:
    try:
        russian = check_llm_russian(settings, config)
    except ServiceUnavailable as exc:
        pytest.skip(str(exc))
    _assert_ok(russian)
    _assert_ok(check_llm_tool_call(settings, config))
    first_token = check_llm_first_token(settings, config)
    _assert_ok(first_token)
