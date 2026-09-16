"""Настройки из окружения: умолчания, переопределение переменными, URL из конфига."""

from __future__ import annotations

import pytest

from common.config import DEFAULT_CONFIG_PATH, load_app_config
from common.settings import LOCALHOST, Settings

ENV_KEYS = [
    "CARD_SERVICE_URL",
    "CARD_SERVICE_USERNAME",
    "CARD_SERVICE_PASSWORD",
    "APP_DB_HOST",
    "APP_DB_PORT",
    "APP_DB_USER",
    "APP_DB_PASSWORD",
    "APP_DB_NAME",
    "QDRANT_URL",
    "QDRANT_PORT",
    "LLM_BASE_URL",
    "VLM_BASE_URL",
    "MCP_URL",
    "LANGFUSE_ENABLED",
    "LANGFUSE_HOST",
    "LANGFUSE_PUBLIC_KEY",
    "LANGFUSE_SECRET_KEY",
]


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    for key in ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    return monkeypatch


def test_defaults_point_to_loopback_and_ports_from_config(clean_env: pytest.MonkeyPatch) -> None:
    settings = Settings(_env_file=None)
    config = load_app_config(DEFAULT_CONFIG_PATH)
    # 127.0.0.1, а не localhost: на Windows localhost сначала уходит в IPv6 и каждый запрос ждёт ~2 с
    assert LOCALHOST == "127.0.0.1"
    assert settings.database_url == f"postgresql+asyncpg://rag:rag@{LOCALHOST}:5432/rag"
    assert settings.resolve_qdrant_url() == f"http://{LOCALHOST}:6333"
    assert settings.resolve_llm_base_url(config) == f"http://{LOCALHOST}:{config.vllm.qwen.port}/v1"
    assert settings.resolve_vlm_base_url(config) == f"http://{LOCALHOST}:{config.vllm.dots.port}/v1"
    assert settings.resolve_mcp_url(config) == f"http://{LOCALHOST}:{config.mcp.port}{config.mcp.path}"
    assert settings.langfuse_enabled is False
    assert settings.card_service_url is None
    assert settings.resolve_card_service_url(config) == f"http://{LOCALHOST}:{config.mock_card_service.port}"


def test_environment_overrides_and_secrets_are_hidden(clean_env: pytest.MonkeyPatch) -> None:
    clean_env.setenv("CARD_SERVICE_URL", "http://cards.example.internal")
    clean_env.setenv("CARD_SERVICE_PASSWORD", "top-secret")
    clean_env.setenv("APP_DB_HOST", "postgres-app")
    clean_env.setenv("APP_DB_PORT", "5434")
    clean_env.setenv("APP_DB_PASSWORD", "p@ss")
    clean_env.setenv("QDRANT_URL", "http://qdrant:6333")
    clean_env.setenv("LLM_BASE_URL", "http://vllm-qwen:8000/v1")
    clean_env.setenv("LANGFUSE_ENABLED", "true")
    clean_env.setenv("LANGFUSE_SECRET_KEY", "sk-lf-x")
    settings = Settings(_env_file=None)
    config = load_app_config(DEFAULT_CONFIG_PATH)
    assert settings.resolve_card_service_url(config) == "http://cards.example.internal"
    assert settings.database_url == "postgresql+asyncpg://rag:p@ss@postgres-app:5434/rag"
    assert settings.resolve_qdrant_url() == "http://qdrant:6333"
    assert settings.resolve_llm_base_url(config) == "http://vllm-qwen:8000/v1"
    assert settings.langfuse_enabled is True
    # секреты не утекают в repr/str
    assert "top-secret" not in repr(settings) and "sk-lf-x" not in repr(settings)
    assert settings.card_service_password.get_secret_value() == "top-secret"


def test_invalid_port_is_rejected(clean_env: pytest.MonkeyPatch) -> None:
    clean_env.setenv("APP_DB_PORT", "70000")
    with pytest.raises(ValueError):
        Settings(_env_file=None)
