"""Настройки из окружения: URL сервисов, учётные записи, секреты (NFR-4, §8.0).

Всё, что различается между хостом разработки, контейнером и контуром заказчика, задаётся
переменными окружения (или `.env`), а не в `configs/app.yaml`. Переключение мок ↔ реальный
сервис карточек — это только `CARD_SERVICE_URL` (FR-2). Пути к внешнему коду SDK и сервиса
карточек живут в `contracts.external_paths`.
"""

from __future__ import annotations

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from common.config import AppConfig

# 127.0.0.1, а не localhost: на Windows имя сначала резолвится в IPv6 ::1, где порты Docker Desktop
# не слушают, и каждый запрос к Qdrant/vLLM ждал ~2 с до отката на IPv4 (замер 2026-09-15)
LOCALHOST = "127.0.0.1"
ASYNCPG_DRIVER = "postgresql+asyncpg"


class Settings(BaseSettings):
    """Переменные окружения приложения; имена совпадают с `.env.example` и docker-compose.yml."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    card_service_url: str | None = Field(
        default=None,
        description="Сервис карточек: мок в dev, реальный FastAPI-сервис заказчика в проде; "
        "по умолчанию мок на порту mock_card_service.port из конфига",
    )
    card_service_username: str = Field(default="", description="HTTP Basic к реальному сервису (N6)")
    card_service_password: SecretStr = Field(
        default=SecretStr(""), description="HTTP Basic к реальному сервису"
    )
    tessa_card_url_base: str = Field(
        default="",
        description="База ссылки на карточку в СЭД (как TESSA_CARD_URL_BASE в asu_toir): к ней "
        "добавляется ID документа, и ссылка появляется в блоке «Источники». Пусто — ссылок нет",
    )

    app_db_host: str = Field(default=LOCALHOST, description="Хост прикладной PostgreSQL")
    app_db_port: int = Field(default=5432, ge=1, le=65535)
    app_db_user: str = Field(default="rag")
    app_db_password: SecretStr = Field(default=SecretStr("rag"))
    app_db_name: str = Field(default="rag")

    qdrant_url: str | None = Field(default=None, description="URL Qdrant; по умолчанию localhost:6333")
    qdrant_port: int = Field(default=6333, ge=1, le=65535, description="Порт Qdrant на хосте (как в compose)")
    llm_base_url: str | None = Field(default=None, description="OpenAI-совместимый endpoint Qwen3")
    vlm_base_url: str | None = Field(default=None, description="OpenAI-совместимый endpoint dots.mocr")
    llm_api_key: SecretStr = Field(default=SecretStr("not-needed"), description="vLLM без ключа")
    mcp_url: str | None = Field(default=None, description="URL MCP-сервера для агента")

    langfuse_enabled: bool = Field(
        default=False, description="Трейсинг включается вместе с профилем observability (FR-8)"
    )
    langfuse_url: str = Field(
        default=f"http://{LOCALHOST}:3000", description="Langfuse self-hosted; имя как в docker-compose.yml"
    )
    langfuse_public_key: str = Field(default="")
    langfuse_secret_key: SecretStr = Field(default=SecretStr(""))

    # Вход в UI (FR-7): в dev один пользователь по логину и паролю; в проде — OIDC через переменные
    # OAUTH_GENERIC_* Chainlit (решение заказчика 2026-09-16), которые Chainlit читает сам.
    ui_username: str = Field(default="admin", description="Логин единственного пользователя UI (dev)")
    ui_password: SecretStr = Field(default=SecretStr("admin"), description="Пароль пользователя UI (dev)")

    @property
    def database_url(self) -> str:
        password = self.app_db_password.get_secret_value()
        return (
            f"{ASYNCPG_DRIVER}://{self.app_db_user}:{password}"
            f"@{self.app_db_host}:{self.app_db_port}/{self.app_db_name}"
        )

    def resolve_qdrant_url(self) -> str:
        return self.qdrant_url or f"http://{LOCALHOST}:{self.qdrant_port}"

    def resolve_card_service_url(self, config: AppConfig) -> str:
        return self.card_service_url or f"http://{LOCALHOST}:{config.mock_card_service.port}"

    def resolve_llm_base_url(self, config: AppConfig) -> str:
        return self.llm_base_url or f"http://{LOCALHOST}:{config.vllm.qwen.port}/v1"

    def resolve_vlm_base_url(self, config: AppConfig) -> str:
        return self.vlm_base_url or f"http://{LOCALHOST}:{config.vllm.dots.port}/v1"

    def resolve_mcp_url(self, config: AppConfig) -> str:
        return self.mcp_url or f"http://{LOCALHOST}:{config.mcp.port}{config.mcp.path}"


def load_settings() -> Settings:
    """Настройки из переменных окружения и `.env` в текущем каталоге."""
    return Settings()
