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

LOCALHOST = "localhost"
ASYNCPG_DRIVER = "postgresql+asyncpg"


class Settings(BaseSettings):
    """Переменные окружения приложения; имена совпадают с `.env.example` и docker-compose.yml."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    card_service_url: str = Field(
        default="http://localhost:8010",
        description="Сервис карточек: мок в dev, реальный FastAPI-сервис заказчика в проде",
    )
    card_service_username: str = Field(default="", description="HTTP Basic к реальному сервису (N6)")
    card_service_password: SecretStr = Field(
        default=SecretStr(""), description="HTTP Basic к реальному сервису"
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
        default=False, description="Трейсинг включается вместе с профилем observability"
    )
    langfuse_host: str = Field(default="http://localhost:3000")
    langfuse_public_key: str = Field(default="")
    langfuse_secret_key: SecretStr = Field(default=SecretStr(""))

    @property
    def database_url(self) -> str:
        password = self.app_db_password.get_secret_value()
        return (
            f"{ASYNCPG_DRIVER}://{self.app_db_user}:{password}"
            f"@{self.app_db_host}:{self.app_db_port}/{self.app_db_name}"
        )

    def resolve_qdrant_url(self) -> str:
        return self.qdrant_url or f"http://{LOCALHOST}:{self.qdrant_port}"

    def resolve_llm_base_url(self, config: AppConfig) -> str:
        return self.llm_base_url or f"http://{LOCALHOST}:{config.vllm.qwen.port}/v1"

    def resolve_vlm_base_url(self, config: AppConfig) -> str:
        return self.vlm_base_url or f"http://{LOCALHOST}:{config.vllm.dots.port}/v1"

    def resolve_mcp_url(self, config: AppConfig) -> str:
        return self.mcp_url or f"http://{LOCALHOST}:{config.mcp.port}{config.mcp.path}"


def load_settings() -> Settings:
    """Настройки из переменных окружения и `.env` в текущем каталоге."""
    return Settings()
