"""LLM-клиент агента (6.1): Qwen3-8B через vLLM, OpenAI-совместимый API, tool calling.

Один класс LlamaIndex (`OpenAILike`) для всех ролей агента; роли различаются режимом размышлений Qwen3
и параметрами сэмплинга (`agent.thinking`, `agent.llm` в конфиге, N9). Режим размышлений передаётся
vLLM через `chat_template_kwargs.enable_thinking`, `top_k` — в теле запроса (в OpenAI API его нет);
оба уходят в `extra_body` клиента OpenAI. Блок размышлений vLLM отдаёт отдельным полем
(`reasoning_content`, парсер `qwen3`), LlamaIndex кладёт его в `ThinkingBlock` сообщения и обратно
в историю не отправляет — как рекомендует Qwen для многоходовых диалогов.
"""

from __future__ import annotations

from typing import Any

from llama_index.core.base.llms.types import ChatMessage, ThinkingBlock
from llama_index.llms.openai_like import OpenAILike
from openai import AsyncOpenAI, OpenAIError

from common.config import AppConfig, LlmRequestOptions, LlmRole
from common.settings import Settings

CHAT_TEMPLATE_KWARGS = "chat_template_kwargs"
EXTRA_BODY = "extra_body"


def request_kwargs(options: LlmRequestOptions) -> dict[str, Any]:
    """Поля запроса chat/completions сверх стандартных: top_p — как есть, специфика vLLM — в extra_body."""
    return {
        "top_p": options.top_p,
        EXTRA_BODY: {"top_k": options.top_k, CHAT_TEMPLATE_KWARGS: options.chat_template_kwargs()},
    }


def build_llm(config: AppConfig, settings: Settings, role: LlmRole) -> OpenAILike:
    """LLM для роли агента: Qwen3 профиля runtime, режим размышлений и сэмплинг по `agent.thinking`."""
    options = config.agent.llm_options(role)
    return OpenAILike(
        model=config.vllm.qwen.served_model_name,
        api_base=settings.resolve_llm_base_url(config),
        api_key=settings.llm_api_key.get_secret_value(),
        is_chat_model=True,
        is_function_calling_model=True,
        context_window=config.vllm.qwen.max_model_len,
        temperature=options.temperature,
        max_tokens=options.max_tokens,
        timeout=config.agent.llm.timeout_s,
        max_retries=config.agent.llm.max_retries,
        # strict-схемы инструментов — расширение OpenAI, vLLM их не требует
        strict=False,
        additional_kwargs=request_kwargs(options),
    )


async def llm_ready(config: AppConfig, settings: Settings, *, timeout_s: float) -> bool:
    """Готов ли vLLM отвечать: список моделей отдаётся без ошибки.

    Пока модель загружается после перезапуска стенда, сервер не слушает порт, и запрос падает ошибкой
    соединения (живой диалог 2026-09-17: вопрос из UI в это окно получил «Connection error»)."""
    client = AsyncOpenAI(
        base_url=settings.resolve_llm_base_url(config),
        api_key=settings.llm_api_key.get_secret_value(),
        timeout=timeout_s,
        max_retries=0,
    )
    try:
        await client.models.list()
    except OpenAIError:
        return False
    finally:
        await client.close()
    return True


def thinking_text(message: ChatMessage) -> str | None:
    """Текст размышлений из ответа модели (если роль работала с включёнными размышлениями)."""
    parts = [block.content for block in message.blocks if isinstance(block, ThinkingBlock) and block.content]
    return "\n".join(parts) or None
