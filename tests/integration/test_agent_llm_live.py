"""LLM-клиент агента на живом vLLM (6.1): tool calling через OpenAILike и режим размышлений по ролям."""

from __future__ import annotations

import time

import httpx
import pytest
from llama_index.core.base.llms.types import ChatMessage, ThinkingBlock
from llama_index.core.tools import FunctionTool

from agent.llm import build_llm, thinking_text
from common.config import AppConfig, load_app_config
from common.settings import Settings

pytestmark = [pytest.mark.integration, pytest.mark.gpu]

TOOL_QUESTION = "Найди действующие приказы об охране труда и назначении ответственных."
REWRITE_QUESTION = "Перепиши вопрос «а для филиалов?» после вопроса «какие требования к СИЗ?» одной фразой."


@pytest.fixture(scope="module")
def config() -> AppConfig:
    return load_app_config()


@pytest.fixture(scope="module")
def settings(config: AppConfig) -> Settings:
    settings = Settings()
    base_url = settings.resolve_llm_base_url(config)
    try:
        httpx.get(f"{base_url}/models", timeout=5).raise_for_status()
    except httpx.HTTPError as exc:
        pytest.skip(f"vLLM {base_url} недоступен: {exc}")
    return settings


def hybrid_search(query: str, top_k: int = 5) -> str:
    """Поиск фрагментов в документах СЭД: приказы, положения, договоры. query — запрос по-русски."""
    return "фрагментов нет"


async def test_tool_loop_role_calls_tool_without_thinking(config: AppConfig, settings: Settings) -> None:
    llm = build_llm(config, settings, "tool_loop")
    tool = FunctionTool.from_defaults(fn=hybrid_search)
    started = time.perf_counter()
    response = await llm.achat_with_tools(
        [tool],
        chat_history=[
            ChatMessage(role="system", content="Ты помощник по документам СЭД. Отвечай по-русски."),
            ChatMessage(role="user", content=TOOL_QUESTION),
        ],
    )
    elapsed = time.perf_counter() - started
    calls = llm.get_tool_calls_from_response(response, error_on_no_tool_call=False)
    assert calls and calls[0].tool_name == "hybrid_search"
    assert calls[0].tool_kwargs.get("query")
    assert thinking_text(response.message) is None, "в цикле инструментов размышления выключены"
    print(f"\ntool call за {elapsed:.1f} с: {calls[0].tool_kwargs}")


async def test_rewrite_role_returns_thinking_block(config: AppConfig, settings: Settings) -> None:
    llm = build_llm(config, settings, "rewrite")
    started = time.perf_counter()
    response = await llm.achat(
        [
            ChatMessage(role="system", content="Отвечай по-русски одной фразой."),
            ChatMessage(role="user", content=REWRITE_QUESTION),
        ]
    )
    elapsed = time.perf_counter() - started
    thinking = thinking_text(response.message)
    assert thinking, "роль rewrite работает с включёнными размышлениями (reasoning_content от vLLM)"
    assert any(isinstance(block, ThinkingBlock) for block in response.message.blocks)
    assert response.message.content and "СИЗ" in response.message.content
    print(f"\nразмышления {len(thinking)} символов, ответ «{response.message.content}» за {elapsed:.1f} с")
