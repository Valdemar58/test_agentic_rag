"""LLM-клиент агента (6.1): режим размышлений и сэмплинг по ролям попадают в запрос к vLLM."""

from __future__ import annotations

import httpx
import respx
from llama_index.core.base.llms.types import ChatMessage, TextBlock, ThinkingBlock

from agent.llm import CHAT_TEMPLATE_KWARGS, EXTRA_BODY, build_llm, llm_ready, thinking_text
from common.config import DEFAULT_CONFIG_PATH, load_app_config
from common.settings import Settings

CONFIG = load_app_config(DEFAULT_CONFIG_PATH)
SETTINGS = Settings(_env_file=None, llm_base_url="http://llm.test/v1")


def test_roles_get_thinking_mode_and_sampling_from_config() -> None:
    loop = CONFIG.agent.llm_options("tool_loop")
    rewrite = CONFIG.agent.llm_options("rewrite")
    assert loop.enable_thinking is False and rewrite.enable_thinking is True
    assert (loop.temperature, loop.top_p, loop.top_k) == (
        CONFIG.agent.llm.sampling.temperature,
        CONFIG.agent.llm.sampling.top_p,
        CONFIG.agent.llm.sampling.top_k,
    )
    assert (rewrite.temperature, rewrite.top_p) == (
        CONFIG.agent.llm.thinking_sampling.temperature,
        CONFIG.agent.llm.thinking_sampling.top_p,
    )
    assert loop.max_tokens == CONFIG.agent.llm.max_tokens
    assert rewrite.max_tokens == CONFIG.agent.llm.thinking_max_tokens


def test_build_llm_targets_vllm_with_tool_calling_and_extra_body() -> None:
    llm = build_llm(CONFIG, SETTINGS, "tool_loop")
    assert llm.model == CONFIG.vllm.qwen.served_model_name
    assert llm.api_base == SETTINGS.resolve_llm_base_url(CONFIG)
    assert llm.metadata.is_function_calling_model and llm.metadata.is_chat_model
    assert llm.metadata.context_window == CONFIG.vllm.qwen.max_model_len
    assert llm.temperature == CONFIG.agent.llm.sampling.temperature
    assert llm.max_tokens == CONFIG.agent.llm.max_tokens
    extra = llm.additional_kwargs[EXTRA_BODY]
    assert extra[CHAT_TEMPLATE_KWARGS] == {"enable_thinking": False}
    assert extra["top_k"] == CONFIG.agent.llm.sampling.top_k
    assert llm.additional_kwargs["top_p"] == CONFIG.agent.llm.sampling.top_p

    answer = build_llm(CONFIG, SETTINGS, "answer")
    assert answer.additional_kwargs[EXTRA_BODY][CHAT_TEMPLATE_KWARGS] == {"enable_thinking": True}
    assert answer.max_tokens == CONFIG.agent.llm.thinking_max_tokens


def test_thinking_text_collects_thinking_blocks_only() -> None:
    message = ChatMessage(
        role="assistant",
        blocks=[ThinkingBlock(content="думаю"), TextBlock(text="ответ"), ThinkingBlock(content="ещё")],
    )
    assert thinking_text(message) == "думаю\nещё"
    assert thinking_text(ChatMessage(role="assistant", content="ответ")) is None


async def test_llm_ready_is_true_only_when_models_endpoint_answers() -> None:
    """Пока vLLM грузит модель, порт не слушается — вопрос из UI должен ждать, а не падать (2026-09-17)."""
    with respx.mock(assert_all_called=False) as router:
        route = router.get("http://llm.test/v1/models")
        route.mock(side_effect=httpx.ConnectError("connection refused"))
        assert await llm_ready(CONFIG, SETTINGS, timeout_s=1) is False
        route.mock(return_value=httpx.Response(503, json={"error": "loading"}))
        assert await llm_ready(CONFIG, SETTINGS, timeout_s=1) is False
        route.mock(return_value=httpx.Response(200, json={"object": "list", "data": [{"id": "qwen3-8b"}]}))
        assert await llm_ready(CONFIG, SETTINGS, timeout_s=1) is True
