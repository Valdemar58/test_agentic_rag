"""Сборка агента из конфига и окружения: LLM по ролям, инструменты MCP по `MCP_URL`."""

from __future__ import annotations

from llama_index.core.llms import LLM

from agent.llm import build_llm
from agent.runner import AgentRunner
from agent.tools import AgentTools, McpTransport
from agent.tracing import build_tracing
from common.config import AppConfig, LlmRole
from common.settings import Settings

ROLES: tuple[LlmRole, ...] = ("rewrite", "tool_loop", "answer", "verify", "summary")


class RoleLlms:
    """LLM для каждой роли агента, собранные один раз."""

    def __init__(self, config: AppConfig, settings: Settings) -> None:
        self._llms: dict[LlmRole, LLM] = {role: build_llm(config, settings, role) for role in ROLES}

    def __call__(self, role: LlmRole) -> LLM:
        return self._llms[role]


async def build_runner(config: AppConfig, settings: Settings) -> AgentRunner:
    """Раннер на живом MCP-сервере (`MCP_URL` или порт из конфига) и vLLM; Langfuse — по флагу окружения."""
    tracing = build_tracing(settings, config.langfuse)
    transport = McpTransport(settings.resolve_mcp_url(config), timeout_s=config.agent.tool_timeout_s)
    tools = AgentTools(transport)
    await tools.load()
    return AgentRunner(config, tools, RoleLlms(config, settings), tracing=tracing)
