"""Инструменты MCP для агента (6.2): схемы через llama-index-tools-mcp, вызовы через `BasicMCPClient`.

Каждый инструмент MCP оборачивается в `FunctionTool` LlamaIndex: бюджет вызовов (FR-1, AC-1.3),
замена псевдонимов D#/S# на UUID, компактное представление результата для LLM и запись свидетельств
в реестр. Ошибка инструмента возвращается LLM текстом, а не исключением: цикл продолжается.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Protocol, cast

from llama_index.core.tools import FunctionTool, ToolMetadata
from llama_index.tools.mcp import BasicMCPClient, McpToolSpec
from mcp.types import CallToolResult, TextContent, Tool
from pydantic import BaseModel, Field

from agent.evidence import EvidenceRegistry, ToolCallRecord
from agent.rendering import TOOL_SEARCH, cut, render
from common.config import ToolOutputSettings

logger = logging.getLogger(__name__)

BUDGET_EXHAUSTED = (
    "Бюджет вызовов инструментов ({limit}) исчерпан: вызов не выполнен. "
    "Заверши работу — напиши заметки для ответа по уже найденному."
)
ERROR_SUMMARY_CHARS = 200


class ToolResultData(BaseModel):
    is_error: bool
    text: str = Field(description="Текстовое содержимое ответа (для ошибок — их текст)")
    structured: dict[str, Any] | None = Field(description="structuredContent инструмента")


class ToolTransport(Protocol):
    """Соединение с MCP-сервером: список инструментов и вызов; в тестах — сервер в памяти."""

    async def list_tools(self) -> list[Tool]: ...

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> ToolResultData: ...


def result_data(result: CallToolResult) -> ToolResultData:
    text = "\n".join(item.text for item in result.content if isinstance(item, TextContent))
    return ToolResultData(is_error=bool(result.isError), text=text, structured=result.structuredContent)


class McpTransport:
    """`BasicMCPClient` из llama-index-tools-mcp поверх streamable-http (сессия на каждый вызов)."""

    def __init__(self, url: str, *, timeout_s: float) -> None:
        self._client = BasicMCPClient(url, timeout=int(timeout_s))

    async def list_tools(self) -> list[Tool]:
        return list((await self._client.list_tools()).tools)

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> ToolResultData:
        return result_data(await self._client.call_tool(name, arguments))


@dataclass
class ToolRun:
    """Состояние одного вопроса: бюджет, записи вызовов и свидетельства этого прогона."""

    registry: EvidenceRegistry
    max_calls: int
    limits: ToolOutputSettings
    calls: list[ToolCallRecord] = field(default_factory=list)
    budget_exhausted: bool = False
    fragment_aliases: list[str] = field(default_factory=list)
    document_aliases: list[str] = field(default_factory=list)

    @property
    def search_queries(self) -> list[str]:
        return [record.query for record in self.calls if record.query]

    def note_aliases(self, fragments: list[str], documents: list[str]) -> None:
        for alias in fragments:
            if alias not in self.fragment_aliases:
                self.fragment_aliases.append(alias)
        for alias in documents:
            if alias not in self.document_aliases:
                self.document_aliases.append(alias)


def _clean(arguments: dict[str, Any]) -> dict[str, Any]:
    """LLM и схема подставляют null в необязательные поля — MCP их не ждёт."""
    return {key: value for key, value in arguments.items() if value is not None}


class AgentTools:
    """Инструменты MCP как `FunctionTool` LlamaIndex, привязанные к состоянию прогона."""

    def __init__(self, transport: ToolTransport) -> None:
        self._transport = transport
        self._specs: list[Tool] = []
        self._schemas: dict[str, type[BaseModel]] = {}

    async def load(self) -> None:
        """Читает список инструментов и строит pydantic-схемы аргументов из их JSON Schema."""
        spec = McpToolSpec(cast(Any, self._transport))
        self._specs = await self._transport.list_tools()
        self._schemas = {
            tool.name: spec.create_model_from_json_schema(tool.inputSchema, model_name=f"{tool.name}_Schema")
            for tool in self._specs
        }

    @property
    def names(self) -> list[str]:
        return [tool.name for tool in self._specs]

    def bind(self, run: ToolRun) -> list[FunctionTool]:
        if not self._specs:
            raise RuntimeError("инструменты не загружены: вызовите load()")
        return [self._bind_one(tool, run) for tool in self._specs]

    def _bind_one(self, tool: Tool, run: ToolRun) -> FunctionTool:
        name = tool.name

        async def call(**kwargs: Any) -> str:
            return await self._call(name, kwargs, run)

        metadata = ToolMetadata(name=name, description=tool.description or "", fn_schema=self._schemas[name])
        return FunctionTool.from_defaults(async_fn=call, tool_metadata=metadata)

    async def _call(self, name: str, kwargs: dict[str, Any], run: ToolRun) -> str:
        if len(run.calls) >= run.max_calls:
            run.budget_exhausted = True
            logger.info("Бюджет вызовов (%d) исчерпан, %s не вызван", run.max_calls, name)
            return BUDGET_EXHAUSTED.format(limit=run.max_calls)
        arguments = run.registry.resolve_arguments(_clean(kwargs))
        query = str(arguments.get("query")) if name == TOOL_SEARCH and arguments.get("query") else None
        started = time.perf_counter()
        try:
            result = await self._transport.call_tool(name, arguments)
        except Exception as exc:
            logger.exception("Инструмент %s: ошибка транспорта", name)
            run.calls.append(
                ToolCallRecord(
                    name=name,
                    arguments=arguments,
                    ok=False,
                    summary=f"Ошибка: {cut(str(exc), ERROR_SUMMARY_CHARS)}",
                    seconds=time.perf_counter() - started,
                    query=query,
                )
            )
            return f"Ошибка инструмента {name}: {exc}"
        seconds = time.perf_counter() - started
        if result.is_error or result.structured is None:
            text = result.text or "пустой ответ"
            run.calls.append(
                ToolCallRecord(
                    name=name,
                    arguments=arguments,
                    ok=False,
                    summary=f"Ошибка: {cut(text, ERROR_SUMMARY_CHARS)}",
                    seconds=seconds,
                    query=query,
                )
            )
            return f"Ошибка инструмента {name}: {text}"
        rendered = render(name, arguments, result.structured, run.registry, run.limits)
        run.note_aliases(rendered.fragment_aliases, rendered.document_aliases)
        run.calls.append(
            ToolCallRecord(
                name=name,
                arguments=arguments,
                ok=True,
                summary=rendered.summary,
                seconds=seconds,
                query=query,
                fragment_aliases=rendered.fragment_aliases,
                document_aliases=rendered.document_aliases,
            )
        )
        return rendered.text
