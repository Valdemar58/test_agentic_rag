"""Инструменты MCP для агента (6.2): схемы через llama-index-tools-mcp, вызовы через `BasicMCPClient`.

Каждый инструмент MCP оборачивается в `FunctionTool` LlamaIndex: бюджет вызовов (FR-1, AC-1.3),
замена псевдонимов D#/S# на UUID, компактное представление результата для LLM и запись свидетельств
в реестр. Ошибка инструмента возвращается LLM текстом, а не исключением: цикл продолжается.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Protocol, cast

from llama_index.core.tools import FunctionTool, ToolMetadata
from llama_index.tools.mcp import BasicMCPClient, McpToolSpec
from mcp.types import CallToolResult, TextContent, Tool
from pydantic import BaseModel, Field

from agent.evidence import FILTERS_ARGUMENT, STATUS_RU, EvidenceRegistry, ToolCallRecord
from agent.rendering import TOOL_RELATED, TOOL_SEARCH, cut, render
from common.config import ToolOutputSettings

logger = logging.getLogger(__name__)

BUDGET_EXHAUSTED = (
    "Бюджет вызовов инструментов ({limit}) исчерпан: вызов не выполнен. "
    "Заверши работу — напиши заметки для ответа по уже найденному."
)
REPEATED_CALL = (
    "Этот вызов уже выполнялся с теми же аргументами; повторно инструмент не вызван (бюджет не тратится). "
    "Прежний результат:\n{text}"
)
CONTEXT_EXHAUSTED = (
    "Контекст для результатов инструментов исчерпан: вызов не выполнен. "
    "Заверши работу — напиши заметки для ответа по уже найденному."
)
CONTEXT_TIGHT = "(результат сокращён: контекст для результатов инструментов почти исчерпан)"
RELATION_TYPE_ARGUMENT = "relation_type"
ERROR_SUMMARY_CHARS = 200


class ToolResultData(BaseModel):
    is_error: bool
    text: str = Field(description="Текстовое содержимое ответа (для ошибок — их текст)")
    structured: dict[str, Any] | None = Field(description="structuredContent инструмента")


class ToolTransport(Protocol):
    """Соединение с MCP-сервером: список инструментов и вызов; в тестах — сервер в памяти."""

    async def list_tools(self) -> list[Tool]: ...

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> ToolResultData: ...

    async def aclose(self) -> None: ...


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

    async def aclose(self) -> None:
        """Закрывает HTTP-клиент MCP до остановки цикла событий (иначе httpx ругается при сборке мусора)."""
        await self._client.http_client.aclose()


@dataclass
class ToolRun:
    """Состояние одного вопроса: бюджет, записи вызовов и свидетельства этого прогона."""

    registry: EvidenceRegistry
    max_calls: int
    limits: ToolOutputSettings
    calls: list[ToolCallRecord] = field(default_factory=list)
    budget_exhausted: bool = False
    context_exhausted: bool = False
    context_chars: int = 0
    fragment_aliases: list[str] = field(default_factory=list)
    document_aliases: list[str] = field(default_factory=list)
    cached_fragment_aliases: list[str] = field(
        default_factory=list, metadata={"doc": "Фрагменты из кэша сессии, а не из вызовов этого вопроса"}
    )
    seen: dict[tuple[str, str], str] = field(default_factory=dict, repr=False)

    @property
    def search_queries(self) -> list[str]:
        return [record.query for record in self.calls if record.query]

    @property
    def exhausted(self) -> bool:
        """Бюджет вызовов или контекст исчерпаны — ответ может быть неполным (FR-1)."""
        return self.budget_exhausted or self.context_exhausted

    def remaining_context(self) -> int:
        return max(0, self.limits.loop_context_chars - self.context_chars)

    def consume_context(self, text: str) -> str:
        """Учитывает текст в контексте цикла; сверх остатка — обрезает с пометкой."""
        remaining = self.remaining_context()
        if len(text) > remaining:
            text = f"{cut(text, max(remaining, 0))}\n{CONTEXT_TIGHT}"
        self.context_chars += len(text)
        return text

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


def normalize_status(value: object) -> object:
    """«действует», «Действующий», «отменён», «проект» → коды статусов индекса; коды и прочее — как есть."""
    text = str(value).strip().casefold()
    if text in STATUS_RU:
        return text
    for code, word in STATUS_RU.items():
        if text.startswith(word[:5]):
            return code
    return value


def _call_key(name: str, arguments: dict[str, Any]) -> tuple[str, str]:
    return name, json.dumps(arguments, ensure_ascii=False, sort_keys=True, default=str)


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

    @property
    def transport(self) -> ToolTransport:
        return self._transport

    async def aclose(self) -> None:
        await self._transport.aclose()

    def bind(self, run: ToolRun) -> list[FunctionTool]:
        if not self._specs:
            raise RuntimeError("инструменты не загружены: вызовите load()")
        return [self._bind_one(tool, run) for tool in self._specs]

    async def call(self, name: str, arguments: dict[str, Any], run: ToolRun) -> str:
        """Вызов инструмента раннером в обход модели: те же бюджет, запись вызова и реестр свидетельств."""
        return await self._call(name, arguments, run)

    def _normalize(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Аргументы, которые модель пишет «почти правильно», не должны стоить вызова из бюджета.

        Живой диалог 2026-09-16: `statuses: ["действует"]` и лишний `top_k` у get_document_content —
        два вызова ушли на ошибки валидации. Неизвестные аргументы отбрасываются по схеме инструмента,
        русские названия статусов переводятся в коды индекса."""
        schema = self._schemas.get(name)
        if schema is not None:
            unknown = [key for key in arguments if key not in schema.model_fields]
            for key in unknown:
                logger.info("Инструмент %s: аргумент %s не по схеме, отброшен", name, key)
                arguments.pop(key)
        filters = arguments.get(FILTERS_ARGUMENT)
        if name == TOOL_SEARCH and isinstance(filters, dict) and isinstance(filters.get("statuses"), list):
            filters = dict(filters)
            filters["statuses"] = [normalize_status(item) for item in filters["statuses"]]
            arguments[FILTERS_ARGUMENT] = filters
        return arguments

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
        if run.remaining_context() < run.limits.context_reserve_chars:
            run.context_exhausted = True
            logger.info("Контекст цикла исчерпан (%d символов), %s не вызван", run.context_chars, name)
            return CONTEXT_EXHAUSTED
        arguments = run.registry.resolve_arguments(self._normalize(name, _clean(kwargs)))
        if name == TOOL_RELATED:
            # все связи с типами и так видны в ответе; фильтр по типу только плодил повторные вызовы
            # (живой прогон 2026-09-16: 7 вызовов подряд с разными типами по одному документу)
            arguments.pop(RELATION_TYPE_ARGUMENT, None)
        key = _call_key(name, arguments)
        if key in run.seen:
            logger.info("Инструмент %s: повторный вызов с теми же аргументами, отдаю прежний результат", name)
            return run.consume_context(REPEATED_CALL.format(text=run.seen[key]))
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
        text = run.consume_context(rendered.text)
        run.seen[key] = text
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
        return text
