"""Фейки для тестов агента: LLM со сценарием шагов и транспорт инструментов к MCP-серверу в памяти."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from fastmcp import Client, FastMCP
from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    ChatResponseAsyncGen,
    ChatResponseGen,
    CompletionResponse,
    LLMMetadata,
)
from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core.tools import BaseTool, ToolSelection
from mcp.types import Tool
from pydantic import Field

from agent.tools import ToolResultData

Step = str | list[ToolSelection]


def tool_step(name: str, **kwargs: Any) -> list[ToolSelection]:
    return [ToolSelection(tool_id=f"call-{name}-{id(kwargs)}", tool_name=name, tool_kwargs=kwargs)]


class ScriptedLLM(FunctionCallingLLM):
    """LLM с заранее заданными шагами: вызов инструмента или текст; записывает входы каждого шага."""

    steps: list[Any] = Field(default_factory=list)
    inputs: list[list[ChatMessage]] = Field(default_factory=list)
    fallback: str = "Заметки: закончил."

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=16384,
            num_output=2048,
            is_chat_model=True,
            is_function_calling_model=True,
            model_name="scripted",
        )

    def _next(self, messages: Sequence[ChatMessage]) -> ChatResponse:
        self.inputs.append(list(messages))
        step: Step = self.steps.pop(0) if self.steps else self.fallback
        if isinstance(step, str):
            message = ChatMessage(role="assistant", content=step)
        else:
            message = ChatMessage(
                role="assistant",
                content="",
                additional_kwargs={"tool_calls": [selection.model_dump() for selection in step]},
            )
        return ChatResponse(message=message, delta=message.content or "")

    def get_tool_calls_from_response(
        self, response: ChatResponse, error_on_no_tool_call: bool = True, **kwargs: Any
    ) -> list[ToolSelection]:
        raw = response.message.additional_kwargs.get("tool_calls", [])
        return [ToolSelection.model_validate(item) for item in raw]

    def _prepare_chat_with_tools(
        self,
        tools: Sequence[BaseTool],
        user_msg: str | ChatMessage | None = None,
        chat_history: list[ChatMessage] | None = None,
        verbose: bool = False,
        allow_parallel_tool_calls: bool = False,
        tool_required: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        messages = list(chat_history or [])
        if isinstance(user_msg, str):
            messages.append(ChatMessage(role="user", content=user_msg))
        elif user_msg is not None:
            messages.append(user_msg)
        return {"messages": messages}

    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        return self._next(messages)

    async def achat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        return self._next(messages)

    def stream_chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponseGen:
        response = self._next(messages)

        def generate() -> ChatResponseGen:
            yield response

        return generate()

    async def astream_chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponseAsyncGen:
        response = self._next(messages)

        async def generate() -> ChatResponseAsyncGen:
            yield response

        return generate()

    def complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponse:
        raise NotImplementedError

    def stream_complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> Any:
        raise NotImplementedError

    async def acomplete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponse:
        raise NotImplementedError

    async def astream_complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> Any:
        raise NotImplementedError


class InMemoryTransport:
    """Транспорт к FastMCP-серверу в памяти (без сети): тот же контракт, что у McpTransport."""

    def __init__(self, server: FastMCP) -> None:
        self._server = server

    async def list_tools(self) -> list[Tool]:
        async with Client(self._server) as client:
            return list(await client.list_tools())

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> ToolResultData:
        async with Client(self._server) as client:
            result = await client.call_tool(name, arguments, raise_on_error=False)
        text = "\n".join(getattr(item, "text", "") for item in result.content)
        return ToolResultData(is_error=result.is_error, text=text, structured=result.structured_content)
