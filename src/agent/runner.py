"""Раннер агента (6.2): цикл инструментов на `FunctionAgent` LlamaIndex, затем итоговый ответ.

Один вопрос → события (`AgentEvent`) для UI и трейса → `Answer`. Цикл: LLM без размышлений выбирает
и вызывает инструменты MCP (бюджет FR-1), в конце пишет заметки. Ответ: LLM с размышлениями
составляет текст по свидетельствам реестра с самопроверкой (FR-1) и ссылками [S#] (FR-4, разбор в 6.5).
"""

from __future__ import annotations

import datetime as dt
import logging
import time
from collections.abc import AsyncIterator, Callable
from typing import Any, Literal, Protocol

from llama_index.core.agent.workflow import AgentOutput, FunctionAgent, ToolCall, ToolCallResult
from llama_index.core.llms import LLM, ChatMessage
from llama_index.core.memory import ChatMemoryBuffer
from pydantic import BaseModel, Field
from workflows.errors import WorkflowTimeoutError

from agent.evidence import EvidenceRegistry, ToolCallRecord
from agent.llm import thinking_text
from agent.prompts import (
    BUDGET_CAVEAT,
    NO_ANSWER_PHRASE,
    answer_system_prompt,
    answer_user_message,
    loop_system_prompt,
)
from agent.rendering import render_evidence, status_text_for
from agent.tools import AgentTools, ToolRun
from common.config import AppConfig, LlmRole

logger = logging.getLogger(__name__)

# Предохранитель шагов LLM в цикле сверх бюджета инструментов: после исчерпания бюджета модели остаётся
# несколько шагов, чтобы написать заметки; дальше LlamaIndex сам просит итоговое сообщение
EXTRA_LLM_STEPS = 3
UNLIMITED_TOKENS = 10**9


class LlmFactory(Protocol):
    def __call__(self, role: LlmRole) -> LLM: ...


# ---------- события ----------


class AgentEvent(BaseModel):
    kind: str


class RunStarted(AgentEvent):
    kind: Literal["run_started"] = "run_started"
    question: str


class ToolStarted(AgentEvent):
    kind: Literal["tool_started"] = "tool_started"
    tool: str
    arguments: dict[str, Any]
    status: str = Field(description="Человекочитаемый статус шага (FR-7)")


class ToolFinished(AgentEvent):
    kind: Literal["tool_finished"] = "tool_finished"
    tool: str
    summary: str
    ok: bool
    seconds: float


class LoopText(AgentEvent):
    kind: Literal["loop_text"] = "loop_text"
    text: str = Field(description="Короткая реплика агента перед вызовом инструментов")


class LoopNotes(AgentEvent):
    kind: Literal["loop_notes"] = "loop_notes"
    text: str


class AnswerDelta(AgentEvent):
    kind: Literal["answer_delta"] = "answer_delta"
    text: str


class Answer(BaseModel):
    question: str
    text: str
    refused: bool = Field(description="Ответ начинается с явного отказа «В документах ответа нет»")
    budget_exhausted: bool
    notes: str | None
    search_queries: list[str]
    tool_calls: list[ToolCallRecord]
    fragment_aliases: list[str]
    document_aliases: list[str]
    thinking: str | None = Field(description="Размышления модели при составлении ответа")
    seconds: float
    loop_seconds: float
    answer_seconds: float


class AnswerReady(AgentEvent):
    kind: Literal["answer_ready"] = "answer_ready"
    answer: Answer


# ---------- сессия и раннер ----------


class AgentSession:
    """Состояние одного диалога: реестр свидетельств (кэш найденных документов, FR-6)."""

    def __init__(self, config: AppConfig) -> None:
        self.registry = EvidenceRegistry(config.agent.session_document_cache)


def _loop_memory() -> ChatMemoryBuffer:
    # Свой буфер без tiktoken: умолчание LlamaIndex качает словарь токенайзера из сети (NFR-5),
    # а состав контекста цикла и так ограничивается раннером
    return ChatMemoryBuffer(token_limit=UNLIMITED_TOKENS, tokenizer_fn=str.split)


class AgentRunner:
    def __init__(
        self,
        config: AppConfig,
        tools: AgentTools,
        llms: LlmFactory,
        *,
        today: Callable[[], dt.date] = dt.date.today,
    ) -> None:
        self._config = config
        self._tools = tools
        self._llms = llms
        self._today = today

    async def ask(self, question: str, session: AgentSession) -> Answer:
        """Вопрос → ответ без промежуточных событий (CLI, eval, тесты)."""
        answer: Answer | None = None
        async for event in self.run(question, session):
            if isinstance(event, AnswerReady):
                answer = event.answer
        assert answer is not None
        return answer

    async def run(self, question: str, session: AgentSession) -> AsyncIterator[AgentEvent]:
        """Вопрос → поток событий; последнее — `AnswerReady`."""
        agent_settings = self._config.agent
        started = time.perf_counter()
        yield RunStarted(question=question)
        run = ToolRun(
            registry=session.registry,
            max_calls=agent_settings.max_tool_calls,
            limits=agent_settings.tool_output,
        )
        notes = ""
        async for event in self._tool_loop(question, run):
            if isinstance(event, LoopNotes):
                notes = event.text
            yield event
        loop_seconds = time.perf_counter() - started

        answer_started = time.perf_counter()
        text = ""
        thinking: str | None = None
        async for delta, final in self._compose(question, notes, run):
            if final is not None:
                text = final.content or ""
                thinking = thinking_text(final)
            elif delta:
                yield AnswerDelta(text=delta)
        text = text.strip()
        if run.budget_exhausted and BUDGET_CAVEAT not in text:
            text = f"{text}\n\n{BUDGET_CAVEAT}".strip()
        session.registry.trim()
        answer = Answer(
            question=question,
            text=text,
            refused=text.startswith(NO_ANSWER_PHRASE),
            budget_exhausted=run.budget_exhausted,
            notes=notes or None,
            search_queries=run.search_queries,
            tool_calls=list(run.calls),
            fragment_aliases=list(run.fragment_aliases),
            document_aliases=list(run.document_aliases),
            thinking=thinking,
            seconds=time.perf_counter() - started,
            loop_seconds=loop_seconds,
            answer_seconds=time.perf_counter() - answer_started,
        )
        yield AnswerReady(answer=answer)

    # ---------- цикл инструментов ----------

    async def _tool_loop(self, question: str, run: ToolRun) -> AsyncIterator[AgentEvent]:
        agent_settings = self._config.agent
        agent = FunctionAgent(
            tools=list(self._tools.bind(run)),
            llm=self._llms("tool_loop"),
            system_prompt=loop_system_prompt(agent_settings.max_tool_calls, self._today().isoformat()),
            timeout=agent_settings.loop_timeout_s,
        )
        handler = agent.run(
            user_msg=question,
            memory=_loop_memory(),
            max_iterations=agent_settings.max_tool_calls + EXTRA_LLM_STEPS,
            early_stopping_method="generate",
        )
        notes = ""
        try:
            async for event in handler.stream_events():
                if isinstance(event, ToolCall):
                    yield ToolStarted(
                        tool=event.tool_name,
                        arguments=event.tool_kwargs,
                        status=status_text_for(event.tool_name, event.tool_kwargs, run.registry),
                    )
                elif isinstance(event, ToolCallResult):
                    record = run.calls[-1] if run.calls and run.calls[-1].name == event.tool_name else None
                    if record is None:
                        # вызов не дошёл до инструмента: бюджет исчерпан или инструмент не найден
                        yield ToolFinished(
                            tool=event.tool_name,
                            summary=str(event.tool_output.content)[:120],
                            ok=not event.tool_output.is_error,
                            seconds=0.0,
                        )
                    else:
                        yield ToolFinished(
                            tool=event.tool_name, summary=record.summary, ok=record.ok, seconds=record.seconds
                        )
                elif isinstance(event, AgentOutput):
                    content = (event.response.content or "").strip()
                    if event.tool_calls and content:
                        yield LoopText(text=content)
                    elif not event.tool_calls:
                        notes = content
            await handler
        except WorkflowTimeoutError:
            logger.warning("Цикл инструментов прерван по времени (%s с)", agent_settings.loop_timeout_s)
            notes = notes or "Цикл поиска прерван по времени; отвечаю по уже найденному."
        yield LoopNotes(text=notes)

    # ---------- итоговый ответ ----------

    async def _compose(
        self, question: str, notes: str, run: ToolRun
    ) -> AsyncIterator[tuple[str, ChatMessage | None]]:
        """Стрим итогового ответа: (дельта, None) по ходу и ("", сообщение) в конце."""
        evidence = render_evidence(
            run.registry, run.fragment_aliases, run.document_aliases, self._config.agent.answer
        )
        messages = [
            ChatMessage(role="system", content=answer_system_prompt(budget_exhausted=run.budget_exhausted)),
            ChatMessage(role="user", content=answer_user_message(question, notes, evidence)),
        ]
        llm = self._llms("answer")
        last: ChatMessage | None = None
        async for response in await llm.astream_chat(messages):
            if response.delta:
                yield response.delta, None
            last = response.message
        yield "", last or ChatMessage(role="assistant", content="")
