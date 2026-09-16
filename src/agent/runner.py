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

from agent.citations import Source, cite_answer
from agent.evidence import EvidenceRegistry, ToolCallRecord
from agent.llm import thinking_text
from agent.memory import ConversationMemory, Turn, render_history
from agent.prompts import (
    BUDGET_CAVEAT,
    CHAT_SYSTEM_PROMPT,
    NO_ANSWER_PHRASE,
    answer_system_prompt,
    answer_user_message,
    chat_user_message,
    loop_system_prompt,
    loop_user_message,
    with_cached_evidence,
)
from agent.rendering import render_cached_evidence, render_evidence, status_text_for
from agent.rewrite import QueryRewriter, RewrittenQuery
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


class QueryRewritten(AgentEvent):
    kind: Literal["query_rewritten"] = "query_rewritten"
    question: str
    query: str = Field(description="Самодостаточный поисковый запрос (FR-6)")
    changed: bool
    needs_search: bool
    relevant_documents: list[str]
    abbreviations: list[str]
    reason: str | None


class CacheUsed(AgentEvent):
    kind: Literal["cache_used"] = "cache_used"
    document_aliases: list[str] = Field(description="Документы из кэша сессии, поданные в цикл (FR-6)")
    fragment_aliases: list[str]


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
    text: str = Field(description="Ответ с нумерованными ссылками [1], [2] и блоком «Источники» (FR-4)")
    sources: list[Source] = Field(description="Источники по номерам ссылок: чанк индекса или карточка")
    unresolved_markers: list[str] = Field(description="Ссылки модели, не найденные в реестре (удалены)")
    refused: bool = Field(description="Ответ начинается с явного отказа «В документах ответа нет»")
    budget_exhausted: bool
    rewritten_query: str | None = Field(description="Переписанный запрос, если отличался от вопроса")
    needs_search: bool = Field(description="False — ответ без инструментов (приветствие и т.п.)")
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
    """Состояние одного диалога: реестр свидетельств (кэш найденных документов) и память ходов (FR-6)."""

    def __init__(self, config: AppConfig) -> None:
        self.registry = EvidenceRegistry(config.agent.session_document_cache)
        self.memory = ConversationMemory()


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
        rewritten = await self._rewrite(question, session)
        yield QueryRewritten(
            question=question,
            query=rewritten.query,
            changed=rewritten.changed,
            needs_search=rewritten.needs_search,
            relevant_documents=rewritten.relevant_documents,
            abbreviations=rewritten.abbreviations,
            reason=rewritten.reason,
        )
        run = ToolRun(
            registry=session.registry,
            max_calls=agent_settings.max_tool_calls,
            limits=agent_settings.tool_output,
        )
        notes = ""
        if rewritten.needs_search:
            user_message = loop_user_message(question, rewritten.query)
            if rewritten.relevant_documents and agent_settings.cached_evidence_chars:
                cached, fragments, documents = render_cached_evidence(
                    session.registry, rewritten.relevant_documents, agent_settings.cached_evidence_chars
                )
                if fragments:
                    run.note_aliases(fragments, documents)
                    user_message = with_cached_evidence(user_message, cached)
                    yield CacheUsed(document_aliases=documents, fragment_aliases=fragments)
            async for event in self._tool_loop(user_message, run):
                if isinstance(event, LoopNotes):
                    notes = event.text
                yield event
        loop_seconds = time.perf_counter() - started

        answer_started = time.perf_counter()
        text = ""
        thinking: str | None = None
        stream = (
            self._compose(question, notes, run)
            if rewritten.needs_search
            else self._compose_chat(question, session)
        )
        async for delta, final in stream:
            if final is not None:
                text = final.content or ""
                thinking = thinking_text(final)
            elif delta:
                yield AnswerDelta(text=delta)
        # FR-4: маркеры [S#]/[D#] → нумерованные ссылки и блок «Источники» по реестру (6.5)
        cited = cite_answer(text, session.registry)
        body = cited.body
        if run.budget_exhausted and BUDGET_CAVEAT not in body:
            body = f"{body}\n\n{BUDGET_CAVEAT}".strip()
        text = f"{body}\n\n{cited.sources_block}" if cited.sources else body
        session.registry.trim()
        rewritten_query = rewritten.query if rewritten.changed else None
        session.memory.add(
            Turn(
                question=question,
                answer=text,
                rewritten_query=rewritten_query,
                document_aliases=list(run.document_aliases),
            )
        )
        answer = Answer(
            question=question,
            text=text,
            sources=cited.sources,
            unresolved_markers=cited.unresolved,
            refused=rewritten.needs_search and text.startswith(NO_ANSWER_PHRASE),
            budget_exhausted=run.budget_exhausted,
            rewritten_query=rewritten.query if rewritten.changed else None,
            needs_search=rewritten.needs_search,
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
        # сводка старых ходов — после выдачи ответа, чтобы не задерживать его (FR-6)
        if session.memory.needs_compaction(agent_settings.memory):
            await session.memory.compact(self._llms("summary"), agent_settings.memory)

    # ---------- переписывание запроса (FR-6) ----------

    async def _rewrite(self, question: str, session: AgentSession) -> RewrittenQuery:
        rewriter = QueryRewriter(self._llms("rewrite"), self._config.agent.rewrite)
        return await rewriter.rewrite(
            question, session.memory.turns, session.registry.documents(), session.memory.summary
        )

    def _history(self, session: AgentSession) -> str:
        rewrite = self._config.agent.rewrite
        return render_history(
            session.memory.recent(rewrite.history_turns),
            answer_chars=rewrite.answer_chars,
            summary=session.memory.summary,
        )

    # ---------- цикл инструментов ----------

    async def _tool_loop(self, user_message: str, run: ToolRun) -> AsyncIterator[AgentEvent]:
        agent_settings = self._config.agent
        agent = FunctionAgent(
            tools=list(self._tools.bind(run)),
            llm=self._llms("tool_loop"),
            system_prompt=loop_system_prompt(agent_settings.max_tool_calls, self._today().isoformat()),
            timeout=agent_settings.loop_timeout_s,
        )
        handler = agent.run(
            user_msg=user_message,
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
        async for item in self._stream_answer(messages):
            yield item

    async def _compose_chat(
        self, question: str, session: AgentSession
    ) -> AsyncIterator[tuple[str, ChatMessage | None]]:
        """Ответ без инструментов: приветствие, вопрос о возможностях, переформулировка прошлого ответа."""
        messages = [
            ChatMessage(role="system", content=CHAT_SYSTEM_PROMPT),
            ChatMessage(role="user", content=chat_user_message(question, self._history(session))),
        ]
        async for item in self._stream_answer(messages):
            yield item

    async def _stream_answer(
        self, messages: list[ChatMessage]
    ) -> AsyncIterator[tuple[str, ChatMessage | None]]:
        llm = self._llms("answer")
        last: ChatMessage | None = None
        async for response in await llm.astream_chat(messages):
            if response.delta:
                yield response.delta, None
            last = response.message
        yield "", last or ChatMessage(role="assistant", content="")
