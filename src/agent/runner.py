"""Раннер агента (6.2): цикл инструментов на `FunctionAgent` LlamaIndex, затем итоговый ответ.

Один вопрос → события (`AgentEvent`) для UI и трейса → `Answer`. Цикл: LLM без размышлений выбирает
и вызывает инструменты MCP (бюджет FR-1), в конце пишет заметки. Ответ: LLM с размышлениями
составляет текст по свидетельствам реестра с самопроверкой (FR-1) и ссылками [S#] (FR-4, разбор в 6.5);
из заметок в промпт ответа попадают только факты с псевдонимами (`agent.notes`), а черновик ответа
проверяется по свидетельствам отдельным вызовом (`agent.verify`) до нумерации ссылок.
"""

from __future__ import annotations

import datetime as dt
import logging
import time
import uuid
from collections.abc import AsyncIterator, Callable, Sequence
from typing import Any, Literal, Protocol

from llama_index.core.agent.workflow import AgentOutput, FunctionAgent, ToolCall, ToolCallResult
from llama_index.core.llms import LLM, ChatMessage
from llama_index.core.memory import ChatMemoryBuffer
from openai import APIStatusError
from pydantic import BaseModel, Field
from workflows.errors import WorkflowRuntimeError, WorkflowTimeoutError

from agent.citations import Source, cite_answer, strip_model_sources
from agent.evidence import EvidenceRegistry, ToolCallRecord
from agent.llm import thinking_text
from agent.memory import ConversationMemory, Turn, render_history
from agent.notes import facts_only
from agent.prompts import (
    BUDGET_CAVEAT,
    CHAT_SYSTEM_PROMPT,
    FORCED_SEARCH_NOTE,
    NO_ANSWER_PHRASE,
    NO_HITS_NOTES,
    answer_system_prompt,
    answer_user_message,
    chat_user_message,
    loop_system_prompt,
    loop_user_message,
    with_cached_evidence,
    with_forced_search,
    with_verification_feedback,
)
from agent.rendering import TOOL_SEARCH, cut, render_cached_evidence, render_evidence, status_text_for
from agent.rewrite import QueryRewriter, RewrittenQuery
from agent.tools import AgentTools, ToolRun
from agent.tracing import NoopTracing, QuestionHandle, Tracing
from agent.verify import AnswerVerifier, Verification, VerifyProblem
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
    queries: list[str] = Field(description="Отдельные запросы для многочастного вопроса (AC-1.1)")
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


class VerifyStarted(AgentEvent):
    kind: Literal["verify_started"] = "verify_started"
    attempt: int = Field(default=1, description="Номер черновика: 2 — после повтора шага ответа")


class AnswerVerified(AgentEvent):
    kind: Literal["answer_verified"] = "answer_verified"
    problems: list[VerifyProblem]
    corrected: bool = Field(description="Из черновика вычеркнуты неподтверждённые предложения")
    parsed: bool = Field(description="Проверка выполнена (ответ проверяющего разобран)")
    seconds: float
    attempt: int = 1


class AnswerRestarted(AgentEvent):
    kind: Literal["answer_restarted"] = "answer_restarted"
    problems: list[VerifyProblem] = Field(description="Замечания, отклонившие весь черновик")


class Answer(BaseModel):
    question: str
    text: str = Field(description="Ответ с нумерованными ссылками [1], [2] и блоком «Источники» (FR-4)")
    trace_id: str | None = Field(default=None, description="Трейс Langfuse этого вопроса (FR-8)")
    sources: list[Source] = Field(description="Источники по номерам ссылок: чанк индекса или карточка")
    unresolved_markers: list[str] = Field(description="Ссылки модели, не найденные в реестре (удалены)")
    refused: bool = Field(description="Ответ начинается с явного отказа «В документах ответа нет»")
    budget_exhausted: bool = Field(description="Бюджет вызовов инструментов исчерпан (FR-1)")
    context_exhausted: bool = Field(description="Контекст цикла для результатов инструментов исчерпан")
    rewritten_query: str | None = Field(description="Переписанный запрос, если отличался от вопроса")
    needs_search: bool = Field(description="False — ответ без инструментов (приветствие и т.п.)")
    notes: str | None
    search_queries: list[str]
    tool_calls: list[ToolCallRecord]
    fragment_aliases: list[str]
    document_aliases: list[str]
    thinking: str | None = Field(description="Размышления модели при составлении ответа")
    verification: Verification | None = Field(
        default=None, description="Итог проверки черновика по свидетельствам; None — не проверялся"
    )
    seconds: float
    loop_seconds: float
    answer_seconds: float
    verify_seconds: float = 0.0


class AnswerReady(AgentEvent):
    kind: Literal["answer_ready"] = "answer_ready"
    answer: Answer


# ---------- сессия и раннер ----------


class AgentSession:
    """Состояние одного диалога: реестр свидетельств (кэш найденных документов) и память ходов (FR-6).

    `session_id` связывает трейсы Langfuse одного диалога (FR-8); UI передаёт id диалога из БД."""

    def __init__(self, config: AppConfig, session_id: str | None = None) -> None:
        self.id = session_id or uuid.uuid4().hex
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
        tracing: Tracing | None = None,
        today: Callable[[], dt.date] = dt.date.today,
    ) -> None:
        self._config = config
        self._tools = tools
        self._llms = llms
        self._tracing: Tracing = tracing or NoopTracing()
        self._today = today

    @property
    def tracing(self) -> Tracing:
        return self._tracing

    async def aclose(self) -> None:
        """Закрывает соединение с MCP и сбрасывает трейсы."""
        await self._tools.aclose()
        self._tracing.flush()

    async def ask(self, question: str, session: AgentSession) -> Answer:
        """Вопрос → ответ без промежуточных событий (CLI, eval, тесты)."""
        answer: Answer | None = None
        async for event in self.run(question, session):
            if isinstance(event, AnswerReady):
                answer = event.answer
        assert answer is not None
        return answer

    async def run(self, question: str, session: AgentSession) -> AsyncIterator[AgentEvent]:
        """Вопрос → поток событий; последнее — `AnswerReady`. Весь вопрос — один трейс (FR-8)."""
        with self._tracing.question(question, session_id=session.id) as trace:
            async for event in self._run(question, session, trace):
                yield event

    async def _run(
        self, question: str, session: AgentSession, trace: QuestionHandle
    ) -> AsyncIterator[AgentEvent]:
        agent_settings = self._config.agent
        started = time.perf_counter()
        yield RunStarted(question=question)
        with self._tracing.step(
            "rewrite", kind="chain", input={"question": question, "turns": len(session.memory.turns)}
        ) as step:
            rewritten = await self.rewrite(question, session)
            step.update(
                output=rewritten.model_dump(exclude={"thinking"}), metadata={"thinking": rewritten.thinking}
            )
        yield QueryRewritten(
            question=question,
            query=rewritten.query,
            queries=rewritten.queries,
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
            user_message = loop_user_message(question, rewritten.query, rewritten.queries)
            cache_used = False
            if rewritten.relevant_documents and agent_settings.cached_evidence_chars:
                cached, fragments, documents = render_cached_evidence(
                    session.registry, rewritten.relevant_documents, agent_settings.cached_evidence_chars
                )
                if fragments:
                    run.note_aliases(fragments, documents)
                    run.cached_fragment_aliases = list(fragments)
                    user_message = with_cached_evidence(user_message, run.consume_context(cached))
                    cache_used = True
                    yield CacheUsed(document_aliases=documents, fragment_aliases=fragments)
            async for event in self._traced_loop(user_message, run):
                if isinstance(event, LoopNotes):
                    notes = event.text  # заметки выдаются после ограждения: оно может их заменить
                    continue
                yield event
            if not cache_used and not run.search_queries and not run.exhausted:
                # ограждение FR-1: модель закончила без единого поиска (живой диалог 2026-09-16: бытовой
                # вопрос про обед) — ищем сами по запросу и по вопросу, модель дочитывает и пишет заметки
                async for event in self._forced_search(question, rewritten.query, user_message, run):
                    if isinstance(event, LoopNotes):
                        notes = event.text
                        continue
                    yield event
            yield LoopNotes(text=notes)
        loop_seconds = time.perf_counter() - started

        answer_started = time.perf_counter()
        text = ""
        thinking: str | None = None
        evidence = ""
        facts = ""
        if rewritten.needs_search:
            evidence = self._render_evidence(run)
            # в промпт ответа из заметок попадают только факты с псевдонимами: вывод шага без размышлений
            # («вернуться не позднее 13:00») иначе становится для ответа «фактом документа»
            facts = facts_only(notes, [*run.fragment_aliases, *run.document_aliases])
        verification: Verification | None = None
        verify_seconds = 0.0
        feedback: list[tuple[str, str]] = []
        rounds = 0
        while True:
            rounds += 1
            stream = (
                self._compose(question, facts, evidence, run, feedback=feedback)
                if rewritten.needs_search
                else self._compose_chat(question, session)
            )
            with self._tracing.step(
                "answer",
                kind="chain",
                input={
                    "notes": facts,
                    "fragments": run.fragment_aliases,
                    "documents": run.document_aliases,
                    "feedback": feedback,
                },
            ) as step:
                attempts = 0
                while True:
                    attempts += 1
                    async for delta, final in stream:
                        if final is not None:
                            text = final.content or ""
                            thinking = thinking_text(final)
                        elif delta:
                            yield AnswerDelta(text=delta)
                    if text.strip() or attempts > agent_settings.answer.empty_retries:
                        break
                    # размышления съели весь лимит токенов, текста нет (живой прогон 2026-09-17): повтор
                    logger.warning("Шаг ответа вернул пустой текст (попытка %d), повторяю", attempts)
                    stream = (
                        self._compose(question, facts, evidence, run, feedback=feedback)
                        if rewritten.needs_search
                        else self._compose_chat(question, session)
                    )
                step.update(output=text, metadata={"thinking": thinking, "attempts": attempts})
            # блок «Ссылки»/«Источники» модели снимается до проверки: иначе после вычёркивания всего
            # черновика строка «Ссылки: [S6]» сходила за текст со ссылкой (живой прогон 2026-09-17)
            text = strip_model_sources(text.strip())
            if not self._should_verify(rewritten.needs_search, text, run):
                break
            yield VerifyStarted(attempt=rounds)
            with self._tracing.step("verify", kind="chain", input={"draft": text}) as step:
                verification, text = await self._verify(question, text, evidence)
                verify_seconds += verification.seconds
                step.update(
                    output={
                        "problems": [problem.model_dump() for problem in verification.problems],
                        "answer": text if verification.corrected else None,
                    },
                    metadata={
                        "corrected": verification.corrected,
                        "emptied": verification.emptied,
                        "parsed": verification.parsed,
                        "thinking": verification.thinking,
                    },
                )
            yield AnswerVerified(
                problems=verification.problems,
                corrected=verification.corrected,
                parsed=verification.parsed,
                seconds=verification.seconds,
                attempt=rounds,
            )
            if not verification.emptied or rounds > agent_settings.verify.retry_answer:
                break
            # отклонён весь черновик (живой прогон 2026-09-17: «вернуться в 12:45» от начала окна при уходе
            # в 12:45): ответ составляется заново с замечаниями проверки в промпте
            feedback = [(problem.claim, problem.reason) for problem in verification.problems]
            yield AnswerRestarted(problems=verification.problems)
        answer_seconds = time.perf_counter() - answer_started - verify_seconds
        # FR-4: маркеры [S#]/[D#] → нумерованные ссылки и блок «Источники» по реестру (6.5)
        cited = cite_answer(text, session.registry)
        body = cited.body
        if run.exhausted and BUDGET_CAVEAT not in body:
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
            trace_id=trace.trace_id,
            sources=cited.sources,
            unresolved_markers=cited.unresolved,
            refused=rewritten.needs_search and text.startswith(NO_ANSWER_PHRASE),
            budget_exhausted=run.budget_exhausted,
            context_exhausted=run.context_exhausted,
            rewritten_query=rewritten.query if rewritten.changed else None,
            needs_search=rewritten.needs_search,
            notes=notes or None,
            search_queries=run.search_queries,
            tool_calls=list(run.calls),
            fragment_aliases=list(run.fragment_aliases),
            document_aliases=list(run.document_aliases),
            thinking=thinking,
            verification=verification,
            seconds=time.perf_counter() - started,
            loop_seconds=loop_seconds,
            answer_seconds=answer_seconds,
            verify_seconds=verify_seconds,
        )
        trace.update(
            output={
                "answer": text,
                "refused": answer.refused,
                "sources": [source.line() for source in cited.sources],
            },
            metadata={
                "rewritten_query": answer.rewritten_query,
                "tool_calls": len(run.calls),
                "budget_exhausted": run.budget_exhausted,
                "unresolved": cited.unresolved,
                "verified": verification is not None and verification.parsed,
                "corrected": verification is not None and verification.corrected,
                "seconds": round(answer.seconds, 2),
            },
        )
        yield AnswerReady(answer=answer)
        # сводка старых ходов — после выдачи ответа, чтобы не задерживать его (FR-6)
        if session.memory.needs_compaction(agent_settings.memory):
            with self._tracing.step(
                "summary", kind="chain", input={"turns": len(session.memory.turns)}
            ) as step:
                await session.memory.compact(self._llms("summary"), agent_settings.memory)
                step.update(output=session.memory.summary)

    # ---------- переписывание запроса (FR-6) ----------

    async def rewrite(self, question: str, session: AgentSession) -> RewrittenQuery:
        """Только развилка и переписывание запроса, без поиска и ответа (проверки маршрутизации)."""
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

    async def _traced_loop(self, user_message: str, run: ToolRun) -> AsyncIterator[AgentEvent]:
        """Цикл инструментов как шаг трейса: заметки и вызовы — в выходе шага."""
        notes = ""
        calls_before = len(run.calls)
        with self._tracing.step("tool_loop", kind="agent", input=user_message) as step:
            async for event in self._tool_loop(user_message, run):
                if isinstance(event, LoopNotes):
                    notes = event.text
                yield event
            calls = [call.model_dump() for call in run.calls[calls_before:]]
            step.update(
                output={"notes": notes, "tool_calls": calls},
                metadata={"budget_exhausted": run.budget_exhausted, "search_queries": run.search_queries},
            )

    async def _forced_search(
        self, question: str, query: str, user_message: str, run: ToolRun
    ) -> AsyncIterator[AgentEvent]:
        """Поиск за модель, если цикл закончился без единого hybrid_search (ограждение FR-1).

        Ищем по переписанному запросу и по исходному вопросу; если что-то нашлось — цикл запускается
        ещё раз с результатами в сообщении, чтобы модель дочитала и написала заметки. Без находок
        заметки говорят об этом прямо, и ответ становится честным отказом."""
        yield LoopText(text=FORCED_SEARCH_NOTE)
        queries: list[str] = []
        for candidate in (query, question):
            if candidate.casefold() not in [item.casefold() for item in queries]:
                queries.append(candidate)
        results: list[str] = []
        first_call = len(run.calls)
        with self._tracing.step("forced_search", kind="tool", input={"queries": queries}) as step:
            for item in queries:
                arguments = {"query": item}
                yield ToolStarted(
                    tool=TOOL_SEARCH,
                    arguments=arguments,
                    status=status_text_for(TOOL_SEARCH, arguments, run.registry),
                )
                before = len(run.calls)
                text = await self._tools.call(TOOL_SEARCH, arguments, run)
                record = run.calls[-1] if len(run.calls) > before else None
                if record is None:
                    # бюджет или контекст исчерпаны: вызов не выполнен, текст объясняет причину
                    yield ToolFinished(tool=TOOL_SEARCH, summary=cut(text, 120), ok=False, seconds=0.0)
                    continue
                yield ToolFinished(
                    tool=TOOL_SEARCH, summary=record.summary, ok=record.ok, seconds=record.seconds
                )
                if record.ok:
                    results.append(text)
            found = [alias for record in run.calls[first_call:] for alias in record.fragment_aliases]
            step.update(output={"fragments": found})
        if not found:
            yield LoopNotes(text=NO_HITS_NOTES)
            return
        async for event in self._traced_loop(with_forced_search(user_message, results), run):
            yield event

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
        except (APIStatusError, WorkflowRuntimeError) as exc:
            # модель отклонила запрос (например, переполнен контекст) — отвечаем по собранному
            logger.error("Цикл инструментов прерван ошибкой модели: %s", exc)
            run.context_exhausted = True
            notes = notes or "Цикл поиска прерван ошибкой модели; отвечаю по уже найденному."
        yield LoopNotes(text=notes)

    # ---------- итоговый ответ ----------

    def _render_evidence(self, run: ToolRun) -> str:
        return render_evidence(
            run.registry,
            run.fragment_aliases,
            run.document_aliases,
            self._config.agent.answer,
            cached_aliases=run.cached_fragment_aliases,
        )

    async def _compose(
        self,
        question: str,
        facts: str,
        evidence: str,
        run: ToolRun,
        feedback: Sequence[tuple[str, str]] = (),
    ) -> AsyncIterator[tuple[str, ChatMessage | None]]:
        """Стрим итогового ответа: (дельта, None) по ходу и ("", сообщение) в конце."""
        user_message = answer_user_message(question, facts, evidence)
        if feedback:
            user_message = with_verification_feedback(user_message, feedback)
        messages = [
            ChatMessage(role="system", content=answer_system_prompt(budget_exhausted=run.budget_exhausted)),
            ChatMessage(role="user", content=user_message),
        ]
        async for item in self._stream_answer(messages):
            yield item

    # ---------- проверка черновика по свидетельствам ----------

    def _should_verify(self, needs_search: bool, draft: str, run: ToolRun) -> bool:
        """Проверяются ответы по документам со свидетельствами; отказ и пустой черновик — нет."""
        if not self._config.agent.verify.enabled or not needs_search or not draft.strip():
            return False
        if draft.lstrip().startswith(NO_ANSWER_PHRASE):
            return False
        return bool(run.fragment_aliases or run.document_aliases)

    async def _verify(self, question: str, draft: str, evidence: str) -> tuple[Verification, str]:
        verifier = AnswerVerifier(self._llms("verify"), self._config.agent.verify)
        return await verifier.verify(question, draft, evidence)

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
