"""Поток вопроса для UI (FR-7): события агента → шаги, стрим токенов и готовый ответ с цитатами.

Не зависит от Chainlit: `ui.app` переводит события в шаги и сообщения, тесты читают их напрямую.
Первый видимый сигнал (AC-7.1) — шаг «Разбираю вопрос»: он появляется сразу, до первого вызова LLM.
Заголовки шагов — те же человекочитаемые статусы, что в консоли, с итогом после стрелки
(«Ищу: «…» → Найдено фрагментов: 5, документов: 2»).
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from typing import Any, Literal

from pydantic import BaseModel, Field

from agent.runner import (
    AgentEvent,
    AgentRunner,
    AgentSession,
    Answer,
    AnswerDelta,
    AnswerReady,
    CacheUsed,
    LoopNotes,
    LoopText,
    QueryRewritten,
    RunStarted,
    ToolFinished,
    ToolStarted,
)
from ui.citations import Citation, citations_for
from ui.state import message_metadata, turn_record

REWRITE_KEY = "rewrite"
CACHE_KEY = "cache"
NOTES_KEY = "notes"
REWRITE_TITLE = "Разбираю вопрос"
NO_SEARCH_TITLE = "Поиск не нужен: отвечаю по истории диалога"
UNCHANGED_TITLE = "Вопрос понят, ищу в документах"
REWRITTEN_TITLE = "Запрос с учётом диалога: «{query}»"
CACHE_TITLE = "Использую ранее найденное: {aliases}"
NOTES_TITLE = "Итоги поиска"
QUERIES_HEADER = "Отдельные поисковые запросы:"
ABBREVIATIONS_LINE = "Аббревиатуры: {items}"
REASON_LINE = "Пояснение: {reason}"
ARROW = " → "


class UiEvent(BaseModel):
    kind: str


class StepStarted(UiEvent):
    kind: Literal["step_started"] = "step_started"
    key: str
    title: str


class StepFinished(UiEvent):
    kind: Literal["step_finished"] = "step_finished"
    key: str
    title: str
    output: str | None = None
    ok: bool = True


class Token(UiEvent):
    kind: Literal["token"] = "token"
    text: str


class Final(UiEvent):
    kind: Literal["final"] = "final"
    text: str = Field(description="Ответ с номерами ссылок и блоком «Источники»")
    citations: list[Citation] = Field(description="Элементы для кликабельных ссылок (FR-4)")
    metadata: dict[str, Any] = Field(description="Metadata сообщения: запись хода и trace_id")
    answer: Answer
    first_signal_s: float
    seconds: float


def _rewrite_title(event: QueryRewritten) -> str:
    if not event.needs_search:
        return NO_SEARCH_TITLE
    if event.changed:
        return REWRITTEN_TITLE.format(query=event.query)
    return UNCHANGED_TITLE


def _rewrite_output(event: QueryRewritten) -> str | None:
    lines: list[str] = []
    if event.queries:
        lines.append(QUERIES_HEADER)
        lines += [f"{index}. {query}" for index, query in enumerate(event.queries, start=1)]
    if event.abbreviations:
        lines.append(ABBREVIATIONS_LINE.format(items=", ".join(event.abbreviations)))
    if event.reason:
        lines.append(REASON_LINE.format(reason=event.reason))
    return "\n".join(lines) or None


def _cache_output(event: CacheUsed, session: AgentSession) -> str:
    labels = []
    for alias in event.document_aliases:
        document = session.registry.document_by_alias(alias)
        labels.append(f"{alias} {document.label}".strip() if document is not None else alias)
    return "\n".join(labels)


async def run_question(runner: AgentRunner, session: AgentSession, question: str) -> AsyncIterator[UiEvent]:
    started = time.perf_counter()
    first_signal: float | None = None
    tool_titles: dict[str, str] = {}
    tools = 0
    notes = 0

    def signal() -> None:
        nonlocal first_signal
        if first_signal is None:
            first_signal = time.perf_counter() - started

    event: AgentEvent
    async for event in runner.run(question, session):
        if isinstance(event, RunStarted):
            signal()
            yield StepStarted(key=REWRITE_KEY, title=REWRITE_TITLE)
        elif isinstance(event, QueryRewritten):
            yield StepFinished(key=REWRITE_KEY, title=_rewrite_title(event), output=_rewrite_output(event))
        elif isinstance(event, CacheUsed):
            title = CACHE_TITLE.format(aliases=", ".join(event.document_aliases))
            yield StepStarted(key=CACHE_KEY, title=title)
            yield StepFinished(key=CACHE_KEY, title=title, output=_cache_output(event, session))
        elif isinstance(event, ToolStarted):
            tools += 1
            key = f"tool-{tools}"
            tool_titles[key] = event.status
            yield StepStarted(key=key, title=event.status)
        elif isinstance(event, ToolFinished):
            key = f"tool-{tools}"
            title = f"{tool_titles.get(key, event.tool)}{ARROW}{event.summary}"
            yield StepFinished(key=key, title=title, ok=event.ok)
        elif isinstance(event, LoopText):
            notes += 1
            key = f"note-{notes}"
            yield StepStarted(key=key, title=event.text)
            yield StepFinished(key=key, title=event.text)
        elif isinstance(event, LoopNotes):
            if event.text:
                yield StepStarted(key=NOTES_KEY, title=NOTES_TITLE)
                yield StepFinished(key=NOTES_KEY, title=NOTES_TITLE, output=event.text)
        elif isinstance(event, AnswerDelta):
            signal()
            yield Token(text=event.text)
        elif isinstance(event, AnswerReady):
            answer = event.answer
            record = turn_record(answer, session, first_signal_s=first_signal)
            yield Final(
                text=answer.text,
                citations=citations_for(answer.sources),
                metadata=message_metadata(record),
                answer=answer,
                first_signal_s=first_signal if first_signal is not None else time.perf_counter() - started,
                seconds=time.perf_counter() - started,
            )
