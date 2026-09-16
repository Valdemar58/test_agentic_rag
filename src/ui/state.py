"""Состояние диалога между сессиями UI (FR-6, FR-7): что сохраняется с ответом и как восстанавливается.

С каждым ответом ассистента в metadata сообщения пишется `TurnRecord`: вопрос, переписанный запрос,
документы ответа, снимок свидетельств (документы и фрагменты этого хода) и `trace_id` Langfuse. При
открытии старого диалога из этих записей собираются память ходов и реестр свидетельств, поэтому
уточнение по ранее найденному документу отвечается из кэша и после перезапуска.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from agent.evidence import EvidenceSnapshot
from agent.memory import Turn
from agent.runner import AgentSession, Answer
from common.config import AppConfig

logger = logging.getLogger(__name__)

AGENT_METADATA_KEY = "agent"
TRACE_ID_KEY = "trace_id"
ASSISTANT_MESSAGE = "assistant_message"


class TurnRecord(BaseModel):
    """Ход диалога в metadata сообщения ассистента."""

    question: str
    rewritten_query: str | None = None
    document_aliases: list[str] = Field(default_factory=list)
    evidence: EvidenceSnapshot = Field(default_factory=EvidenceSnapshot)
    trace_id: str | None = None
    tool_calls: int = 0
    refused: bool = False
    budget_exhausted: bool = False
    context_exhausted: bool = False
    seconds: float = 0.0
    loop_seconds: float = 0.0
    answer_seconds: float = 0.0
    first_signal_s: float | None = Field(default=None, description="Первый видимый сигнал, с (AC-7.1)")


def turn_record(answer: Answer, session: AgentSession, *, first_signal_s: float | None) -> TurnRecord:
    return TurnRecord(
        question=answer.question,
        rewritten_query=answer.rewritten_query,
        document_aliases=list(answer.document_aliases),
        evidence=session.registry.snapshot(answer.document_aliases, answer.fragment_aliases),
        trace_id=answer.trace_id,
        tool_calls=len(answer.tool_calls),
        refused=answer.refused,
        budget_exhausted=answer.budget_exhausted,
        context_exhausted=answer.context_exhausted,
        seconds=round(answer.seconds, 3),
        loop_seconds=round(answer.loop_seconds, 3),
        answer_seconds=round(answer.answer_seconds, 3),
        first_signal_s=round(first_signal_s, 3) if first_signal_s is not None else None,
    )


def message_metadata(record: TurnRecord) -> dict[str, Any]:
    """Metadata сообщения ассистента: запись хода и `trace_id` для связи с фидбэком (FR-7, FR-8)."""
    return {AGENT_METADATA_KEY: record.model_dump(mode="json"), TRACE_ID_KEY: record.trace_id}


def restore_session(config: AppConfig, session_id: str, steps: Iterable[Mapping[str, Any]]) -> AgentSession:
    """Сессия агента по сохранённым шагам диалога (в порядке создания)."""
    session = AgentSession(config, session_id=session_id)
    for step in steps:
        if step.get("type") != ASSISTANT_MESSAGE:
            continue
        raw = (step.get("metadata") or {}).get(AGENT_METADATA_KEY)
        if not raw:
            continue
        try:
            record = TurnRecord.model_validate(raw)
        except ValidationError as exc:
            logger.warning("Запись хода диалога не разбирается, ход пропущен: %s", exc)
            continue
        session.registry.restore(record.evidence)
        session.memory.add(
            Turn(
                question=record.question,
                answer=str(step.get("output") or ""),
                rewritten_query=record.rewritten_query,
                document_aliases=record.document_aliases,
            )
        )
    return session
