"""Память диалога (FR-6, 6.4): ходы «вопрос — ответ» сессии, буфер последних N и сводка старых.

Буфер хранит последние `memory.buffer_turns` ходов дословно; когда ходов больше или их объём превышает
`summary_trigger_chars`, старые ходы сжимаются LLM роли `summary` (без размышлений, N9) в сводку —
она накапливается (прошлая сводка + сжимаемые ходы) и подаётся в промпты вместе с буфером.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from llama_index.core.llms import LLM, ChatMessage
from pydantic import BaseModel, Field

from agent.prompts import SUMMARY_SYSTEM_PROMPT, summary_user_message
from agent.rendering import cut
from common.config import MemorySettings

logger = logging.getLogger(__name__)

USER_LABEL = "Пользователь"
ASSISTANT_LABEL = "Ассистент"
SUMMARY_LABEL = "Сводка предыдущего диалога"
EMPTY_HISTORY = "(это первый вопрос в диалоге)"
# в сводку старые ответы уходят целиком — обрезка не нужна
NO_CUT = 10**9


class Turn(BaseModel):
    question: str
    answer: str
    rewritten_query: str | None = Field(default=None, description="Поисковый запрос, если переписывался")
    document_aliases: list[str] = Field(
        default_factory=list, description="Документы, на которых строился ответ"
    )

    @property
    def chars(self) -> int:
        return len(self.question) + len(self.answer)


class ConversationMemory:
    """Ходы диалога по порядку и сводка сжатых старых ходов."""

    def __init__(self) -> None:
        self.turns: list[Turn] = []
        self.summary: str | None = None
        self.compactions = 0

    def add(self, turn: Turn) -> None:
        self.turns.append(turn)

    def recent(self, count: int) -> list[Turn]:
        return self.turns[-count:] if count > 0 else []

    def needs_compaction(self, settings: MemorySettings) -> bool:
        return bool(self.overflow(settings))

    def chars(self) -> int:
        return sum(turn.chars for turn in self.turns)

    def overflow(self, settings: MemorySettings) -> list[Turn]:
        """Ходы, которые уйдут в сводку: всё сверх буфера последних ходов (или сверх объёма)."""
        keep = settings.buffer_turns
        if len(self.turns) > keep:
            return self.turns[:-keep]
        if self.chars() > settings.summary_trigger_chars and len(self.turns) > 1:
            return self.turns[:-1]
        return []

    async def compact(self, llm: LLM, settings: MemorySettings) -> bool:
        """Сжимает старые ходы в сводку; возвращает True, если сводка обновилась."""
        old = self.overflow(settings)
        if not old:
            return False
        messages = [
            ChatMessage(
                role="system", content=SUMMARY_SYSTEM_PROMPT.format(max_words=settings.summary_max_words)
            ),
            ChatMessage(
                role="user",
                content=summary_user_message(self.summary, render_history(old, answer_chars=NO_CUT)),
            ),
        ]
        response = await llm.achat(messages)
        summary = (response.message.content or "").strip()
        if not summary:
            logger.warning("Суммаризация диалога вернула пустой текст: ходы остаются в буфере")
            return False
        self.summary = summary
        self.turns = self.turns[len(old) :]
        self.compactions += 1
        return True


def render_history(turns: Sequence[Turn], *, answer_chars: int, summary: str | None = None) -> str:
    """История для промптов: сводка (если есть) и последние ходы с обрезанными ответами."""
    lines: list[str] = []
    if summary:
        lines.append(f"{SUMMARY_LABEL}: {summary}")
    for turn in turns:
        lines.append(f"{USER_LABEL}: {turn.question}")
        lines.append(f"{ASSISTANT_LABEL}: {cut(turn.answer, answer_chars)}")
    return "\n".join(lines) if lines else EMPTY_HISTORY
