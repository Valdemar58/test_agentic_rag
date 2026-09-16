"""Память диалога (FR-6): ходы «вопрос — ответ» текущей сессии для переписывания запроса и ответа.

Здесь — буфер ходов и его представление для промптов; суммаризация старых ходов при переполнении
добавляется в 6.4.
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, Field

from agent.rendering import cut

USER_LABEL = "Пользователь"
ASSISTANT_LABEL = "Ассистент"
SUMMARY_LABEL = "Сводка предыдущего диалога"
EMPTY_HISTORY = "(это первый вопрос в диалоге)"


class Turn(BaseModel):
    question: str
    answer: str
    rewritten_query: str | None = Field(default=None, description="Поисковый запрос, если переписывался")


class ConversationMemory:
    """Ходы диалога по порядку; `recent(n)` — последние n для промптов."""

    def __init__(self) -> None:
        self.turns: list[Turn] = []
        self.summary: str | None = None

    def add(self, turn: Turn) -> None:
        self.turns.append(turn)

    def recent(self, count: int) -> list[Turn]:
        return self.turns[-count:] if count > 0 else []


def render_history(turns: Sequence[Turn], *, answer_chars: int, summary: str | None = None) -> str:
    """История для промптов: сводка (если есть) и последние ходы с обрезанными ответами."""
    lines: list[str] = []
    if summary:
        lines.append(f"{SUMMARY_LABEL}: {summary}")
    for turn in turns:
        lines.append(f"{USER_LABEL}: {turn.question}")
        lines.append(f"{ASSISTANT_LABEL}: {cut(turn.answer, answer_chars)}")
    return "\n".join(lines) if lines else EMPTY_HISTORY
