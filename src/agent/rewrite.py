"""Переписывание запроса (6.3, FR-6): уточняющий вопрос + история → самодостаточный поисковый запрос.

LLM роли `rewrite` (с размышлениями, N9) получает историю диалога, известные документы сессии и новый
вопрос и отвечает JSON: `query` — самодостаточный запрос для поиска; `intent` — намерение из закрытого
списка (вопрос к документам, приветствие, вопрос о возможностях, просьба переформулировать прошлый
ответ); `relevant_documents` — псевдонимы уже найденных документов, к которым относится вопрос (кэш
сессии, 6.4); `abbreviations` — аббревиатуры для расшифровки (глоссарий, этап 8). Ответ разбирается
устойчиво: JSON ищется в тексте, при провале поиск идёт по исходному вопросу.

Без поиска обрабатываются только три названных намерения, причём переформулировка — лишь при непустой
истории: свободного флага «поиск не нужен» у модели нет (живой диалог 2026-09-16: бытовой вопрос «во
сколько вернуться с обеда» ушёл в режим беседы и получил выдуманный отказ без единого поиска).
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Sequence
from typing import Literal, cast, get_args

from llama_index.core.llms import LLM, ChatMessage
from pydantic import BaseModel, Field, ValidationError

from agent.evidence import ALIAS_RE, DOC_PREFIX, KnownDocument
from agent.llm import thinking_text
from agent.memory import Turn, render_history
from agent.prompts import REWRITE_SYSTEM_PROMPT, rewrite_user_message
from common.config import RewriteSettings

logger = logging.getLogger(__name__)

JSON_RE = re.compile(r"\{.*\}", re.DOTALL)
FALLBACK_REASON = "ответ модели не разобран, поиск по исходному вопросу"
# больше отдельных запросов не имеет смысла при бюджете в 8 вызовов (FR-1)
MAX_QUERIES = 4

Intent = Literal["documents", "greeting", "capabilities", "rephrase"]
INTENT_DOCUMENTS: Intent = "documents"
INTENT_REPHRASE: Intent = "rephrase"
INTENTS: tuple[str, ...] = get_args(Intent)


class RewrittenQuery(BaseModel):
    question: str
    query: str = Field(description="Самодостаточный поисковый запрос")
    queries: list[str] = Field(
        default_factory=list, description="Отдельные поисковые запросы для многочастного вопроса (AC-1.1)"
    )
    intent: Intent = Field(
        default=INTENT_DOCUMENTS,
        description="Намерение: только greeting, capabilities и rephrase обходятся без поиска",
    )
    relevant_documents: list[str] = Field(default_factory=list, description="Псевдонимы документов сессии")
    abbreviations: list[str] = Field(default_factory=list, description="Аббревиатуры для глоссария")
    reason: str | None = None
    thinking: str | None = None

    @property
    def needs_search(self) -> bool:
        return self.intent == INTENT_DOCUMENTS

    @property
    def changed(self) -> bool:
        return " ".join(self.query.split()).casefold() != " ".join(self.question.split()).casefold()


def resolve_intent(data: dict[str, object], *, has_history: bool) -> Intent:
    """Намерение из ответа модели; неизвестное или устаревший флаг needs_search — вопрос к документам.

    Переформулировать нечего, если история пуста: такое намерение тоже становится вопросом к документам."""
    raw = str(data.get("intent") or "").strip().casefold()
    if raw not in INTENTS:
        raw = INTENT_DOCUMENTS if data.get("needs_search", True) else INTENT_REPHRASE
    intent = cast(Intent, raw)
    if intent == INTENT_REPHRASE and not has_history:
        return INTENT_DOCUMENTS
    return intent


def _strings(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [" ".join(str(item).split()) for item in value if str(item).strip()]


def parse_rewrite(
    question: str, text: str, thinking: str | None = None, *, has_history: bool = True
) -> RewrittenQuery:
    """JSON из ответа модели → `RewrittenQuery`; что угодно другое → поиск по исходному вопросу."""
    match = JSON_RE.search(text or "")
    if match is None:
        return RewrittenQuery(question=question, query=question, reason=FALLBACK_REASON, thinking=thinking)
    try:
        data = json.loads(match.group(0))
        if not isinstance(data, dict):
            raise TypeError("ожидался объект JSON")
        query = " ".join(str(data.get("query") or "").split()) or question
        documents = [
            item
            for item in _strings(data.get("relevant_documents"))
            if ALIAS_RE.match(item) and item.startswith(DOC_PREFIX)
        ]
        queries = _strings(data.get("queries"))[:MAX_QUERIES]
        return RewrittenQuery(
            question=question,
            query=query,
            queries=queries if len(queries) >= 2 else [],
            intent=resolve_intent(data, has_history=has_history),
            relevant_documents=documents,
            abbreviations=_strings(data.get("abbreviations")),
            reason=str(data["reason"]) if data.get("reason") else None,
            thinking=thinking,
        )
    except (json.JSONDecodeError, ValidationError, TypeError, ValueError) as exc:
        logger.warning("Переписывание запроса: ответ модели не разобран (%s): %.200s", exc, text)
        return RewrittenQuery(question=question, query=question, reason=FALLBACK_REASON, thinking=thinking)


class QueryRewriter:
    def __init__(self, llm: LLM, settings: RewriteSettings) -> None:
        self._llm = llm
        self._settings = settings

    async def rewrite(
        self, question: str, turns: Sequence[Turn], documents: Sequence[KnownDocument], summary: str | None
    ) -> RewrittenQuery:
        if not self._settings.enabled:
            return RewrittenQuery(question=question, query=question)
        history = render_history(
            turns[-self._settings.history_turns :] if self._settings.history_turns else [],
            answer_chars=self._settings.answer_chars,
            summary=summary,
        )
        last_documents = turns[-1].document_aliases if turns else []
        messages = [
            ChatMessage(role="system", content=REWRITE_SYSTEM_PROMPT),
            ChatMessage(
                role="user", content=rewrite_user_message(question, history, documents, last_documents)
            ),
        ]
        response = await self._llm.achat(messages)
        return parse_rewrite(
            question,
            response.message.content or "",
            thinking_text(response.message),
            has_history=bool(turns) or bool(summary),
        )
