"""LLM-судья голден-сета (§10.2): корректность ответа, покрытие ссылками и поддержка ссылок фрагментами.

Судит та же Qwen3-8B, но отдельным промптом и без размышлений: температура 0 (`eval.judge`), ответ —
один объект JSON со счётчиками, а не рассуждение. Судья видит вопрос, эталон из голден-сета, ответ агента
с нумерованными ссылками и тексты процитированных фрагментов; он ничего не переписывает — только считает
утверждения и сверяет их с фрагментами. Разбор устойчив к тексту вокруг объекта (как в `agent/verify.py`);
неразобранный ответ помечается `parsed = false` и в метрики не идёт — вопрос попадает в отчёт как
«судья не ответил», а не как ноль баллов.

Ручная валидация судьи на первых 20 вопросах (§10.2, задача 9.4) сравнивает `correctness` с оценкой
человека: согласие ниже 80 % — повод править промпт, а не метрики.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Sequence

from llama_index.core.llms import LLM, ChatMessage
from llama_index.llms.openai_like import OpenAILike
from openai import OpenAIError
from pydantic import BaseModel, Field

from agent.citations import Source
from agent.llm import request_kwargs
from common.config import AppConfig, JudgeSettings, LlmRequestOptions
from common.settings import Settings

logger = logging.getLogger(__name__)

_DECODER = json.JSONDecoder(strict=False)
MAX_SCORE = 5
MIN_SCORE = 1

JUDGE_SYSTEM_PROMPT = """Ты строгий оценщик ответов помощника по документам организации. Тебе дают вопрос, \
эталонный ответ (что считается верным), ответ помощника с нумерованными ссылками и тексты фрагментов \
документов, на которые эти ссылки указывают.

Считай по ответу помощника:
1. correctness — насколько ответ соответствует эталону по существу, целое число от 1 до 5:
   5 — всё существенное из эталона есть и нет неверного; 4 — верно, но неполно; 3 — верно наполовину \
или расплывчато; 2 — в основном неверно; 1 — неверно или не по вопросу.
   Если эталон говорит, что правильный ответ — отказ («в документах ответа нет»), то честный отказ — \
это 5, даже если помощник дополнительно перечислил, что в документах есть рядом; выдуманный ответ \
вместо отказа — 1.
2. statements — сколько в ответе фактических утверждений о содержании документов (заголовки, вводные \
фразы и вопросы не считаются).
3. with_citation — сколько из них снабжены ссылкой [n].
4. citations — сколько всего ссылок [n] в тексте ответа.
5. supported — сколько ссылок действительно подтверждают утверждение, рядом с которым стоят: текст \
фрагмента содержит то, что утверждается.
6. refusal — true, если ответ прямо говорит, что ответа в документах нет, и ничего не выдумывает.
7. conflict — true, если ответ опирается на действующий документ или прямо называет, что другой документ \
отменён либо противоречит.

Ответь строго одним объектом JSON и ничем больше:
{"correctness": 1-5, "statements": N, "with_citation": N, "citations": N, "supported": N, \
"refusal": true|false, "conflict": true|false, "comment": "одна фраза, что не так"}

Не переписывай ответ, не добавляй своих фактов и не оценивай стиль."""

JUDGE_USER_TEMPLATE = """Вопрос: {question}

Эталонный ответ: {expected}

Ответ помощника:
{answer}

Фрагменты по ссылкам:
{sources}"""

NO_SOURCES = "(ссылок в ответе нет)"


class JudgeVerdict(BaseModel):
    correctness: int = Field(default=0, ge=0, le=MAX_SCORE, description="Соответствие эталону, 1–5")
    statements: int = Field(default=0, ge=0, description="Фактических утверждений в ответе")
    with_citation: int = Field(default=0, ge=0, description="Утверждений со ссылкой (M2)")
    citations: int = Field(default=0, ge=0, description="Ссылок в тексте")
    supported: int = Field(default=0, ge=0, description="Ссылок, подтверждающих утверждение (M3)")
    refusal: bool = Field(default=False, description="Ответ — честный отказ (M4)")
    conflict: bool = Field(default=False, description="Опёрся на действующий документ или назвал отмену (M5)")
    comment: str = ""
    parsed: bool = Field(default=True, description="Ответ судьи разобран")
    seconds: float = 0.0


def parse_verdict(text: str) -> JudgeVerdict:
    """JSON судьи → вердикт; что не разбирается — `parsed = false` (в метрики не идёт)."""
    start = (text or "").find("{")
    if start < 0:
        return JudgeVerdict(parsed=False)
    try:
        data, _ = _DECODER.raw_decode(text, start)
        if not isinstance(data, dict):
            raise TypeError("ожидался объект JSON")
        counts = {
            key: max(0, int(data.get(key) or 0))
            for key in ("statements", "with_citation", "citations", "supported")
        }
        score = int(data.get("correctness") or 0)
        return JudgeVerdict(
            correctness=min(MAX_SCORE, max(0, score)),
            refusal=bool(data.get("refusal")),
            conflict=bool(data.get("conflict")),
            comment=str(data.get("comment") or "")[:400],
            **counts,
        )
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        logger.warning("Судья: ответ не разобран (%s): %.200s", exc, text)
        return JudgeVerdict(parsed=False)


def render_sources(sources: Sequence[Source], *, chars: int) -> str:
    """Тексты процитированных фрагментов для судьи: номер, документ, координаты, текст."""
    if not sources:
        return NO_SOURCES
    blocks: list[str] = []
    for source in sources:
        where = source.breadcrumbs or source.label
        text = " ".join((source.text or "").split())
        if len(text) > chars:
            text = f"{text[:chars].rstrip()}…"
        blocks.append(f"[{source.number}] {where}\n{text or '(текст недоступен)'}")
    return "\n\n".join(blocks)


def judge_user_message(
    question: str, expected: str, answer: str, sources: Sequence[Source], *, chars: int
) -> str:
    return JUDGE_USER_TEMPLATE.format(
        question=question, expected=expected, answer=answer, sources=render_sources(sources, chars=chars)
    )


def build_judge_llm(config: AppConfig, settings: Settings) -> OpenAILike:
    """LLM судьи: та же модель, что у агента, но температура и лимит — из `eval.judge`, без размышлений."""
    judge = config.eval.judge
    options = LlmRequestOptions(
        temperature=judge.temperature,
        top_p=config.agent.llm.sampling.top_p,
        top_k=config.agent.llm.sampling.top_k,
        max_tokens=judge.max_tokens,
        enable_thinking=False,
    )
    return OpenAILike(
        model=config.vllm.qwen.served_model_name,
        api_base=settings.resolve_llm_base_url(config),
        api_key=settings.llm_api_key.get_secret_value(),
        is_chat_model=True,
        context_window=config.vllm.qwen.max_model_len,
        temperature=options.temperature,
        max_tokens=options.max_tokens,
        timeout=config.agent.llm.timeout_s,
        max_retries=config.agent.llm.max_retries,
        additional_kwargs=request_kwargs(options),
    )


class AnswerJudge:
    """Оценка одного ответа: один вызов LLM, один JSON-вердикт."""

    def __init__(self, llm: LLM, settings: JudgeSettings) -> None:
        self._llm = llm
        self._settings = settings

    async def judge(
        self, question: str, expected: str, answer: str, sources: Sequence[Source]
    ) -> JudgeVerdict:
        message = judge_user_message(question, expected, answer, sources, chars=self._settings.source_chars)
        started = time.perf_counter()
        try:
            response = await self._llm.achat(
                [
                    ChatMessage(role="system", content=JUDGE_SYSTEM_PROMPT),
                    ChatMessage(role="user", content=message),
                ]
            )
        except (OpenAIError, ValueError) as exc:
            logger.error("Судья не ответил на вопрос «%.60s»: %s", question, exc)
            return JudgeVerdict(parsed=False, seconds=time.perf_counter() - started)
        verdict = parse_verdict(response.message.content or "")
        return verdict.model_copy(update={"seconds": time.perf_counter() - started})
