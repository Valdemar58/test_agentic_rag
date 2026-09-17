"""Проверка черновика ответа по свидетельствам отдельным вызовом LLM (роль `verify`, с размышлениями).

Живые диалоги 2026-09-17: шаг ответа посчитал «12:45 + 45 минут = 13:30», но подчинился выводу заметок
«не позднее 13:00», придумал правило и подпер его ссылками на фрагменты о контроле явки. Ни промпт, ни
самопроверка внутри того же вызова этого не ловят, поэтому проверка — отдельный шаг. Роли разделены:
LLM только находит предложения черновика, которые не подтверждены свидетельствами (числа, даты, названия,
правила, которых нет во фрагментах по ссылке или которые выведены без показанного расчёта), а вычёркивает
их детерминированный код — по одному предложению на замечание, без переписывания. Так проверяющий не может
дописать ничего своего (в живом прогоне он добавлял выдуманные примеры и «исправлял» верный расчёт).
Предложение с расчётом «ЧЧ:ММ + N мин = ЧЧ:ММ», у которого арифметика сходится, защищено от удаления, как и
предложения с его результатом; если после вычёркивания текст пуст или потерял все ссылки — остаётся черновик.
Любой сбой проверки тоже оставляет черновик: ответ пользователь получает всегда.
"""

from __future__ import annotations

import difflib
import json
import logging
import re
import time
from collections.abc import Collection
from dataclasses import dataclass
from typing import Literal, NamedTuple

from llama_index.core.llms import LLM, ChatMessage
from openai import OpenAIError
from pydantic import BaseModel, Field, ValidationError

from agent.llm import thinking_text
from agent.prompts import VERIFY_SYSTEM_PROMPT, verify_user_message
from common.config import VerifySettings

logger = logging.getLogger(__name__)

MARKER_RE = re.compile(r"\[(?:[SD]\d+|\d+)\]")
# число, время «13:30», дата «19.08.2024», сумма «26 702» — сравниваются как строки без пробелов
NUMBER_RE = re.compile(r"\d+(?:[:.,]\d+)*")
# время в вопросе или ответе: «12:45», «12.45», «12-45»
TIME_RE = re.compile(r"(?<!\d)(\d{1,2})[:.\-](\d{2})(?!\d)")
# явный расчёт времени, допускаются пояснения в скобках: «12:45 (уход) + 45 минут (перерыв) = 13:30»
TIME_SUM_RE = re.compile(
    r"(\d{1,2}):(\d{2})\s*(?:\([^)]*\))?\s*\+\s*(\d{1,3})\s*мин\w*\s*(?:\([^)]*\))?\s*=\s*(\d{1,2}):(\d{2})"
)
# границы предложений с сохранением разделителей: переводы строк и конец предложения перед заглавной буквой
_SPLIT_RE = re.compile(r"(\n+|(?<=[.!?;])[ \t]+(?=[А-ЯЁA-Z«\"(*\-–—]))")
_NORMALIZE_RE = re.compile(r"[^\w:]+")
# строка из одного маркера списка или заголовка-обёртки, оставшаяся после вычёркивания
_EMPTY_LINE_RE = re.compile(r"^[ \t]*(?:[-–—•*]+|\d+[.)])[ \t]*$")
# короткий заголовок с двоеточием («**Детали:**», «Расчёт:»), после которого ничего не осталось
_HEADING_LINE_RE = re.compile(r"^[\s*_#]*[^:\n]{1,40}:[\s*_]*$")
# JSON без экранирования переводов строк внутри строк (strict=False) и только первый объект (raw_decode)
_DECODER = json.JSONDecoder(strict=False)
# замечание короче стольких слов ищется не подстрокой, а по похожести: «15:00» есть и в верных предложениях
SUBSTRING_MIN_WORDS = 3

Action = Literal["removed", "kept", "unmatched"]


class VerifyProblem(BaseModel):
    claim: str = Field(description="Предложение черновика, не подтверждённое свидетельствами")
    reason: str = ""
    action: Action = Field(
        default="unmatched",
        description="removed — предложение вычеркнуто; kept — защищено расчётом или вычёркивание отклонено; "
        "unmatched — в черновике не найдено",
    )


class Verification(BaseModel):
    problems: list[VerifyProblem] = Field(default_factory=list)
    corrected: bool = Field(default=False, description="Из черновика вычеркнуты неподтверждённые предложения")
    emptied: bool = Field(
        default=False, description="Замечания отклонили весь черновик: вычёркивать нечего, оставлен как есть"
    )
    parsed: bool = Field(default=True, description="Ответ проверяющего получен и разобран")
    seconds: float = 0.0
    thinking: str | None = None


@dataclass(frozen=True)
class ParsedVerification:
    problems: list[VerifyProblem]
    parsed: bool


def numbers_in(text: str) -> set[str]:
    """Числа текста без псевдонимов и номеров ссылок ([S1], [2] — не числа)."""
    return {item.replace(" ", "") for item in NUMBER_RE.findall(MARKER_RE.sub("", text))}


def times_in(text: str) -> set[str]:
    """Времена «12:45», «12.45», «12-45» текста в виде «12:45»."""
    return {f"{int(hours)}:{minutes}" for hours, minutes in TIME_RE.findall(text)}


def computed_times(text: str, *, starts: Collection[str] | None = None) -> set[str]:
    """Результаты явных расчётов «ЧЧ:ММ + N мин = ЧЧ:ММ» в тексте, у которых арифметика сходится.

    Со `starts` учитываются только расчёты от одного из этих времён: защищён расчёт от времени из вопроса
    («ушёл в 12:45»), а не от начала окна «12:00 + 45 мин» (живой прогон 2026-09-17)."""
    results: set[str] = set()
    for hours, minutes, delta, result_hours, result_minutes in TIME_SUM_RE.findall(text):
        if starts is not None and f"{int(hours)}:{minutes}" not in starts:
            continue
        total = int(hours) * 60 + int(minutes) + int(delta)
        if divmod(total, 60) == (int(result_hours), int(result_minutes)):
            results.add(f"{int(result_hours)}:{result_minutes}")
    return results


def normalize(text: str) -> str:
    return _NORMALIZE_RE.sub(" ", MARKER_RE.sub("", text).casefold().replace("ё", "е")).strip()


def parse_verification(text: str) -> ParsedVerification:
    """JSON проверяющего → замечания; что не разбирается — «проверка не удалась»."""
    start = (text or "").find("{")
    if start < 0:
        return ParsedVerification([], parsed=False)
    try:
        data, _ = _DECODER.raw_decode(text, start)
        if not isinstance(data, dict):
            raise TypeError("ожидался объект JSON")
        raw_problems = data.get("problems") or []
        if not isinstance(raw_problems, list):
            raise TypeError("problems должен быть списком")
        problems = [
            VerifyProblem(claim=str(item.get("claim") or "").strip(), reason=str(item.get("reason") or ""))
            for item in raw_problems
            if isinstance(item, dict) and str(item.get("claim") or "").strip()
        ]
    except (json.JSONDecodeError, ValidationError, TypeError, ValueError) as exc:
        logger.warning("Проверка ответа: ответ модели не разобран (%s): %.200s", exc, text)
        return ParsedVerification([], parsed=False)
    return ParsedVerification(problems, parsed=True)


def _pieces(text: str) -> list[tuple[str, str]]:
    """Черновик → пары (предложение, разделитель после него) с сохранением исходного форматирования."""
    parts = _SPLIT_RE.split(text)
    pieces: list[tuple[str, str]] = []
    for index in range(0, len(parts), 2):
        separator = parts[index + 1] if index + 1 < len(parts) else ""
        if parts[index] or separator:
            pieces.append((parts[index], separator))
    return pieces


def _long(text: str) -> bool:
    return len(text.split()) >= SUBSTRING_MIN_WORDS


def _matches(claim: str, sentences: list[str], ratio: float) -> set[int]:
    """Предложения черновика, о которых замечание.

    Каждое предложение замечания — предложение черновика, содержащее его (замечание может цитировать часть
    предложения); замечание на абзац — все предложения черновика, которые оно охватывает, вместе с короткими
    соседями («Детали:», «12:45 [S6].»), тоже входящими в него (живой прогон 2026-09-17); короткое замечание
    без совпадений — самое похожее предложение не ниже `ratio`."""
    key = normalize(claim)
    if not key:
        return set()
    parts = [part for part in (normalize(piece) for piece, _ in _pieces(claim)) if _long(part)]
    found: set[int] = set()
    exact = {part for part in (normalize(piece) for piece, _ in _pieces(claim)) if part}
    for index, sentence in enumerate(sentences):
        if _long(sentence) and sentence in key:
            found.add(index)
        elif sentence in exact or any(part in sentence for part in parts):
            found.add(index)
    grown = True
    while grown:
        grown = False
        for index in sorted(found):
            for neighbour in (index - 1, index + 1):
                if 0 <= neighbour < len(sentences) and neighbour not in found:
                    sentence = sentences[neighbour]
                    if sentence and not _long(sentence) and sentence in key:
                        found.add(neighbour)
                        grown = True
    if found:
        return found
    scored = [
        (difflib.SequenceMatcher(None, key, sentence).ratio(), index)
        for index, sentence in enumerate(sentences)
        if sentence
    ]
    if not scored:
        return set()
    best, index = max(scored)
    return {index} if best >= ratio else set()


class Applied(NamedTuple):
    text: str
    corrected: bool
    emptied: bool


def apply_problems(
    draft: str, problems: list[VerifyProblem], *, match_ratio: float, question: str = ""
) -> Applied:
    """Вычёркивает из черновика предложения, о которых замечания.

    Защищены предложения с верным явным расчётом от времени из вопроса и с его результатом. Если после
    вычёркивания текст пуст или без единой ссылки при ссылках в черновике — черновик остаётся, замечания
    помечаются как kept, а `emptied` говорит раннеру, что отклонён весь черновик."""
    pieces = _pieces(draft)
    sentences = [normalize(piece) for piece, _ in pieces]
    starts = times_in(question)
    protected_numbers = computed_times(draft, starts=starts)
    removed: set[int] = set()
    for problem in problems:
        indexes = _matches(problem.claim, sentences, match_ratio)
        if not indexes:
            problem.action = "unmatched"
            continue
        protected = {
            index
            for index in indexes
            if computed_times(pieces[index][0], starts=starts)
            or (numbers_in(pieces[index][0]) & protected_numbers)
        }
        if protected == indexes:
            problem.action = "kept"
            continue
        removed |= indexes - protected
        problem.action = "removed"
    if not removed:
        return Applied(draft, False, False)
    kept: list[list[str]] = []
    for index, (piece, separator) in enumerate(pieces):
        if index not in removed:
            kept.append([piece, separator])
        elif kept and "\n" in separator and len(separator) > len(kept[-1][1]):
            kept[-1][1] = separator  # абзацный отступ вычеркнутого предложения переходит к предыдущему
    text = "".join(piece + separator for piece, separator in kept)
    lines = [line for line in text.splitlines() if not _EMPTY_LINE_RE.match(line)]
    while lines and _HEADING_LINE_RE.match(lines[-1]):
        lines.pop()  # заголовок «Детали:», под которым ничего не осталось
    text = re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()
    if not text or (MARKER_RE.search(draft) and not MARKER_RE.search(text)):
        logger.info("Проверка ответа: после вычёркивания текст пуст или без ссылок, оставлен черновик")
        for problem in problems:
            if problem.action == "removed":
                problem.action = "kept"
        return Applied(draft, False, True)
    return Applied(text, True, False)


class AnswerVerifier:
    def __init__(self, llm: LLM, settings: VerifySettings) -> None:
        self._llm = llm
        self._settings = settings

    async def verify(self, question: str, draft: str, evidence: str) -> tuple[Verification, str]:
        """Черновик → (итог проверки, текст ответа): без неподтверждённых предложений или черновик."""
        started = time.perf_counter()
        messages = [
            ChatMessage(role="system", content=VERIFY_SYSTEM_PROMPT),
            ChatMessage(role="user", content=verify_user_message(question, draft, evidence)),
        ]
        try:
            response = await self._llm.achat(messages)
        except OpenAIError as exc:
            logger.error("Проверка ответа не выполнена, черновик оставлен: %s", exc)
            return Verification(parsed=False, seconds=time.perf_counter() - started), draft
        parsed = parse_verification(response.message.content or "")
        applied = apply_problems(
            draft, parsed.problems, match_ratio=self._settings.claim_match_ratio, question=question
        )
        verification = Verification(
            problems=parsed.problems,
            corrected=applied.corrected,
            emptied=applied.emptied,
            parsed=parsed.parsed,
            seconds=time.perf_counter() - started,
            thinking=thinking_text(response.message),
        )
        return verification, applied.text
