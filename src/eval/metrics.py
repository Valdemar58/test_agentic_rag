"""Метрики голден-сета M1–M8 (§10.3) и отчёт: markdown для людей, json для машин и diff.

Считаются по итогам прогона (`QuestionOutcome`): часть детерминированно (попадание ожидаемых документов
в источники, резолвятся ли ссылки, явный отказ, опора на действующий документ), часть — по вердикту
судьи (корректность, покрытие ссылками, поддержка ссылок фрагментами). Вопрос, по которому судья не
ответил, в судейские метрики не входит и виден в отчёте отдельной строкой: занижать метрику из-за сбоя
оценщика нельзя.

M8 (инжест) приходит из отчёта инжеста, а не из прогона: считать его заново eval-harness не может.
"""

from __future__ import annotations

import datetime as dt
import re
from collections.abc import Iterable, Sequence
from typing import Any, Literal

from pydantic import BaseModel, Field

from agent.citations import Source
from eval.golden_set import Category
from eval.judge import JudgeVerdict

ACTIVE_STATUS = "действует"
# «отменён», «утратил силу», «недействующий», «противоречит» — прямое указание на конфликт документов
CANCELLED_RE = re.compile(r"отмен|утрат\w+ силу|недейств|противореч|заменён|заменен", re.IGNORECASE)
RETRIEVAL_CATEGORIES: tuple[Category, ...] = ("simple", "multi_doc", "duplicated")
P95 = 0.95

Status = Literal["достигнута", "не достигнута", "не измерена"]


class QuestionOutcome(BaseModel):
    """Итог одного вопроса голден-сета: ответ агента, детерминированные признаки и вердикт судьи."""

    id: str
    category: Category
    derived: bool = False
    question: str
    answer: str
    expected_doc_ids: list[str] = Field(default_factory=list)
    expected_any: bool = Field(
        default=False, description="Достаточно любого ожидаемого документа (категория duplicated)"
    )
    sources: list[Source] = Field(default_factory=list, description="Источники ответа (нужны судье)")
    refused: bool = False
    unresolved: list[str] = Field(default_factory=list, description="Ссылки, не найденные в реестре (M3)")
    tool_calls: int = 0
    first_signal_s: float = Field(default=0.0, description="До первого события потока (шаг в UI), M6")
    first_token_s: float = Field(default=0.0, description="До первого токена ответа — сверх ТЗ, для NFR-2")
    seconds: float = 0.0
    follow_up: str | None = None
    follow_up_answer: str | None = None
    follow_up_seconds: float = 0.0
    trace_id: str | None = None
    judge: JudgeVerdict | None = None
    follow_up_judge: JudgeVerdict | None = Field(default=None, description="Вердикт по второму ходу")
    error: str | None = Field(default=None, description="Прогон вопроса упал: метрики по нему не считаются")

    @property
    def found_doc_ids(self) -> list[str]:
        return sorted({source.doc_id for source in self.sources})

    @property
    def citations(self) -> int:
        return len(self.sources)

    @property
    def hit(self) -> bool:
        """Ожидаемые документы есть в источниках ответа (M1).

        По умолчанию нужны все; у вопросов с `expected_any` (один и тот же ответ записан в нескольких
        документах, категория `duplicated`) достаточно любого — иначе метрика наказывала бы за ссылку
        на равноценный документ."""
        found = self.found_doc_ids
        if self.expected_any:
            return any(doc_id in found for doc_id in self.expected_doc_ids)
        return all(doc_id in found for doc_id in self.expected_doc_ids)

    @property
    def mentions_conflict(self) -> bool:
        return bool(CANCELLED_RE.search(self.answer))

    @property
    def only_active_sources(self) -> bool:
        return bool(self.sources) and all(source.doc_status == ACTIVE_STATUS for source in self.sources)

    @property
    def refusal_ok(self) -> bool:
        """Честный отказ: явная формулировка агента или вердикт судьи (M4)."""
        return self.refused or bool(self.judge and self.judge.parsed and self.judge.refusal)

    @property
    def conflict_ok(self) -> bool:
        """Опёрся на нужный документ и учёл статус: только действующие источники, названная отмена
        или вердикт судьи (M5).

        Опоры на «какой-нибудь действующий документ» мало: в живом прогоне 2026-09-17 ответ про
        отменённую инструкцию сослался на посторонний действующий приказ и формально проходил проверку.
        Поэтому засчитывается только ответ, в источниках которого есть документ из голден-сета."""
        if not self.hit:
            return False
        if self.only_active_sources or self.mentions_conflict:
            return True
        return bool(self.judge and self.judge.parsed and self.judge.conflict)

    @property
    def judged(self) -> JudgeVerdict | None:
        return self.judge if self.judge and self.judge.parsed else None


class Metric(BaseModel):
    id: str
    name: str
    value: float | None = Field(description="Значение; None — метрика не измерена")
    target: str = Field(description="Цель MVP по §10.3")
    reached: Status
    detail: str = ""

    def rendered(self) -> str:
        if self.value is None:
            return "—"
        return f"{self.value:.0%}" if self.id not in {"M6", "M7"} else f"{self.value:.2f}"


def _share(passed: int, total: int) -> float | None:
    return passed / total if total else None


def percentile(values: Sequence[float], share: float) -> float | None:
    """Персентиль по ближайшему рангу: p95 первого сигнала (M6)."""
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(share * len(ordered) + 0.5) - 1))
    return ordered[index]


def _status(value: float | None, target: float, *, greater_is_better: bool = True) -> Status:
    if value is None:
        return "не измерена"
    reached = value >= target if greater_is_better else value <= target
    return "достигнута" if reached else "не достигнута"


class EvalMetrics(BaseModel):
    metrics: list[Metric]
    counts: dict[str, int] = Field(default_factory=dict, description="Вопросов по категориям")
    judged: int = 0
    unjudged: int = 0
    failed: int = 0

    def by_id(self, metric_id: str) -> Metric | None:
        return next((metric for metric in self.metrics if metric.id == metric_id), None)


def compute_metrics(
    outcomes: Sequence[QuestionOutcome], *, first_signal_budget_s: float, ingest_share: float | None = None
) -> EvalMetrics:
    """Метрики M1–M8 по итогам прогона; M8 — из отчёта инжеста, если он передан."""
    done = [outcome for outcome in outcomes if outcome.error is None]
    by_category: dict[str, list[QuestionOutcome]] = {}
    for outcome in done:
        by_category.setdefault(outcome.category, []).append(outcome)

    def hits(category: Category) -> tuple[int, int]:
        items = by_category.get(category, [])
        return sum(item.hit for item in items), len(items)

    simple_hits, simple_total = hits("simple")
    multi_hits, multi_total = hits("multi_doc")
    duplicated_hits, duplicated_total = hits("duplicated")
    retrieval = [item for item in done if item.category in RETRIEVAL_CATEGORIES]

    judged = [item for item in done if item.judged is not None]
    statements = sum(item.judge.statements for item in judged if item.judge)
    with_citation = sum(item.judge.with_citation for item in judged if item.judge)
    citations = sum(item.judge.citations for item in judged if item.judge)
    supported = sum(item.judge.supported for item in judged if item.judge)
    resolved = sum(not item.unresolved for item in done)

    no_answer = by_category.get("no_answer", [])
    contradiction = by_category.get("contradiction", [])
    simple_scores = [
        item.judge.correctness
        for item in by_category.get("simple", [])
        if item.judged is not None and item.judge
    ]
    first_signals = [item.first_signal_s for item in done]

    simple_share = _share(simple_hits, simple_total)
    multi_share = _share(multi_hits, multi_total)
    citation_support = _share(supported, citations)
    metrics = [
        Metric(
            id="M1-simple",
            name="Retrieval hit rate, simple",
            value=simple_share,
            target="≥ 85 %",
            reached=_status(simple_share, 0.85),
            detail=f"{simple_hits} из {simple_total}",
        ),
        Metric(
            id="M1-multi_doc",
            name="Retrieval hit rate, multi_doc",
            value=multi_share,
            target="≥ 70 %",
            reached=_status(multi_share, 0.70),
            detail=f"{multi_hits} из {multi_total}",
        ),
        Metric(
            id="M1-duplicated",
            name="Retrieval hit rate, duplicated",
            value=_share(duplicated_hits, duplicated_total),
            target="—",
            reached="не измерена" if not duplicated_total else "достигнута",
            detail=f"{duplicated_hits} из {duplicated_total}; всего по трём категориям "
            f"{sum(item.hit for item in retrieval)} из {len(retrieval)}",
        ),
        Metric(
            id="M2",
            name="Citation coverage",
            value=_share(with_citation, statements),
            target="≥ 90 %",
            reached=_status(_share(with_citation, statements), 0.90),
            detail=f"утверждений со ссылкой {with_citation} из {statements} (судья)",
        ),
        Metric(
            id="M3-resolve",
            name="Citation validity: ссылки резолвятся",
            value=_share(resolved, len(done)),
            target="100 %",
            reached=_status(_share(resolved, len(done)), 1.0),
            detail=f"ответов без потерянных ссылок {resolved} из {len(done)}",
        ),
        Metric(
            id="M3-support",
            name="Citation validity: ссылка подтверждает утверждение",
            value=citation_support,
            target="≥ 90 %",
            reached=_status(citation_support, 0.90),
            detail=f"подтверждающих ссылок {supported} из {citations} (судья)",
        ),
        Metric(
            id="M4",
            name="Отказ при отсутствии ответа",
            value=_share(sum(item.refusal_ok for item in no_answer), len(no_answer)),
            target="≥ 80 %",
            reached=_status(_share(sum(item.refusal_ok for item in no_answer), len(no_answer)), 0.80),
            detail=f"{sum(item.refusal_ok for item in no_answer)} из {len(no_answer)}",
        ),
        Metric(
            id="M5",
            name="Обработка противоречий",
            value=_share(sum(item.conflict_ok for item in contradiction), len(contradiction)),
            target="≥ 70 %",
            reached=_status(
                _share(sum(item.conflict_ok for item in contradiction), len(contradiction)), 0.70
            ),
            detail=f"{sum(item.conflict_ok for item in contradiction)} из {len(contradiction)}",
        ),
        Metric(
            id="M6",
            name="Первый сигнал, p95 (с)",
            value=percentile(first_signals, P95),
            target=f"≤ {first_signal_budget_s:.0f} с",
            reached=_status(percentile(first_signals, P95), first_signal_budget_s, greater_is_better=False),
            detail=f"замеров {len(first_signals)}",
        ),
        Metric(
            id="M7",
            name="Answer correctness, simple (1–5)",
            value=(sum(simple_scores) / len(simple_scores)) if simple_scores else None,
            target="≥ 4.0",
            reached=_status((sum(simple_scores) / len(simple_scores)) if simple_scores else None, 4.0),
            detail=f"оценено вопросов {len(simple_scores)} из {simple_total}",
        ),
        Metric(
            id="M8",
            name="Инжест: доля обработанных файлов",
            value=ingest_share,
            target="≥ 95 %",
            reached=_status(ingest_share, 0.95),
            detail="из отчёта инжеста" if ingest_share is not None else "отчёт инжеста не найден",
        ),
    ]
    return EvalMetrics(
        metrics=metrics,
        counts={category: len(items) for category, items in sorted(by_category.items())},
        judged=len(judged),
        unjudged=len(done) - len(judged),
        failed=len(outcomes) - len(done),
    )


def run_payload(
    outcomes: Sequence[QuestionOutcome], metrics: EvalMetrics, *, corpus: str, synthetic: bool
) -> dict[str, Any]:
    """Машиночитаемый отчёт: метрики и итог по каждому вопросу (по нему же считается diff)."""
    return {
        "created_at": dt.datetime.now(dt.UTC).isoformat(),
        "corpus": corpus,
        "synthetic": synthetic,
        "metrics": [metric.model_dump() for metric in metrics.metrics],
        "counts": metrics.counts,
        "judged": metrics.judged,
        "unjudged": metrics.unjudged,
        "failed": metrics.failed,
        "questions": [
            {
                **outcome.model_dump(),
                "hit": outcome.hit,
                "found_doc_ids": outcome.found_doc_ids,
                "citations": outcome.citations,
            }
            for outcome in outcomes
        ],
    }


def _question_row(outcome: QuestionOutcome) -> str:
    judge = outcome.judged
    score = f"{judge.correctness}" if judge else "—"
    comment = (judge.comment if judge else outcome.error or "судья не ответил").replace("|", "/")
    flags = []
    if outcome.category == "no_answer":
        flags.append("отказ" if outcome.refusal_ok else "НЕТ ОТКАЗА")
    if outcome.category == "contradiction":
        flags.append("статус учтён" if outcome.conflict_ok else "СТАТУС НЕ УЧТЁН")
    if outcome.derived:
        flags.append("расчёт")
    if outcome.unresolved:
        flags.append(f"потеряны ссылки: {len(outcome.unresolved)}")
    return (
        f"| {outcome.id} | {'да' if outcome.hit else 'нет'} | {score} | {outcome.citations} | "
        f"{outcome.seconds:.0f} с | {', '.join(flags)} | {comment[:120]} |"
    )


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _extra_lines(outcomes: Sequence[QuestionOutcome]) -> list[str]:
    """Сверх §10.3: время ответа, второй ход сценариев и вопросы на расчёт — для NFR-2 и наблюдения."""
    done = [outcome for outcome in outcomes if outcome.error is None]
    if not done:
        return []
    seconds = [outcome.seconds for outcome in done]
    tokens = [outcome.first_token_s for outcome in done if outcome.first_token_s]
    follow_ups = [
        outcome.follow_up_judge.correctness
        for outcome in done
        if outcome.follow_up_judge and outcome.follow_up_judge.parsed
    ]
    derived = [
        outcome.judge.correctness for outcome in done if outcome.derived and outcome.judged and outcome.judge
    ]
    lines = [
        "",
        "## Дополнительно (сверх §10.3)",
        "",
        f"- Полный ответ: медиана {percentile(seconds, 0.5):.0f} с, p95 {percentile(seconds, P95):.0f} с "
        f"(NFR-2: простой вопрос ≤ 45 с, сложный ≤ 120 с)",
    ]
    if tokens:
        lines.append(
            f"- Первый токен ответа: медиана {percentile(tokens, 0.5):.0f} с, "
            f"p95 {percentile(tokens, P95):.0f} с"
        )
    if follow_ups:
        mean = _mean([float(item) for item in follow_ups])
        lines.append(
            f"- Второй ход сценариев clarification: средняя оценка судьи {mean:.2f} по {len(follow_ups)}"
        )
    if derived:
        mean = _mean([float(item) for item in derived])
        lines.append(f"- Вопросы на расчёт (derived): средняя оценка судьи {mean:.2f} по {len(derived)}")
    return lines


def render_markdown(
    outcomes: Sequence[QuestionOutcome],
    metrics: EvalMetrics,
    *,
    corpus: str,
    synthetic: bool,
    diff: Sequence[str] = (),
) -> str:
    """Отчёт для людей: метрики с целями, срез по категориям, таблица вопросов, diff с прошлым прогоном."""
    lines = [
        "# Отчёт eval-harness",
        "",
        f"- Корпус: `{corpus}`",
        f"- Вопросов: {len(outcomes)} ({', '.join(f'{k}: {v}' for k, v in metrics.counts.items())})",
        f"- Оценено судьёй: {metrics.judged}; без вердикта судьи: {metrics.unjudged}; "
        f"с ошибкой прогона: {metrics.failed}",
    ]
    if synthetic:
        lines.append("- **ДАННЫЕ СИНТЕТИЧЕСКИЕ: метрики M1–M8 не считаются достигнутыми (§9 ТЗ)**")
    lines += _extra_lines(outcomes)
    lines += [
        "",
        "## Метрики (§10.3)",
        "",
        "| ID | Метрика | Значение | Цель | Итог | Пояснение |",
        "|---|---|---|---|---|---|",
    ]
    for metric in metrics.metrics:
        lines.append(
            f"| {metric.id} | {metric.name} | {metric.rendered()} | {metric.target} | "
            f"{metric.reached} | {metric.detail} |"
        )
    lines += [
        "",
        "## Вопросы",
        "",
        "| id | документы найдены | судья | ссылок | время | пометки | комментарий |",
        "|---|---|---|---|---|---|---|",
    ]
    lines += [_question_row(outcome) for outcome in outcomes]
    if diff:
        lines += ["", "## Изменения относительно прошлого прогона", ""]
        lines += [f"- {line}" for line in diff]
    return "\n".join(lines) + "\n"


def diff_runs(previous: dict[str, Any] | None, current: dict[str, Any]) -> list[str]:
    """Что изменилось с прошлого прогона: значения метрик и итоги по вопросам."""
    if not previous:
        return []
    lines: list[str] = []
    old_metrics = {item["id"]: item for item in previous.get("metrics", [])}
    for metric in current.get("metrics", []):
        old = old_metrics.get(metric["id"])
        if not old or old.get("value") is None or metric.get("value") is None:
            continue
        delta = float(metric["value"]) - float(old["value"])
        if abs(delta) >= 0.005:
            lines.append(
                f"{metric['id']}: {float(old['value']):.2f} → {float(metric['value']):.2f} ({delta:+.2f})"
            )
    old_questions = {item["id"]: item for item in previous.get("questions", [])}
    for question in current.get("questions", []):
        old = old_questions.get(question["id"])
        if not old:
            lines.append(f"{question['id']}: новый вопрос")
            continue
        if bool(old.get("hit")) != bool(question.get("hit")):
            lines.append(
                f"{question['id']}: документы найдены {'нет → да' if question['hit'] else 'да → нет'}"
            )
        old_score = (old.get("judge") or {}).get("correctness")
        new_score = (question.get("judge") or {}).get("correctness")
        if old_score and new_score and abs(int(new_score) - int(old_score)) >= 1:
            lines.append(f"{question['id']}: оценка судьи {old_score} → {new_score}")
    missing = set(old_questions) - {item["id"] for item in current.get("questions", [])}
    lines += [f"{item}: вопроса больше нет в сете" for item in sorted(missing)]
    return lines


def latest_report(paths: Iterable[Any]) -> Any | None:
    """Самый свежий json-отчёт прошлого прогона (по имени файла)."""
    reports = sorted(paths)
    return reports[-1] if reports else None
