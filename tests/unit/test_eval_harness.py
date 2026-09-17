"""Eval-harness (9.3, §10.2): отбор вопросов, прогон с фейковым агентом и судьёй, метрики, отчёт, diff."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from agent.citations import Source
from eval.golden_set import GoldenQuestion, GoldenSet, parse_golden_set
from eval.judge import JudgeVerdict, parse_verdict, render_sources
from eval.metrics import (
    QuestionOutcome,
    compute_metrics,
    diff_runs,
    percentile,
    render_markdown,
    run_payload,
)
from eval.run import ingest_success_share, run_eval, select

BUDGET_S = 5.0


def _source(number: int, doc_id: str, *, status: str = "действует", text: str = "текст пункта") -> Source:
    return Source(
        number=number,
        alias=f"S{number}",
        kind="fragment",
        doc_id=doc_id,
        chunk_id=f"chunk-{number}",
        label="Приказ №144 от 15.01.2026",
        doc_status=status,
        breadcrumbs="Приказ №144 → Раздел 1 → п. 1.1",
        text=text,
    )


def _question(**overrides: Any) -> GoldenQuestion:
    data: dict[str, Any] = {
        "id": "simple-01",
        "category": "simple",
        "question": "Когда сдаётся отчёт?",
        "expected_answer": "До пятого числа.",
        "expected_doc_ids": ["doc-1"],
    }
    data.update(overrides)
    return GoldenQuestion.model_validate(data)


def _outcome(**overrides: Any) -> QuestionOutcome:
    data: dict[str, Any] = {
        "id": "simple-01",
        "category": "simple",
        "question": "Когда сдаётся отчёт?",
        "answer": "Отчёт сдаётся до пятого числа [1].",
        "expected_doc_ids": ["doc-1"],
        "sources": [_source(1, "doc-1")],
        "first_signal_s": 0.01,
        "seconds": 20.0,
        "judge": JudgeVerdict(correctness=5, statements=2, with_citation=2, citations=1, supported=1),
    }
    data.update(overrides)
    return QuestionOutcome.model_validate(data)


def _golden(*questions: GoldenQuestion) -> GoldenSet:
    return parse_golden_set(
        {
            "corpus": "data/corpus",
            "synthetic": True,
            "questions": [question.model_dump() for question in questions],
        }
    )


def test_select_by_id_category_and_limit() -> None:
    golden = _golden(
        _question(),
        _question(id="simple-02"),
        _question(id="na-01", category="no_answer", expected_doc_ids=[]),
    )
    assert [item.id for item in select(golden)] == ["simple-01", "simple-02", "na-01"]
    assert [item.id for item in select(golden, only=["no_answer"])] == ["na-01"]
    assert [item.id for item in select(golden, only=["simple-02", "na-01"])] == ["simple-02", "na-01"]
    assert [item.id for item in select(golden, limit=2)] == ["simple-01", "simple-02"]


def test_hit_refusal_and_conflict_are_deterministic() -> None:
    assert _outcome().hit
    assert not _outcome(expected_doc_ids=["doc-1", "doc-2"]).hit
    assert _outcome(
        expected_doc_ids=["doc-1", "doc-2"], sources=[_source(1, "doc-1"), _source(2, "doc-2")]
    ).hit

    # отказ: по явному признаку раннера или по вердикту судьи
    assert _outcome(category="no_answer", refused=True, expected_doc_ids=[]).refusal_ok
    assert _outcome(
        category="no_answer",
        expected_doc_ids=[],
        judge=JudgeVerdict(correctness=5, refusal=True),
    ).refusal_ok
    assert not _outcome(category="no_answer", expected_doc_ids=[]).refusal_ok

    # противоречие: опора только на действующие документы или прямое упоминание отмены
    assert _outcome(category="contradiction").conflict_ok
    cancelled = _outcome(category="contradiction", sources=[_source(1, "doc-1", status="отменён")])
    assert not cancelled.conflict_ok
    assert cancelled.model_copy(update={"answer": "Приказ отменён приказом № 149 [1]."}).conflict_ok
    # ссылки на посторонний действующий документ мало: нужного документа в источниках нет (живой прогон)
    aside = _outcome(category="contradiction", sources=[_source(1, "doc-9")])
    assert not aside.conflict_ok


def test_metrics_follow_the_targets() -> None:
    outcomes = [_outcome(id=f"simple-{index:02d}", seconds=20.0 + index) for index in range(1, 5)]
    outcomes.append(
        _outcome(id="simple-05", sources=[_source(1, "doc-9")], judge=JudgeVerdict(correctness=2))
    )
    outcomes += [
        _outcome(
            id="multi-01",
            category="multi_doc",
            expected_doc_ids=["doc-1", "doc-2"],
            sources=[_source(1, "doc-1"), _source(2, "doc-2")],
        ),
        _outcome(id="na-01", category="no_answer", expected_doc_ids=[], sources=[], refused=True),
        _outcome(id="na-02", category="no_answer", expected_doc_ids=[], sources=[], refused=False),
        _outcome(id="contra-01", category="contradiction"),
        _outcome(id="broken", error="RuntimeError: стенд недоступен"),
    ]
    metrics = compute_metrics(outcomes, first_signal_budget_s=BUDGET_S, ingest_share=0.991)

    simple = metrics.by_id("M1-simple")
    assert simple and simple.value == pytest.approx(4 / 5) and simple.reached == "не достигнута"
    assert metrics.by_id("M1-multi_doc").value == 1.0  # type: ignore[union-attr]
    assert metrics.by_id("M4").value == pytest.approx(0.5)  # type: ignore[union-attr]
    assert metrics.by_id("M4").reached == "не достигнута"  # type: ignore[union-attr]
    assert metrics.by_id("M5").value == 1.0  # type: ignore[union-attr]
    assert metrics.by_id("M6").reached == "достигнута"  # type: ignore[union-attr]
    m7 = metrics.by_id("M7")
    assert m7 and m7.value == pytest.approx((5 * 4 + 2) / 5) and m7.reached == "достигнута"
    assert metrics.by_id("M8").value == pytest.approx(0.991)  # type: ignore[union-attr]
    assert metrics.failed == 1 and metrics.judged == 9


def test_unjudged_answers_do_not_lower_metrics() -> None:
    outcomes = [
        _outcome(id="simple-01"),
        _outcome(id="simple-02", judge=JudgeVerdict(parsed=False)),
    ]
    metrics = compute_metrics(outcomes, first_signal_budget_s=BUDGET_S)
    m7 = metrics.by_id("M7")
    assert m7 and m7.value == 5.0 and "1 из 2" in m7.detail
    assert metrics.judged == 1 and metrics.unjudged == 1


def test_percentile_and_first_signal_budget() -> None:
    assert percentile([], 0.95) is None
    assert percentile([1.0], 0.95) == 1.0
    assert percentile([1.0, 2.0, 3.0, 10.0], 0.95) == 10.0
    slow = [_outcome(id=f"q{index}", first_signal_s=6.0) for index in range(3)]
    assert compute_metrics(slow, first_signal_budget_s=BUDGET_S).by_id("M6").reached == "не достигнута"  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_run_eval_calls_judge_and_keeps_order() -> None:
    questions = [
        _question(),
        _question(id="cl-01", category="clarification", follow_up="а кто?", follow_up_expected="директор"),
    ]
    seen: list[str] = []

    async def ask(question: GoldenQuestion) -> QuestionOutcome:
        seen.append(question.id)
        return _outcome(
            id=question.id, category=question.category, judge=None, follow_up_answer="Директор [1]."
        )

    async def judge(
        question: GoldenQuestion, outcome: QuestionOutcome
    ) -> tuple[JudgeVerdict, JudgeVerdict | None]:
        follow_up = JudgeVerdict(correctness=4) if question.follow_up else None
        return JudgeVerdict(correctness=5, statements=1, with_citation=1, citations=1, supported=1), follow_up

    outcomes = await run_eval(questions, ask=ask, judge=judge)
    assert seen == ["simple-01", "cl-01"]
    assert [outcome.judge.correctness for outcome in outcomes if outcome.judge] == [5, 5]
    assert outcomes[1].follow_up_judge and outcomes[1].follow_up_judge.correctness == 4


@pytest.mark.asyncio
async def test_run_eval_survives_a_broken_question() -> None:
    async def ask(question: GoldenQuestion) -> QuestionOutcome:
        return _outcome(id=question.id, error="RuntimeError: vLLM недоступен", answer="", judge=None)

    async def judge(*_: Any) -> tuple[JudgeVerdict, JudgeVerdict | None]:
        raise AssertionError("судья не должен вызываться на упавшем вопросе")

    outcomes = await run_eval([_question()], ask=ask, judge=judge)
    assert outcomes[0].error and outcomes[0].judge is None


def test_report_and_diff(tmp_path: Path) -> None:
    outcomes = [
        _outcome(id="simple-01"),
        _outcome(id="na-01", category="no_answer", expected_doc_ids=[], sources=[], refused=True),
        _outcome(id="broken", error="RuntimeError: стенд недоступен"),
    ]
    metrics = compute_metrics(outcomes, first_signal_budget_s=BUDGET_S)
    payload = run_payload(outcomes, metrics, corpus="data/final", synthetic=False)
    markdown = render_markdown(outcomes, metrics, corpus="data/final", synthetic=False)
    assert "## Метрики (§10.3)" in markdown and "M1-simple" in markdown
    assert "simple-01" in markdown and "broken" in markdown
    assert "Дополнительно" in markdown

    previous = json.loads(json.dumps(payload))
    previous["questions"][0]["hit"] = False
    previous["questions"][0]["judge"]["correctness"] = 3
    previous["metrics"][0]["value"] = 0.5
    previous["questions"].append({"id": "ушедший", "hit": True})
    diff = diff_runs(previous, payload)
    assert any("M1-simple" in line for line in diff)
    assert any("simple-01" in line and "нет → да" in line for line in diff)
    assert any("оценка судьи 3 → 5" in line for line in diff)
    assert any("ушедший" in line for line in diff)
    assert diff_runs(None, payload) == []


def test_judge_prompt_and_parsing() -> None:
    rendered = render_sources([_source(1, "doc-1", text="Отчёт сдаётся до пятого числа.")], chars=20)
    assert "[1] Приказ №144" in rendered and rendered.endswith("…")
    assert render_sources([], chars=50) == "(ссылок в ответе нет)"

    verdict = parse_verdict(
        'Оценка: {"correctness": 4, "statements": 3, "with_citation": 3, "citations": 2, '
        '"supported": 1, "refusal": false, "conflict": true, "comment": "неполно"}'
    )
    assert verdict.parsed and verdict.correctness == 4 and verdict.supported == 1 and verdict.conflict
    assert parse_verdict('{"correctness": 9}').correctness == 5, "оценка обрезается по шкале"
    assert parse_verdict('{"correctness": -3, "statements": -1}').statements == 0
    assert not parse_verdict("судья ничего не вернул").parsed
    assert not parse_verdict('{"correctness": "пять"}').parsed


def test_ingest_share_from_report(tmp_path: Path) -> None:
    assert ingest_success_share(tmp_path) is None
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "ingest_8.json").write_text(json.dumps({"success_share": 0.5}), encoding="utf-8")
    (reports / "ingest_9.json").write_text(json.dumps({"success_share": 0.991}), encoding="utf-8")
    assert ingest_success_share(tmp_path) == pytest.approx(0.991)
    (reports / "ingest_9.json").write_text("не json", encoding="utf-8")
    assert ingest_success_share(tmp_path) is None
