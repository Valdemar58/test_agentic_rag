"""Схема голден-сета (9.2, §10.1): поля, правила категорий, квоты; проверка реального сета, если он есть."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from common.config import ROOT
from eval.golden_set import (
    MIN_DERIVED,
    MIN_QUESTIONS,
    QUOTAS,
    GoldenSetError,
    load_golden_set,
    parse_golden_set,
)

EXAMPLE = ROOT / "eval" / "golden_set.example.yaml"
REAL = ROOT / "eval" / "golden_set.yaml"


def _question(**overrides: Any) -> dict[str, Any]:
    question = {
        "id": "simple-01",
        "category": "simple",
        "question": "Когда сдаётся отчёт?",
        "expected_answer": "До пятого числа.",
        "expected_doc_ids": ["doc-1"],
    }
    question.update(overrides)
    return question


def _set(*questions: dict[str, Any]) -> dict[str, Any]:
    return {"corpus": "data/corpus", "synthetic": True, "questions": list(questions)}


def test_example_file_is_valid_and_shows_the_format() -> None:
    golden = load_golden_set(EXAMPLE)
    assert golden.synthetic, "пример помечен как синтетический (§9 ТЗ)"
    assert set(golden.counts) == set(QUOTAS), "в примере есть вопрос каждой категории"
    clarification = golden.by_category("clarification")[0]
    assert clarification.follow_up and clarification.follow_up_expected
    assert [question.id for question in golden.derived] == ["simple-02"]
    # пример — образец полей, а не сет: квоты он не закрывает и честно об этом сообщает
    gaps = golden.gaps()
    assert any(str(MIN_QUESTIONS) in problem for problem in gaps)
    assert any(problem.startswith("simple:") for problem in gaps)


def test_category_rules() -> None:
    with pytest.raises(GoldenSetError, match="expected_doc_id"):
        parse_golden_set(_set(_question(category="no_answer", id="na-1")))
    with pytest.raises(GoldenSetError, match="expected_doc_id"):
        parse_golden_set(_set(_question(expected_doc_ids=[])))
    with pytest.raises(GoldenSetError, match="multi_doc"):
        parse_golden_set(_set(_question(category="multi_doc", id="md-1")))
    with pytest.raises(GoldenSetError, match="follow_up"):
        parse_golden_set(_set(_question(category="clarification", id="cl-1")))
    with pytest.raises(GoldenSetError, match="follow_up_expected"):
        parse_golden_set(_set(_question(follow_up="а кто согласовал?")))
    with pytest.raises(GoldenSetError, match="повторяющиеся id"):
        parse_golden_set(_set(_question(), _question()))
    with pytest.raises(GoldenSetError, match="category"):
        parse_golden_set(_set(_question(category="прочее")))
    with pytest.raises(GoldenSetError, match="лишн|extra|не прошёл"):
        parse_golden_set(_set(_question(unexpected="поле не по схеме")))


def test_gaps_are_empty_on_a_complete_set() -> None:
    questions: list[dict[str, Any]] = []
    for category, quota in QUOTAS.items():
        for number in range(quota):
            extra: dict[str, Any] = {"expected_doc_ids": ["doc-1", "doc-2"]}
            if category == "no_answer":
                extra = {"expected_doc_ids": []}
            if category == "clarification":
                extra |= {"follow_up": "а в филиалах?", "follow_up_expected": "то же правило"}
            questions.append(
                _question(id=f"{category}-{number:02d}", category=category, derived=number < 1, **extra)
            )
    golden = parse_golden_set(_set(*questions))
    assert len(golden.questions) >= MIN_QUESTIONS
    assert len(golden.derived) >= MIN_DERIVED
    assert golden.gaps() == []


def test_missing_file_and_broken_yaml(tmp_path: Path) -> None:
    with pytest.raises(GoldenSetError, match="не найден"):
        load_golden_set(tmp_path / "нет.yaml")
    broken = tmp_path / "broken.yaml"
    broken.write_text("questions: [ {id: x", encoding="utf-8")
    with pytest.raises(GoldenSetError, match="YAML"):
        load_golden_set(broken)
    empty = tmp_path / "list.yaml"
    empty.write_text("- вопрос\n", encoding="utf-8")
    with pytest.raises(GoldenSetError, match="словарь"):
        load_golden_set(empty)


@pytest.mark.skipif(not REAL.exists(), reason="реальный голден-сет не собран (он вне git)")
def test_real_golden_set_matches_the_requirements() -> None:
    golden = load_golden_set(REAL)
    assert not golden.synthetic, "сет по реальному корпусу: метрики считаются только на нём"
    assert golden.gaps() == [], f"состав не по §10.1: {golden.gaps()}"
    ids = {question.id for question in golden.questions}
    assert len(ids) == len(golden.questions)
    raw = yaml.safe_load(REAL.read_text(encoding="utf-8"))
    assert isinstance(raw, dict) and raw.get("questions"), "файл читается как YAML верхнего уровня"
