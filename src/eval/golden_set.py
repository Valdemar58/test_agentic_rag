"""Голден-сет вопросов (§10.1 ТЗ): схема, загрузка и проверка состава.

Файл YAML человекочитаемый и правится руками: список `questions`, у каждого — `id`, `category`,
`question`, `expected_answer` (эталон или ключевые факты), `expected_doc_ids` (документы, обязанные
быть в источниках) и `notes`. Сверх полей ТЗ есть `follow_up`/`follow_up_expected` — без них
двухходовый сценарий категории `clarification` не описать, — и флаг `derived` для вопросов, ответ на
которые требует расчёта из значений документа и данных пользователя (решение заказчика 2026-09-17).

Состав проверяется правилами ТЗ: не меньше `MIN_QUESTIONS` вопросов и квоты по категориям; вопрос
`no_answer` не ссылается на документы, `multi_doc` требует минимум двух, `clarification` — уточнения.
Реальный сет собран по экспортированному корпусу и лежит вне git (вопросы и эталоны содержат
реквизиты документов заказчика); в репозитории — синтетический пример того же формата.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

Category = Literal["simple", "multi_doc", "duplicated", "no_answer", "contradiction", "clarification"]

QUOTAS: dict[Category, int] = {
    "simple": 15,
    "multi_doc": 10,
    "duplicated": 5,
    "no_answer": 8,
    "contradiction": 7,
    "clarification": 5,
}
MIN_QUESTIONS = 50
# вопросов на выводимые величины (решение заказчика 2026-09-17): расчёт из документа и данных пользователя
MIN_DERIVED = 5
EXAMPLE_PATH = Path("eval/golden_set.example.yaml")


class GoldenSetError(Exception):
    """Файл голден-сета не читается, не разбирается или не проходит проверку состава."""


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class GoldenQuestion(StrictModel):
    id: str = Field(min_length=1, description="Устойчивый идентификатор, например simple-01")
    category: Category
    question: str = Field(min_length=1, description="Вопрос пользователя, как он будет задан в чате")
    expected_answer: str = Field(min_length=1, description="Эталонный ответ или ключевые факты")
    expected_doc_ids: list[str] = Field(
        default_factory=list, description="Документы, обязанные быть в источниках ответа (M1)"
    )
    follow_up: str | None = Field(default=None, description="Второй вопрос двухходового сценария")
    follow_up_expected: str | None = Field(default=None, description="Ключевые факты ответа на уточнение")
    derived: bool = Field(
        default=False, description="Ответ требует расчёта из документа и данных пользователя"
    )
    notes: str = Field(default="", description="Пояснение составителя: почему так, на что смотреть")

    @model_validator(mode="after")
    def _category_rules(self) -> GoldenQuestion:
        if self.category == "no_answer" and self.expected_doc_ids:
            raise ValueError(f"{self.id}: у вопроса без ответа в корпусе не должно быть expected_doc_ids")
        if self.category != "no_answer" and not self.expected_doc_ids:
            raise ValueError(f"{self.id}: нужен хотя бы один expected_doc_id")
        if self.category == "multi_doc" and len(self.expected_doc_ids) < 2:
            raise ValueError(f"{self.id}: multi_doc — это ответ по двум и более документам")
        if self.category == "clarification" and not self.follow_up:
            raise ValueError(f"{self.id}: clarification — двухходовый сценарий, нужен follow_up")
        if self.follow_up and not self.follow_up_expected:
            raise ValueError(f"{self.id}: к follow_up нужен follow_up_expected")
        return self


class GoldenSet(StrictModel):
    corpus: str = Field(description="Корпус, по которому составлен сет (каталог архива экспорта)")
    synthetic: bool = Field(
        default=False, description="Сет составлен по синтетическим данным: метрики не считаются (§9 ТЗ)"
    )
    questions: list[GoldenQuestion]

    @model_validator(mode="after")
    def _unique_ids(self) -> GoldenSet:
        duplicates = [item for item, count in Counter(q.id for q in self.questions).items() if count > 1]
        if duplicates:
            raise ValueError(f"повторяющиеся id: {', '.join(sorted(duplicates))}")
        return self

    @property
    def counts(self) -> dict[str, int]:
        return dict(Counter(question.category for question in self.questions))

    @property
    def derived(self) -> list[GoldenQuestion]:
        return [question for question in self.questions if question.derived]

    def by_category(self, category: Category) -> list[GoldenQuestion]:
        return [question for question in self.questions if question.category == category]

    def gaps(self) -> list[str]:
        """Чего не хватает до требований §10.1: общее число, квоты категорий, вопросы на расчёт."""
        problems: list[str] = []
        if len(self.questions) < MIN_QUESTIONS:
            problems.append(f"вопросов {len(self.questions)}, нужно не меньше {MIN_QUESTIONS}")
        counts = self.counts
        for category, quota in QUOTAS.items():
            found = counts.get(category, 0)
            if found < quota:
                problems.append(f"{category}: {found}, нужно не меньше {quota}")
        if len(self.derived) < MIN_DERIVED:
            problems.append(f"derived: {len(self.derived)}, нужно не меньше {MIN_DERIVED}")
        return problems


def parse_golden_set(raw: object, *, source: str = "<память>") -> GoldenSet:
    try:
        return GoldenSet.model_validate(raw)
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(part) for part in error['loc']) or '<корень>'}: {error['msg']}"
            for error in exc.errors()
        )
        raise GoldenSetError(f"{source} не прошёл проверку: {problems}") from exc


def load_golden_set(path: Path) -> GoldenSet:
    """Читает YAML и проверяет схему; состав категорий проверяется отдельно через `gaps()`."""
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise GoldenSetError(f"голден-сет не найден: {path}") from exc
    except yaml.YAMLError as exc:
        raise GoldenSetError(f"голден-сет {path} не разбирается как YAML: {exc}") from exc
    if not isinstance(raw, dict):
        raise GoldenSetError(f"голден-сет {path}: ожидался словарь верхнего уровня")
    return parse_golden_set(raw, source=str(path))
