"""Reranker BAAI/bge-reranker-v2-m3 (FR-2.1): кросс-энкодер над парами «запрос — чанк», CPU в рантайме (§2).

Веса из локального `models/` (офлайн). Оценка — sigmoid от логита модели, 0…1. Кандидатов не
больше `retrieval.rerank_candidates`; на CPU Linear-слои квантуются в int8 (N17: замер 2026-09-16 —
20 пар за 1,5 с вместо 3,2 с при почти тех же оценках). Маленький батч быстрее большого: пары
разной длины, а батч дополняется до самой длинной.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Protocol

from common.config import RerankerSettings

logger = logging.getLogger(__name__)


class Reranker(Protocol):
    def score(self, query: str, texts: list[str]) -> list[float]: ...


class FakeReranker:
    """Доля слов запроса, встреченных в тексте — для тестов без весов модели."""

    def score(self, query: str, texts: list[str]) -> list[float]:
        words = {word for word in query.lower().split() if word}
        if not words:
            return [0.0 for _ in texts]
        return [len(words & set(text.lower().split())) / len(words) for text in texts]


def _quantize_linear_layers(model: Any) -> None:
    """Динамическое int8-квантование Linear-слоёв на месте (torch.ao; предупреждения о миграции в torchao
    подавлены — API в torch 2.x работает, переход на torchao потребует новой зависимости)."""
    import warnings

    import torch

    quantize_dynamic: Any = torch.ao.quantization.quantize_dynamic
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8, inplace=True)


class BgeReranker:
    def __init__(self, model_dir: Path, settings: RerankerSettings) -> None:
        self._model_dir = model_dir
        self._settings = settings
        self._model: Any = None

    def _load(self) -> Any:
        if self._model is None:
            from sentence_transformers import CrossEncoder

            model = CrossEncoder(
                str(self._model_dir),
                device=self._settings.device,
                local_files_only=True,
                max_length=self._settings.max_length,
            )
            quantized = self._settings.quantize_int8 and self._settings.device == "cpu"
            if quantized:
                _quantize_linear_layers(model)
            self._model = model
            logger.info(
                "Reranker загружен из %s на %s%s",
                self._model_dir,
                self._settings.device,
                " (int8 Linear-слои)" if quantized else "",
            )
        return self._model

    def score(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        scores = self._load().predict(
            [(query, text) for text in texts],
            batch_size=self._settings.batch_size,
            show_progress_bar=False,
        )
        return [float(value) for value in scores]
