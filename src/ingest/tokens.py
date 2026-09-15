"""Подсчёт токенов для чанкинга: тем же токенайзером, которым bge-m3 читает чанк (XLM-R)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol


class TokenCounter(Protocol):
    def count(self, text: str) -> int: ...


class WordTokenCounter:
    """Слова как токены — для тестов и проверок без весов модели."""

    def count(self, text: str) -> int:
        return len(text.split())


class HfTokenCounter:
    """Токенайзер модели эмбеддингов из локального каталога весов (офлайн, без спецтокенов)."""

    def __init__(self, model_dir: Path) -> None:
        self._model_dir = model_dir
        self._tokenizer: Any = None

    def _load(self) -> Any:
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(str(self._model_dir), local_files_only=True)
        return self._tokenizer

    def count(self, text: str) -> int:
        if not text:
            return 0
        return len(self._load().encode(text, add_special_tokens=False))
