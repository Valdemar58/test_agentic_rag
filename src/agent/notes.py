"""Заметки цикла инструментов → ориентир для шага ответа: только факты с псевдонимами.

Живые диалоги 2026-09-16/17: шаг заметок (LLM без размышлений) в 5 из 29 случаев дописывал собственный
вывод («таким образом, вернуться не позднее 13:00»), а шаг ответа принимал его за факт документа и подпирал
ссылками. Промпт просит факты строками с псевдонимами; здесь это правило закрепляется детерминированно:
предложение без псевдонима фрагмента или документа из реестра и предложение, начинающееся с вводного
слова вывода, в промпт ответа не попадают. Заметки целиком остаются в трейсе и в шаге «Итоги поиска» UI.
"""

from __future__ import annotations

import re
from collections.abc import Collection

from agent.evidence import DOC_PREFIX, FRAGMENT_PREFIX

ALIAS_TOKEN_RE = re.compile(rf"\b((?:{DOC_PREFIX}|{FRAGMENT_PREFIX})\d+)\b")
# граница предложения — только перед заглавной буквой или кавычкой, чтобы не резать «п. 6.3» и «№ 176»
_SENTENCE_RE = re.compile(r"(?<=[.!?;])\s+(?=[А-ЯЁA-Z«\"(])")
INFERENCE_STARTS = ("таким образом", "следовательно", "значит", "итак", "это означает", "то есть", "вывод")
BULLETS = "-–—•*·"


def facts_only(notes: str, known_aliases: Collection[str]) -> str:
    """Предложения заметок, которые ссылаются на известные псевдонимы и не начинаются с вывода."""
    known = set(known_aliases)
    lines: list[str] = []
    for raw in notes.splitlines():
        kept: list[str] = []
        for sentence in _SENTENCE_RE.split(raw.strip()):
            text = sentence.strip()
            head = text.lstrip(BULLETS).strip().casefold()
            if not text or head.startswith(INFERENCE_STARTS):
                continue
            if not (set(ALIAS_TOKEN_RE.findall(text)) & known):
                continue
            kept.append(text)
        if kept:
            lines.append(" ".join(kept))
    return "\n".join(lines)
