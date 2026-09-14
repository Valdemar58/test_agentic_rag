"""Обезличивает сырой ответ Тессы cards/get для использования как тестовой фикстуры.

Что делает:
- собирает ФИО из полей карточки (CreatedByName, AuthorName, UserName и т. п.), группирует
  варианты написания одного человека по фамилии и заменяет их псевдонимами «С.С. СотрудникN»
  во всех строковых значениях документа, включая свободный текст истории согласования;
- обнуляет служебные блоки Info (подписанные токены сервера), они для RAG не нужны;
- проверяет, что ни одно исходное ФИО в результате не осталось.

Запуск: uv run python scripts/anonymize_card_example.py <вход.json> <выход.json>
Исходный файл с реальными ФИО не коммитится (data/ в .gitignore).
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

NAME_KEYS = frozenset(
    {
        "CreatedByName",
        "ModifiedByName",
        "AuthorName",
        "RegistratorName",
        "SignedByName",
        "UserName",
        "PerformerName",
        "CompletedByName",
        "ProcessOwnerName",
        "ApprovedBy",
        "DisapprovedBy",
        "ControllerName",
        "ResponsibleName",
    }
)
SYSTEM_NAMES = frozenset({"System", "Система"})
MIN_SURNAME_LENGTH = 4
# ФИО в свободном тексте: «Фамилия И.О.» и «И.О. Фамилия»
NAME_IN_TEXT = re.compile(
    r"[А-ЯЁ][а-яё]+(?:-[А-ЯЁ][а-яё]+)?\s[А-ЯЁ]\.\s?[А-ЯЁ]\.|[А-ЯЁ]\.\s?[А-ЯЁ]\.\s[А-ЯЁ][а-яё]+(?:-[А-ЯЁ][а-яё]+)?"
)


def _split_key(key: str) -> str:
    return key.split("::", 1)[0]


def _collect_names(node: Any, names: set[str]) -> None:
    if isinstance(node, dict):
        for key, value in node.items():
            if _split_key(key) in NAME_KEYS and isinstance(value, str):
                for part in value.split(","):
                    cleaned = part.strip()
                    if cleaned and cleaned not in SYSTEM_NAMES:
                        names.add(cleaned)
            _collect_names(value, names)
    elif isinstance(node, list):
        for item in node:
            _collect_names(item, names)
    elif isinstance(node, str):
        for match in NAME_IN_TEXT.findall(node):
            names.add(match.strip())


def _surname(full_name: str) -> str:
    tokens = [token for token in re.split(r"[\s.]+", full_name) if len(token) >= MIN_SURNAME_LENGTH]
    return max(tokens, key=len) if tokens else full_name


def build_replacements(names: set[str]) -> list[tuple[str, str]]:
    """Пары «исходная строка → псевдоним», длинные строки первыми."""
    surnames = sorted({_surname(name) for name in names})
    person_index = {surname: index + 1 for index, surname in enumerate(surnames)}
    replacements: list[tuple[str, str]] = []
    for name in names:
        index = person_index[_surname(name)]
        replacements.append((name, f"С.С. Сотрудник{index}"))
    for surname, index in person_index.items():
        replacements.append((surname, f"Сотрудник{index}"))
    replacements.sort(key=lambda pair: len(pair[0]), reverse=True)
    return replacements


def _apply(node: Any, replacements: list[tuple[str, str]]) -> Any:
    if isinstance(node, dict):
        return {key: _apply(value, replacements) for key, value in node.items()}
    if isinstance(node, list):
        return [_apply(item, replacements) for item in node]
    if isinstance(node, str):
        result = node
        for source, target in replacements:
            result = result.replace(source, target)
        return result
    return node


def anonymize(payload: dict[str, Any]) -> tuple[dict[str, Any], int]:
    """Возвращает обезличенную копию и число заменённых персон."""
    names: set[str] = set()
    _collect_names(payload, names)
    replacements = build_replacements(names)
    result = _apply(payload, replacements)
    result["Info"] = None
    card = result.get("Card")
    if isinstance(card, dict):
        card["Info"] = None
    serialized = json.dumps(result, ensure_ascii=False)
    leftovers = [source for source, _ in replacements if source in serialized]
    if leftovers:
        raise RuntimeError(f"В результате остались исходные строки: {len(leftovers)}")
    persons = len({_surname(name) for name in names})
    return result, persons


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__)
        return 2
    source = Path(argv[1])
    target = Path(argv[2])
    payload = json.loads(source.read_text(encoding="utf-8"))
    result, persons = anonymize(payload)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"Обезличено персон: {persons}. Результат: {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
