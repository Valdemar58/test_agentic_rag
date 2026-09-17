"""Ориентир для ответа из заметок цикла: только предложения с псевдонимами и без вводных слов вывода."""

from __future__ import annotations

from agent.notes import facts_only

KNOWN = ["S1", "S2", "D1"]


def test_sentences_without_known_aliases_and_conclusions_are_dropped() -> None:
    notes = (
        "Найдены S1 и S2 из документа D1 (действует). Продолжительность перерыва составляет 45 минут.\n"
        "— перерыв 45 минут в диапазоне с 12:00 до 15:00 [S2]\n"
        "— Таким образом, вернуться нужно не позднее 15:00 [S2]\n"
        "Это означает, что перерыв заканчивается в 13:00 [S1].\n"
        "Следовательно, ответ ясен.\n"
        "Пункт п. 6.3 описывает перерыв [S1]. Итак, всё найдено.\n"
        "Фрагмент S9 не из этого прогона."
    )
    assert facts_only(notes, KNOWN).splitlines() == [
        "Найдены S1 и S2 из документа D1 (действует).",
        "— перерыв 45 минут в диапазоне с 12:00 до 15:00 [S2]",
        "Пункт п. 6.3 описывает перерыв [S1].",
    ]


def test_empty_or_alias_free_notes_give_empty_orientation() -> None:
    assert facts_only("", KNOWN) == ""
    assert facts_only("Поиск по запросу ничего не нашёл.", KNOWN) == ""
    assert facts_only("Ничего не найдено [S1]", []) == ""
