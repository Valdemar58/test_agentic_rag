"""Проверка черновика: разбор замечаний проверяющего и детерминированное вычёркивание предложений."""

from __future__ import annotations

from agent.verify import VerifyProblem, apply_problems, computed_times, parse_verification, times_in

DRAFT = (
    "Прямой ответ: вернуться нужно не позднее 13:00 [S3].\n\n"
    "**Детали:**\n"
    "- Перерыв длится 45 минут в окне с 12:00 до 15:00 [S1].\n"
    "- Перерыв не включается в рабочее время [S1]. Опоздание фиксирует руководитель [S4].\n"
    "- Вернуться нужно к 13:00, потому что перерыв должен быть использован целиком в диапазоне [S3][S4]."
)


def _problem(claim: str, reason: str = "нет в свидетельствах") -> VerifyProblem:
    return VerifyProblem(claim=claim, reason=reason)


def test_verdict_json_is_parsed_with_raw_newlines_and_trailing_text() -> None:
    """Живой прогон 2026-09-17: два вердикта из шести не разобрались — сырые переводы строк в строках JSON
    и текст после закрывающей скобки."""
    text = (
        'Проверил.\n{"problems": [{"claim": "вернуться нужно\nне позднее 13:00", "reason": "нет\nв S3"}, '
        '{"claim": ""}, "мусор"]}\nПояснение. {"лишний": "объект"}'
    )
    parsed = parse_verification(text)
    assert parsed.parsed and [(p.claim, p.reason) for p in parsed.problems] == [
        ("вернуться нужно\nне позднее 13:00", "нет\nв S3")
    ]
    assert parse_verification('{"problems": []}').problems == []
    for garbage in ("", "замечаний нет", "[1, 2]", '{"problems": "не список"}', "{незакрытый"):
        result = parse_verification(garbage)
        assert not result.parsed and result.problems == [], garbage


def test_flagged_sentences_are_removed_and_formatting_survives() -> None:
    """Диалог d66bf07c… 2026-09-17: вывод «не позднее 13:00» и придуманное правило со ссылками [S3][S4]."""
    problems = [
        _problem("Прямой ответ: вернуться нужно не позднее 13:00 [S3]."),
        _problem("Вернуться нужно к 13:00, потому что перерыв должен быть использован целиком в диапазоне"),
        _problem("Опоздание фиксирует руководитель"),
    ]
    text, corrected, _ = apply_problems(DRAFT, problems, match_ratio=0.6)
    assert corrected and [problem.action for problem in problems] == ["removed", "removed", "removed"]
    assert text == (
        "**Детали:**\n- Перерыв длится 45 минут в окне с 12:00 до 15:00 [S1].\n"
        "- Перерыв не включается в рабочее время [S1]."
    ), text


def test_short_claims_match_by_similarity_and_unmatched_claims_are_reported() -> None:
    problems = [_problem("не позднее 13:00 [S3]"), _problem("о командировках"), _problem("13:00")]
    text, corrected, _ = apply_problems(DRAFT, problems, match_ratio=0.6)
    assert corrected and problems[0].action == "removed" and "не позднее 13:00" not in text
    assert problems[1].action == "unmatched" and problems[2].action == "unmatched"
    assert "45 минут" in text and "использован целиком" in text


def test_sentences_with_verified_calculation_from_the_question_are_protected() -> None:
    """Живой прогон 2026-09-17: проверяющий оспаривал верный расчёт 12:45 + 45 мин = 13:30; расчёт же от
    начала окна «12:00 + 45 мин = 12:45» при уходе в 12:45 защищать нельзя."""
    draft = (
        "Прямой ответ: вернуться нужно не позднее 13:30.\n"
        "Расчёт: 12:45 (уход) + 45 минут (перерыв) = 13:30 [S6].\n"
        "Перерыв — 45 минут в диапазоне с 12:00 до 15:00 [S6]. Перерыв согласуется с руководителем [S6]."
    )
    question = "во сколько вернуться, если я ушёл в 12.45"
    assert times_in(question) == {"12:45"} and times_in("ушёл в 12-45 и 9:05") == {"12:45", "9:05"}
    assert computed_times(draft) == {"13:30"} and computed_times("12:45 + 45 мин = 13:00") == set()
    assert (
        computed_times(draft, starts={"12:45"}) == {"13:30"} and computed_times(draft, starts=set()) == set()
    )
    problems = [
        _problem("Прямой ответ: вернуться нужно не позднее 13:30."),
        _problem("Расчёт: 12:45 (уход) + 45 минут (перерыв) = 13:30 [S6]."),
        _problem("Перерыв согласуется с руководителем [S6]."),
    ]
    text, corrected, _ = apply_problems(draft, problems, match_ratio=0.6, question=question)
    assert corrected and [problem.action for problem in problems] == ["kept", "kept", "removed"]
    assert "13:30" in text and "согласуется" not in text

    window_start = (
        "Вернуться нужно не позднее 12:45 [S6].\nРасчёт: 12:00 + 45 минут = 12:45 [S6].\n"
        "Перерыв 45 минут [S6]."
    )
    problems = [
        _problem("Вернуться нужно не позднее 12:45 [S6]."),
        _problem("Расчёт: 12:00 + 45 минут = 12:45 [S6]."),
    ]
    text, corrected, _ = apply_problems(window_start, problems, match_ratio=0.6, question=question)
    assert corrected and [problem.action for problem in problems] == ["removed", "removed"]
    assert text == "Перерыв 45 минут [S6]."


def test_claim_spanning_several_sentences_removes_all_of_them() -> None:
    """Живой прогон 2026-09-17: замечание на абзац из двух предложений вычеркнуло не то предложение."""
    draft = (
        "Во сколько вернуться: 12:45 [S6]. Продолжительность перерыва 45 минут с 12:00 до 15:00 [S6].\n\n"
        "Детали:\n- Перерыв не включается в рабочее время [S6]."
    )
    claim = "Во сколько вернуться: 12:45. Продолжительность перерыва 45 минут с 12:00 до 15:00 [S6"
    problems = [_problem(claim)]
    text, corrected, _ = apply_problems(draft, problems, match_ratio=0.6)
    assert corrected and problems[0].action == "removed"
    assert text == "Детали:\n- Перерыв не включается в рабочее время [S6]."


def test_removal_that_empties_the_text_or_drops_all_markers_is_rejected() -> None:
    draft = "Отчёт сдаётся до пятого числа [S1]. Об этом сказано в приказе."
    problems = [_problem("Отчёт сдаётся до пятого числа [S1].")]
    text, corrected, emptied = apply_problems(draft, problems, match_ratio=0.6)
    assert text == draft and not corrected and emptied and problems[0].action == "kept"
    single = apply_problems(
        "Только одно предложение [S1].", [_problem("Только одно предложение")], match_ratio=0.6
    )
    assert single == ("Только одно предложение [S1].", False, True), "отклонён весь черновик"
    assert apply_problems(draft, [], match_ratio=0.6) == (draft, False, False)


def test_partial_sentence_claims_short_neighbours_and_empty_headings() -> None:
    """Живой прогон 2026-09-17: замечание цитировало часть предложения после двоеточия, а строка
    «12:45 [S6].» и заголовок «Детали:» внутри отклонённого абзаца оставались."""
    draft = (
        "Во сколько вернуться, если вы ушли в 12:45:\n"
        "**12:45** [S6].\n\n"
        "**Детали:**\n"
        "- Согласно [S6], перерыв начинается в 12:00 и длится 45 минут. Это означает, что перерыв должен "
        "завершиться в 12:45 (12:00 + 45 минут).\n"
        "- Перерыв не включается в рабочее время [S6]."
    )
    claim = (
        "Во сколько вернуться, если вы ушли в 12:45: **12:45** [S6]. **Детали:** - Согласно [S6], перерыв "
        "начинается в 12:00 и длится 45 минут. Это означает, что перерыв должен завершиться в 12:45"
    )
    applied = apply_problems(draft, [_problem(claim)], match_ratio=0.6, question="ушёл в 12:45")
    assert applied.corrected and not applied.emptied
    assert applied.text == "- Перерыв не включается в рабочее время [S6]."

    partial = (
        "Это означает, что время возвращения определяется так: если ушли в 12:00, вернуться в 12:45 [S1]."
    )
    draft = (
        f"Перерыв 45 минут в окне с 12:00 до 15:00 [S1]. {partial}\n\n"
        "**Детали:**\n- Перерыв не включается [S1]."
    )
    applied = apply_problems(draft, [_problem("если ушли в 12:00, вернуться в 12:45")], match_ratio=0.6)
    assert applied.corrected
    assert applied.text == (
        "Перерыв 45 минут в окне с 12:00 до 15:00 [S1].\n\n**Детали:**\n- Перерыв не включается [S1]."
    )
    applied = apply_problems(
        "Ответ: 12:45 [S1].\n**Детали:**\n- вернуться в 12:45 [S1].",
        [_problem("Ответ: 12:45 [S1]. - вернуться в 12:45 [S1].")],
        match_ratio=0.6,
    )
    assert not applied.corrected and applied.emptied, "остался бы один заголовок — отклонён весь черновик"
