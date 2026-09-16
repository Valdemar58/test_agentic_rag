"""Память диалога (6.4, FR-6): буфер последних ходов и сводка старых через LLM роли summary."""

from __future__ import annotations

from agent.memory import ConversationMemory, Turn, render_history
from agent.prompts import NO_SUMMARY
from common.config import DEFAULT_CONFIG_PATH, load_app_config
from tests.unit.agent_fakes import ScriptedLLM

CONFIG = load_app_config(DEFAULT_CONFIG_PATH)
MEMORY = CONFIG.agent.memory.model_copy(update={"buffer_messages": 4, "summary_trigger_chars": 1000})


def _turns(count: int, *, answer: str = "ответ") -> list[Turn]:
    return [Turn(question=f"вопрос {index}", answer=f"{answer} {index}") for index in range(1, count + 1)]


async def test_old_turns_are_summarized_and_summary_accumulates() -> None:
    memory = ConversationMemory()
    for turn in _turns(3):
        memory.add(turn)
    assert MEMORY.buffer_turns == 2 and memory.needs_compaction(MEMORY)
    llm = ScriptedLLM(steps=["Сводка: спрашивали 1.", "Сводка: спрашивали 1 и 2."])

    assert await memory.compact(llm, MEMORY)
    assert memory.summary == "Сводка: спрашивали 1." and [t.question for t in memory.turns] == [
        "вопрос 2",
        "вопрос 3",
    ]
    system, user = llm.inputs[0]
    assert str(MEMORY.summary_max_words) in str(system.content)
    assert f"Предыдущая сводка:\n{NO_SUMMARY}" in str(user.content)
    assert "Пользователь: вопрос 1" in str(user.content) and "вопрос 2" not in str(user.content)
    assert not memory.needs_compaction(MEMORY)

    memory.add(Turn(question="вопрос 4", answer="ответ 4"))
    assert await memory.compact(llm, MEMORY)
    assert memory.summary == "Сводка: спрашивали 1 и 2." and memory.compactions == 2
    assert "Предыдущая сводка:\nСводка: спрашивали 1." in str(llm.inputs[1][1].content)
    assert [t.question for t in memory.turns] == ["вопрос 3", "вопрос 4"]


async def test_large_answers_trigger_compaction_by_chars() -> None:
    memory = ConversationMemory()
    for turn in _turns(2, answer="о" * 800):
        memory.add(turn)
    assert memory.needs_compaction(MEMORY) and [t.question for t in memory.overflow(MEMORY)] == ["вопрос 1"]
    assert await memory.compact(ScriptedLLM(steps=["Сводка."]), MEMORY)
    assert memory.summary == "Сводка." and [t.question for t in memory.turns] == ["вопрос 2"]


async def test_empty_summary_keeps_turns_and_no_compaction_within_buffer() -> None:
    memory = ConversationMemory()
    for turn in _turns(2):
        memory.add(turn)
    llm = ScriptedLLM()
    assert not await memory.compact(llm, MEMORY) and not llm.inputs, "в пределах буфера LLM не вызывается"
    memory.add(Turn(question="вопрос 3", answer="ответ 3"))
    assert not await memory.compact(ScriptedLLM(steps=["   "]), MEMORY)
    assert memory.summary is None and len(memory.turns) == 3


def test_history_rendering_includes_summary_and_recent_turns() -> None:
    memory = ConversationMemory()
    memory.summary = "Обсуждали отчётность."
    memory.add(Turn(question="Когда?", answer="До пятого."))
    text = render_history(memory.recent(5), answer_chars=100, summary=memory.summary)
    assert (
        text
        == "Сводка предыдущего диалога: Обсуждали отчётность.\nПользователь: Когда?\nАссистент: До пятого."
    )
