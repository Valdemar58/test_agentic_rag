"""Кликабельные цитаты (FR-4, AC-4.2): элемент с текстом фрагмента или карточки для каждой ссылки [1], [2].

Имя элемента совпадает с маркером ссылки в тексте ответа — так Chainlit превращает маркер в ссылку,
открывающую элемент сбоку. Содержимое — реквизиты документа, статус, путь к разделу и текст именно
того фрагмента, на который сослался ответ.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from agent.citations import SED_LINK_TITLE, Source

# Имя сводного элемента не должно встречаться в тексте ответа: иначе Chainlit превратит заголовок
# блока «Источники» в ссылку-бейдж
SOURCES_NAME = "Список источников"
DOCUMENT_FALLBACK = "Документ"
NO_TEXT = "(текст фрагмента недоступен)"
CARD_TITLE = "Карточка документа СЭД:"
CARD_NOT_REQUESTED = "Карточка документа не запрашивалась: сведения из результатов поиска."
CONTEXT_TITLE = "*Контекст раздела:*"
SEPARATOR = "---"


@dataclass(frozen=True)
class Citation:
    name: str
    content: str


def citation_name(number: int) -> str:
    return f"[{number}]"


def citation_content(source: Source) -> str:
    status = f" — {source.doc_status}" if source.doc_status else ""
    lines = [f"**{source.label or DOCUMENT_FALLBACK}**{status}"]
    if source.url:
        lines.append(f"[{SED_LINK_TITLE}]({source.url})")
    if source.kind == "document":
        lines.append("")
        if source.text:
            lines += [CARD_TITLE, "", source.text]
        else:
            lines.append(CARD_NOT_REQUESTED)
        return "\n".join(lines)
    if source.breadcrumbs:
        lines.append(f"Раздел: {source.breadcrumbs}")
    if source.clause:
        lines.append(f"Пункт: {source.clause}")
    if source.page_no:
        lines.append(f"Страница: {source.page_no}")
    lines += ["", SEPARATOR, "", source.text or NO_TEXT]
    if source.context and source.context.strip() != (source.text or "").strip():
        lines += ["", SEPARATOR, "", CONTEXT_TITLE, "", source.context]
    return "\n".join(lines)


def sources_summary(sources: Sequence[Source]) -> Citation:
    """Сводный элемент: Chainlit после ответа сам открывает боковую панель со всеми элементами ответа
    и называет её именем последнего — так панель называется «Список источников», а не «[6]»."""
    return Citation(name=SOURCES_NAME, content="\n".join(f"- {source.line()}" for source in sources))


def citations_for(sources: Sequence[Source]) -> list[Citation]:
    """Элементы ответа: по одному на ссылку [1], [2]… и сводный список источников последним."""
    if not sources:
        return []
    items = [
        Citation(name=citation_name(source.number), content=citation_content(source)) for source in sources
    ]
    return [*items, sources_summary(sources)]
