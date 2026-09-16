"""`get_document_content` (5.4): разделы документа по порядку, раздел по id, бюджет и продолжение."""

from __future__ import annotations

import pytest

from common.config import DEFAULT_CONFIG_PATH, load_app_config
from mcp_server.content import NOT_INDEXED_NOTE, ContentNotFoundError, DocumentReader
from tests.unit.index_data import InMemoryCorpus

CONFIG = load_app_config(DEFAULT_CONFIG_PATH)


@pytest.fixture
def corpus() -> InMemoryCorpus:
    corpus = InMemoryCorpus()
    doc_id = corpus.add(
        "order",
        [
            "Утвердить положение.",
            "Контроль за собой.",
            "Ввести в действие с первого числа.",
            "Ознакомить всех.",
        ],
        number="144",
        sections=2,
        file_name="ДокШаблон Приказ 144.docx",
    )
    corpus.add(
        "appendix",
        ["Общие положения регламента.", "Порядок работы."],
        number="144",
        sections=2,
        file_role="appendix",
        file_name="Приложение - Регламент.docx",
        doc_id=doc_id,
    )
    corpus.add("other", ["Другой документ."], number="9")
    return corpus


def reader(corpus: InMemoryCorpus, **overrides: object) -> DocumentReader:
    return DocumentReader(corpus.index.client, CONFIG.qdrant, CONFIG.retrieval.model_copy(update=overrides))


def test_whole_document_is_returned_main_file_first_in_order(corpus: InMemoryCorpus) -> None:
    content = reader(corpus).read(corpus.docs["order"])
    assert content.label == "Приказ №144" and content.doc_status == "active" and content.note is None
    assert content.total_sections == 4 and len(content.sections) == 4 and not content.truncated
    assert [section.file_role for section in content.sections] == ["main", "main", "appendix", "appendix"]
    assert [section.heading for section in content.sections] == [
        "1. Раздел 1",
        "2. Раздел 2",
        "1. Раздел 1",
        "2. Раздел 2",
    ]
    assert "Утвердить положение" in content.sections[0].text and "Контроль" in content.sections[0].text
    assert content.sections[2].breadcrumbs.startswith(
        "Приказ №144 → Приложение «Приложение - Регламент.docx»"
    )
    assert all(section.tokens > 0 for section in content.sections)


def test_budget_truncates_and_offset_continues(corpus: InMemoryCorpus) -> None:
    small = reader(corpus, content_max_tokens=1)
    first = small.read(corpus.docs["order"])
    assert len(first.sections) == 1 and first.truncated and first.next_offset == 1
    second = small.read(corpus.docs["order"], offset=first.next_offset or 0)
    assert second.offset == 1 and second.sections[0].section_id != first.sections[0].section_id
    last = small.read(corpus.docs["order"], offset=3)
    assert not last.truncated and last.next_offset is None and len(last.sections) == 1
    beyond = small.read(corpus.docs["order"], offset=99)
    assert beyond.offset == 3 and len(beyond.sections) == 1
    by_count = reader(corpus, content_max_sections=2).read(corpus.docs["order"])
    assert len(by_count.sections) == 2 and by_count.next_offset == 2


def test_section_by_parent_or_child_id(corpus: InMemoryCorpus) -> None:
    whole = reader(corpus).read(corpus.docs["order"])
    section_id = whole.sections[1].section_id
    single = reader(corpus).read(corpus.docs["order"], section_id=section_id)
    assert single.sections[0].section_id == section_id and single.total_sections == 1
    assert "Ввести в действие" in single.sections[0].text
    # id child-чанка тоже принимается — возвращается его раздел
    hit = corpus.searcher().search("ввести в действие", top_k=1).hits[0]
    via_child = reader(corpus).read(corpus.docs["order"], section_id=hit.chunk_id)
    assert via_child.sections[0].section_id == hit.parent_id == section_id


def test_missing_document_and_sections_are_reported(corpus: InMemoryCorpus) -> None:
    empty = reader(corpus).read("00000000-0000-0000-0000-000000000000")
    assert empty.sections == [] and empty.note == NOT_INDEXED_NOTE and empty.total_sections == 0
    with pytest.raises(ContentNotFoundError, match="не найден"):
        reader(corpus).read(corpus.docs["order"], section_id="00000000-0000-0000-0000-000000000000")
    with pytest.raises(ContentNotFoundError, match="не найден"):
        reader(corpus).read(corpus.docs["order"], section_id="не-uuid")
    other = reader(corpus).read(corpus.docs["other"]).sections[0].section_id
    with pytest.raises(ContentNotFoundError, match="принадлежит документу"):
        reader(corpus).read(corpus.docs["order"], section_id=other)
