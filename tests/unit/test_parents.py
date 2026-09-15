"""Parent-child (4.4): child → parent-раздел, окна для длинных разделов, уровень документа, id."""

from __future__ import annotations

from docling_core.types.doc.document import DoclingDocument
from docling_core.types.doc.labels import DocItemLabel

from common.config import DEFAULT_CONFIG_PATH, load_app_config
from ingest.chunking import StructuralChunker
from ingest.parents import build_chunk_set
from ingest.tokens import WordTokenCounter

SETTINGS = load_app_config(DEFAULT_CONFIG_PATH).ingest.chunking
ROOT = ("Приказ №144 от 15.01.2026",)
SHA = "c" * 64
COUNTER = WordTokenCounter()


def _document(clauses_per_section: int = 3, words: int = 20) -> DoclingDocument:
    doc = DoclingDocument(name="приказ")
    doc.add_heading("Об утверждении Положения", level=1)
    doc.add_text(label=DocItemLabel.TEXT, text="В целях обеспечения ПРИКАЗЫВАЮ:")
    for section in (1, 2):
        doc.add_heading(f"{section}. Раздел номер {section}", level=2)
        for clause in range(1, clauses_per_section + 1):
            body = " ".join(f"слово{i}" for i in range(words))
            doc.add_text(label=DocItemLabel.TEXT, text=f"{section}.{clause}. {body}.")
    return doc


def _chunks(doc: DoclingDocument, sha: str = SHA) -> list:  # type: ignore[type-arg]
    return StructuralChunker(SETTINGS, COUNTER).chunk(doc, ROOT, file_sha256=sha)


def test_children_point_to_section_parent_that_contains_all_clauses() -> None:
    chunk_set = build_chunk_set(_chunks(_document()), SETTINGS, COUNTER, file_sha256=SHA)
    assert len(chunk_set.children) == 7 and all(child.parent_id for child in chunk_set.children)
    # преамбула — свой parent (без раздела), затем по одному parent на раздел
    assert [parent.section_key for parent in chunk_set.parents] == [
        "",
        "Раздел 1. Раздел номер 1",
        "Раздел 2. Раздел номер 2",
    ]
    clause_22 = next(child for child in chunk_set.children if child.clause == "2.2")
    parent = chunk_set.parent_of(clause_22)
    assert parent is not None and parent.heading == "2. Раздел номер 2"
    assert parent.breadcrumbs == (ROOT[0], "Раздел 2. Раздел номер 2") and "п. 2.2" not in parent.text
    assert parent.text.startswith(f"{ROOT[0]}{SETTINGS.breadcrumb_separator}Раздел 2. Раздел номер 2\n")
    assert all(f"2.{i}. " in parent.body for i in (1, 2, 3)) and clause_22.body in parent.body
    assert parent.child_ids == tuple(
        child.chunk_id for child in chunk_set.children if child.section_key == parent.section_key
    )
    assert (parent.part, parent.parts) == (1, 1) and parent.page_no is None
    assert parent.tokens == len(parent.text.split())


def test_long_section_is_split_into_parent_windows() -> None:
    chunks = _chunks(_document(clauses_per_section=12, words=400))
    settings = SETTINGS.model_copy(update={"parent_max_tokens": 1000})
    chunk_set = build_chunk_set(chunks, settings, COUNTER, file_sha256=SHA)
    section_1 = [parent for parent in chunk_set.parents if parent.section_key == "Раздел 1. Раздел номер 1"]
    assert len(section_1) == 6 and all(parent.tokens <= 1000 for parent in section_1)
    assert [parent.part for parent in section_1] == [1, 2, 3, 4, 5, 6]
    assert all(parent.parts == 6 for parent in section_1)
    # каждый child ровно в одном окне, порядок сохранён
    covered = [child_id for parent in chunk_set.parents for child_id in parent.child_ids]
    assert covered == [child.chunk_id for child in chunk_set.children]
    assert len(set(covered)) == len(covered)


def test_document_level_parent_and_fixed_strategy() -> None:
    chunk_set = build_chunk_set(_chunks(_document()), SETTINGS, COUNTER, file_sha256=SHA, level="document")
    assert len(chunk_set.parents) == 1
    parent = chunk_set.parents[0]
    assert parent.breadcrumbs == ROOT and parent.section_key == "" and parent.heading is None
    assert len(parent.child_ids) == 7 and "ПРИКАЗЫВАЮ" in parent.body and "2.3. " in parent.body

    letter = DoclingDocument(name="письмо")
    letter.add_heading("Ответ на запрос", level=1)
    for i in range(8):
        letter.add_text(label=DocItemLabel.TEXT, text=" ".join(f"Абзац {i} слово {j}." for j in range(60)))
    fixed = _chunks(letter)
    assert all(chunk.strategy == "fixed" for chunk in fixed) and len(fixed) >= 2
    fixed_set = build_chunk_set(fixed, SETTINGS, COUNTER, file_sha256=SHA)
    assert all(parent.tokens <= SETTINGS.parent_max_tokens for parent in fixed_set.parents)
    assert all(parent.breadcrumbs == (ROOT[0], "Ответ на запрос") for parent in fixed_set.parents)
    assert {child.parent_id for child in fixed_set.children} == {
        parent.chunk_id for parent in fixed_set.parents
    }


def test_parent_ids_are_deterministic_and_empty_input_is_fine() -> None:
    first = build_chunk_set(_chunks(_document()), SETTINGS, COUNTER, file_sha256=SHA)
    second = build_chunk_set(_chunks(_document()), SETTINGS, COUNTER, file_sha256=SHA)
    other = build_chunk_set(_chunks(_document(), sha="d" * 64), SETTINGS, COUNTER, file_sha256="d" * 64)
    assert [parent.chunk_id for parent in first.parents] == [parent.chunk_id for parent in second.parents]
    assert {parent.chunk_id for parent in first.parents}.isdisjoint(
        parent.chunk_id for parent in other.parents
    )
    assert {child.chunk_id for child in first.children}.isdisjoint(
        parent.chunk_id for parent in first.parents
    )
    empty = build_chunk_set([], SETTINGS, COUNTER, file_sha256=SHA)
    assert empty.children == [] and empty.parents == []
