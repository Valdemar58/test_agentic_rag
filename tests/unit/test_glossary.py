"""Глоссарий (8.1, FR-5): заголовки разделов, кандидаты «термин — определение», подтверждение, сборка."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import cast

import pytest
from docling_core.types.doc.document import DoclingDocument
from docling_core.types.doc.labels import DocItemLabel
from qdrant_client import QdrantClient

from common.config import DEFAULT_CONFIG_PATH, DocStatus, load_app_config
from ingest.chunking import StructuralChunker
from ingest.embeddings import FakeEmbedder
from ingest.glossary import (
    GlossaryBuilder,
    GlossaryRecord,
    Section,
    SectionVerdict,
    is_glossary_heading,
    normalize_term,
    parse_candidates,
    parse_verdict,
    split_entry,
)
from ingest.glossary_index import GlossaryIndex
from ingest.index import ChunkIndex
from ingest.metadata import ChunkPayload, child_payload, parent_payload
from ingest.parents import build_chunk_set
from ingest.tokens import WordTokenCounter
from mcp_server.glossary import EMPTY_NOTE, UNAVAILABLE_NOTE, QdrantGlossary, matches, order
from tests.unit.index_data import document_meta, file_meta, new_ids

CONFIG = load_app_config(DEFAULT_CONFIG_PATH)
GLOSSARY = CONFIG.glossary
CHUNKING = CONFIG.ingest.chunking
HEADINGS = [normalize_term(item) for item in GLOSSARY.headings]
TERMS_HEADING = "2. Термины и определения"


def _document(sections: dict[str, list[str]]) -> DoclingDocument:
    document = DoclingDocument(name="положение")
    document.add_heading("Положение об охране труда", level=1)
    for heading, paragraphs in sections.items():
        document.add_heading(heading, level=2)
        for text in paragraphs:
            document.add_text(label=DocItemLabel.TEXT, text=text)
    return document


def _payloads(sections: dict[str, list[str]]) -> tuple[list[ChunkPayload], list[ChunkPayload]]:
    doc_id, row_id, sha = new_ids()
    counter = WordTokenCounter()
    chunks = StructuralChunker(CHUNKING, counter).chunk(
        _document(sections), ("Положение №5",), file_sha256=sha
    )
    chunk_set = build_chunk_set(chunks, CHUNKING, counter, file_sha256=sha)
    meta = document_meta(doc_id, doc_kind="Положение", number="5")
    file = file_meta(row_id, sha, name="положение.docx")
    separator = CHUNKING.breadcrumb_separator
    return (
        [child_payload(meta, file, chunk, separator) for chunk in chunk_set.children],
        [parent_payload(meta, file, parent, separator) for parent in chunk_set.parents],
    )


def _chunk(body: str, *, heading: str = TERMS_HEADING) -> ChunkPayload:
    children, _ = _payloads({heading: [line for line in body.splitlines() if line.strip()]})
    payload = children[0]
    return payload.model_copy(update={"body": body})


@dataclass
class FakeConfirmer:
    """Подтверждение по сценарию: по вердикту на раздел в порядке обхода."""

    verdicts: list[SectionVerdict]
    seen: list[Section] = field(default_factory=list)

    async def confirm(self, section: Section) -> SectionVerdict:
        self.seen.append(section)
        return self.verdicts.pop(0) if self.verdicts else SectionVerdict(glossary=True)


class _Stack:
    """Индекс документов и коллекция глоссария на встроенном Qdrant с фейковым эмбеддером."""

    def __init__(self) -> None:
        self.embedder = FakeEmbedder(dense_dim=8)
        client = QdrantClient(":memory:")
        self.chunks = ChunkIndex(client, CONFIG.qdrant, self.embedder.dense_dim)
        self.chunks.ensure_collections()
        self.glossary = GlossaryIndex(client, CONFIG.qdrant, self.embedder.dense_dim)

    def add(self, sections: dict[str, list[str]]) -> None:
        children, parents = _payloads(sections)
        self.chunks.upsert(children, self.embedder.encode([child.text for child in children]), parents)

    def builder(self, confirmer: FakeConfirmer) -> GlossaryBuilder:
        return GlossaryBuilder(
            GLOSSARY,
            source=self.chunks,
            store=self.glossary,
            embedder=self.embedder,
            confirmer=confirmer,
        )


def test_headings_and_entry_split() -> None:
    assert is_glossary_heading("2. Термины и определения", HEADINGS)
    assert is_glossary_heading("ТЕРМИНЫ, ОПРЕДЕЛЕНИЯ И СОКРАЩЕНИЯ", HEADINGS)
    assert is_glossary_heading("Раздел 3. Используемые сокращения", HEADINGS)
    assert not is_glossary_heading("4. Режим рабочего времени", HEADINGS)
    assert not is_glossary_heading("", HEADINGS)

    assert split_entry("ПВТР — правила внутреннего трудового распорядка;") == (
        "ПВТР",
        "правила внутреннего трудового распорядка;",
    )
    assert split_entry("- СИЗ: средства индивидуальной защиты") == ("СИЗ", "средства индивидуальной защиты")
    assert split_entry("| ЛНА | локальный нормативный акт |") == ("ЛНА", "локальный нормативный акт")
    assert split_entry("Работник обязан соблюдать требования охраны труда") is None
    assert split_entry("|---|---|") is None
    # реальный корпус: обычный дефис — и разделитель, и часть слова
    assert split_entry("ТН - транспортная накладная") == ("ТН", "транспортная накладная")
    assert split_entry("расчётный листок - документ о начисленной оплате") == (
        "расчётный листок",
        "документ о начисленной оплате",
    )
    assert split_entry("информационно - справочные документы") is None


def test_candidates_keep_terms_and_drop_text() -> None:
    chunk = _chunk(
        "\n".join(
            [
                "2.1. В настоящем Положении используются следующие термины и определения:",
                "ПВТР — правила внутреннего трудового распорядка;",
                "СИЗ – средства индивидуальной защиты работника;",
                "1) Работник: физическое лицо, вступившее в трудовые отношения с Работодателем.",
                "Должностное лицо, ответственное за охрану труда в подразделении, — назначается приказом",
                "АС — ас",
                "приказ (распоряжение — локальный нормативный акт Общества .",
                "Настоящее Положение вводится в действие приказом",
            ]
        )
    )
    candidates = parse_candidates(chunk, GLOSSARY)
    assert [item.term for item in candidates] == ["ПВТР", "СИЗ", "Работник", "приказ"]
    assert candidates[0].definition == "правила внутреннего трудового распорядка"
    assert candidates[2].definition.startswith("физическое лицо")
    assert candidates[3].definition == "локальный нормативный акт Общества."
    assert candidates[0].chunk.chunk_id == chunk.chunk_id


def test_table_rows_without_header_row() -> None:
    chunk = _chunk(
        "\n".join(
            [
                "| Термин | Определение |",
                "|---|---|",
                "| ЛНА | локальный нормативный акт организации |",
                "| ДОУ | документационное обеспечение управления |",
            ]
        )
    )
    assert [item.term for item in parse_candidates(chunk, GLOSSARY)] == ["ЛНА", "ДОУ"]


def test_verdict_parsing_is_robust() -> None:
    verdict = parse_verdict('Вот ответ: {"glossary": true, "reject": [2, "3"], "reason": "строка 2 — пункт"}')
    assert verdict.parsed and verdict.glossary and verdict.reject == {2, 3}
    assert parse_verdict('{"glossary": false, "reason": "перечень должностей"}').glossary is False
    assert parse_verdict('{"reject": []} и ещё текст после объекта').glossary is True
    assert parse_verdict("модель ничего не вернула").parsed is False
    assert parse_verdict('{"reject": "две"}').parsed is False


@pytest.mark.asyncio
async def test_builder_writes_only_confirmed_entries() -> None:
    stack = _Stack()
    stack.add(
        {
            TERMS_HEADING: [
                "ПВТР — правила внутреннего трудового распорядка;",
                "СИЗ — средства индивидуальной защиты работника;",
                "Работодатель — организация, вступившая в трудовые отношения с работником.",
            ],
            "3. Режим рабочего времени": ["Перерыв для отдыха и питания — 45 минут в течение смены."],
        }
    )
    confirmer = FakeConfirmer([SectionVerdict(glossary=True, reject={2})])
    report = await stack.builder(confirmer).build()

    assert len(confirmer.seen) == 1, "подтверждается только раздел глоссария"
    assert confirmer.seen[0].heading == TERMS_HEADING
    assert report.sections == 1 and report.confirmed == 1
    assert report.candidates == 3 and report.rejected_entries == 1 and report.records == 2
    records = {record.term: record for record in stack.glossary.records()}
    assert set(records) == {"ПВТР", "Работодатель"}
    assert records["ПВТР"].definition == "правила внутреннего трудового распорядка"
    assert records["ПВТР"].term_key == "пвтр"
    assert records["ПВТР"].section_id and records["ПВТР"].doc_label.startswith("Положение №5")


@pytest.mark.asyncio
async def test_section_without_confirmation_is_not_indexed() -> None:
    stack = _Stack()
    stack.add({TERMS_HEADING: ["ПВТР — правила внутреннего трудового распорядка;"]})

    rejected = await stack.builder(
        FakeConfirmer([SectionVerdict(glossary=False, reason="не глоссарий")])
    ).build()
    assert rejected.rejected_sections == 1 and rejected.records == 0
    assert stack.glossary.count() == 0

    failed = await stack.builder(FakeConfirmer([SectionVerdict(glossary=False, parsed=False)])).build()
    assert failed.unconfirmed == 1 and failed.records == 0


@pytest.mark.asyncio
async def test_rebuild_is_idempotent_and_removes_stale_records() -> None:
    stack = _Stack()
    stack.add(
        {
            TERMS_HEADING: [
                "ПВТР — правила внутреннего трудового распорядка;",
                "СИЗ — средства индивидуальной защиты работника;",
            ]
        }
    )
    first = await stack.builder(FakeConfirmer([])).build()
    ids = {record.record_id for record in stack.glossary.records()}

    again = await stack.builder(FakeConfirmer([])).build()
    assert again.records == first.records == 2
    assert {record.record_id for record in stack.glossary.records()} == ids, "id точек детерминированы"
    assert again.removed == 0

    without_siz = await stack.builder(FakeConfirmer([SectionVerdict(glossary=True, reject={2})])).build()
    assert without_siz.records == 1 and without_siz.removed == 1
    assert [record.term for record in stack.glossary.records()] == ["ПВТР"]


@pytest.mark.asyncio
async def test_lookup_by_term_and_by_vector() -> None:
    stack = _Stack()
    stack.add({TERMS_HEADING: ["ПВТР — правила внутреннего трудового распорядка;"]})
    await stack.builder(FakeConfirmer([])).build()

    assert [record.term for record in stack.glossary.by_term("пвтр", 5)] == ["ПВТР"]
    assert stack.glossary.by_term("сиз", 5) == []
    found = stack.glossary.search(stack.embedder.encode(["правила внутреннего трудового распорядка"])[0], 5)
    assert [record.term for record in found] == ["ПВТР"]


@pytest.mark.asyncio
async def test_glossary_lookup_tool_answers_by_term_and_explains_miss() -> None:
    stack = _Stack()
    stack.add({TERMS_HEADING: ["ПВТР — правила внутреннего трудового распорядка;"]})
    await stack.builder(FakeConfirmer([])).build()
    lookup = QdrantGlossary(stack.glossary, stack.embedder, GLOSSARY)

    found = lookup.lookup("ПВТР.")
    assert [entry.term for entry in found.entries] == ["ПВТР"] and found.note is None
    assert found.entries[0].definition == "правила внутреннего трудового распорядка"
    assert found.entries[0].doc_status == "active" and found.entries[0].section_id

    # векторный поиск найдёт запись по смыслу, но термин не тот — инструмент отвечает «нет»
    missing = lookup.lookup("СИЗ")
    assert missing.entries == [] and missing.note is not None and "СИЗ" in missing.note
    assert QdrantGlossary(_Stack().glossary, stack.embedder, GLOSSARY).lookup("ПВТР").note == EMPTY_NOTE


def test_lookup_orders_active_first_and_drops_repeats() -> None:
    def record(term: str, definition: str, status: DocStatus, label: str) -> GlossaryRecord:
        return GlossaryRecord(
            record_id=f"{term}-{label}",
            term=term,
            term_key=normalize_term(term),
            definition=definition,
            doc_id=label,
            doc_label=label,
            doc_kind="Приказ",
            doc_status=status,
            section_id="s1",
            chunk_id="c1",
            file_row_id="f1",
            breadcrumbs="",
        )

    ordered = order(
        [
            record("ПВТР", "правила внутреннего трудового распорядка", "cancelled", "Приказ №1"),
            record("ПВТР", "Правила внутреннего трудового распорядка.", "active", "Приказ №2"),
            record("ПВТР", "правила распорядка организации", "active", "Приказ №3"),
        ]
    )
    assert [item.doc_label for item in ordered] == ["Приказ №2", "Приказ №3"]

    assert matches("пвтр", "пвтр", GLOSSARY.match_ratio)
    assert matches("правила внутреннего трудового распорядка", "правила внутреннего распорядка", 0.75)
    assert not matches("сиз", "пвтр", GLOSSARY.match_ratio)


def test_lookup_survives_unavailable_collection() -> None:
    class Broken:
        def exists(self) -> bool:
            raise RuntimeError("Qdrant недоступен")

    result = QdrantGlossary(cast(GlossaryIndex, Broken()), FakeEmbedder(dense_dim=8), GLOSSARY).lookup("ПВТР")
    assert result.entries == [] and result.note == UNAVAILABLE_NOTE
