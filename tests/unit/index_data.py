"""Тестовые payload'ы индекса: документ из нескольких пунктов, как их пишет инжест (без весов моделей)."""

from __future__ import annotations

import datetime as dt
from typing import Any
from uuid import uuid4

from docling_core.types.doc.document import DoclingDocument
from docling_core.types.doc.labels import DocItemLabel

from common.config import DEFAULT_CONFIG_PATH, DocStatus, load_app_config
from ingest.chunking import StructuralChunker
from ingest.metadata import ChunkPayload, DocumentMetadata, FileMetadata, child_payload, parent_payload
from ingest.parents import build_chunk_set
from ingest.tokens import WordTokenCounter

CONFIG = load_app_config(DEFAULT_CONFIG_PATH)
CHUNKING = CONFIG.ingest.chunking


def document_meta(
    doc_id: str,
    status: DocStatus = "active",
    *,
    doc_kind: str = "Приказ",
    number: str = "144",
    doc_date: dt.date | None = None,
    department: str | None = "Отдел охраны труда",
    subject: str | None = None,
) -> DocumentMetadata:
    return DocumentMetadata(
        doc_id=doc_id,
        tessa_card_id=doc_id,
        card_type_name="OrderMKC",
        card_type_caption=doc_kind,
        doc_kind=doc_kind,
        doc_number=number,
        doc_date=doc_date,
        doc_date_ts=int(dt.datetime(doc_date.year, doc_date.month, doc_date.day, tzinfo=dt.UTC).timestamp())
        if doc_date
        else None,
        doc_status=status,
        doc_status_name=None,
        state_id=6,
        state_name=None,
        approval_state=None,
        approval_state_name=None,
        department=department,
        department_id=None,
        author=None,
        relations=[],
        subject=subject,
        comment=None,
        signed_by=None,
        direction_activity=[],
        approvers=[],
        responsible=[],
        validity_period=None,
        card_version=None,
        card_modified=None,
    )


def file_meta(row_id: str, sha: str, *, name: str = "документ.docx", role: str = "main") -> FileMetadata:
    return FileMetadata(
        file_sha256=sha,
        file_name=name,
        file_row_id=row_id,
        file_category="Документ",
        file_role=role,
        parse_route="native",
    )


def make_payloads(
    doc_id: str,
    row_id: str,
    sha: str,
    texts: list[str],
    status: DocStatus = "active",
    *,
    doc_kind: str = "Приказ",
    number: str = "144",
    doc_date: dt.date | None = None,
    department: str | None = "Отдел охраны труда",
    subject: str | None = None,
    file_role: str = "main",
    file_name: str = "документ.docx",
    sections: int = 1,
) -> tuple[list[ChunkPayload], list[ChunkPayload]]:
    """Документ «Тема / N. Раздел / N.M. текст» → child- и parent-payload'ы; тексты делятся по разделам."""
    doc = DoclingDocument(name="документ")
    doc.add_heading("Тема", level=1)
    per_section = max(1, -(-len(texts) // sections))
    for section in range(sections):
        doc.add_heading(f"{section + 1}. Раздел {section + 1}", level=2)
        for index, text in enumerate(texts[section * per_section : (section + 1) * per_section], start=1):
            doc.add_text(label=DocItemLabel.TEXT, text=f"{section + 1}.{index}. {text}")
    counter = WordTokenCounter()
    label = f"{doc_kind} №{number}"
    # корень крошек как в инжесте: основной файл — документ; приложение — документ и имя файла
    root = (label,) if file_role == "main" else (label, f"Приложение «{file_name}»")
    chunks = StructuralChunker(CHUNKING, counter).chunk(doc, root, file_sha256=sha)
    chunk_set = build_chunk_set(chunks, CHUNKING, counter, file_sha256=sha)
    meta = document_meta(
        doc_id,
        status,
        doc_kind=doc_kind,
        number=number,
        doc_date=doc_date,
        department=department,
        subject=subject,
    )
    file = file_meta(row_id, sha, name=file_name, role=file_role)
    separator = CHUNKING.breadcrumb_separator
    return (
        [child_payload(meta, file, chunk, separator) for chunk in chunk_set.children],
        [parent_payload(meta, file, parent, separator) for parent in chunk_set.parents],
    )


def new_ids() -> tuple[str, str, str]:
    """doc_id, file_row_id, sha256 для нового тестового документа."""
    return str(uuid4()), str(uuid4()), uuid4().hex * 2


class InMemoryCorpus:
    """Индекс на встроенном Qdrant с фейковыми эмбеддером и reranker'ом — для тестов поиска и MCP."""

    def __init__(self) -> None:
        from qdrant_client import QdrantClient

        from ingest.embeddings import FakeEmbedder
        from ingest.index import ChunkIndex

        self.embedder = FakeEmbedder(dense_dim=8)
        self.index = ChunkIndex(QdrantClient(":memory:"), CONFIG.qdrant, self.embedder.dense_dim)
        self.index.ensure_collections()
        self.docs: dict[str, str] = {}

    def add(
        self,
        name: str,
        texts: list[str],
        status: DocStatus = "active",
        *,
        doc_kind: str = "Приказ",
        number: str = "144",
        doc_date: dt.date | None = None,
        department: str | None = "Отдел охраны труда",
        subject: str | None = None,
        file_role: str = "main",
        file_name: str = "документ.docx",
        sections: int = 1,
        doc_id: str | None = None,
    ) -> str:
        """Добавляет файл документа; с `doc_id` — ещё один файл (приложение) уже добавленного документа."""
        new_doc_id, row_id, sha = new_ids()
        doc_id = doc_id or new_doc_id
        children, parents = make_payloads(
            doc_id,
            row_id,
            sha,
            texts,
            status,
            doc_kind=doc_kind,
            number=number,
            doc_date=doc_date,
            department=department,
            subject=subject,
            file_role=file_role,
            file_name=file_name,
            sections=sections,
        )
        self.index.upsert(children, self.embedder.encode([child.text for child in children]), parents)
        self.docs[name] = doc_id
        return doc_id

    def searcher(self, **overrides: object) -> Any:
        from mcp_server.reranker import FakeReranker
        from mcp_server.retrieval import HybridSearcher

        retrieval = CONFIG.retrieval.model_copy(update=overrides)
        return HybridSearcher(self.index.client, CONFIG.qdrant, retrieval, self.embedder, FakeReranker())


def standard_corpus() -> InMemoryCorpus:
    """Действующий приказ, отменённый приказ, договор другого подразделения, проект приказа."""
    corpus = InMemoryCorpus()
    corpus.add(
        "order_144",
        ["Отчёт по охране труда сдаётся до пятого числа.", "Контроль оставляю за собой."],
        doc_date=dt.date(2026, 1, 15),
        subject="Об охране труда",
    )
    corpus.add(
        "order_109_cancelled",
        ["Отчёт по охране труда сдаётся до десятого числа."],
        "cancelled",
        number="109",
        doc_date=dt.date(2025, 2, 10),
    )
    corpus.add(
        "contract_d1",
        ["Поставщик сдаёт отчёт о поставке ежемесячно.", "Срок действия договора до конца года."],
        doc_kind="Договорной документ",
        number="Д-1",
        doc_date=dt.date(2023, 11, 1),
        department="Отдел закупок",
    )
    corpus.add(
        "draft_160",
        ["Проект приказа об отчёте по охране труда."],
        "draft",
        number="",
        doc_date=dt.date(2026, 9, 1),
        department="Отдел закупок",
    )
    return corpus
