"""Индекс Qdrant (4.6) на встроенном клиенте: коллекции, upsert, идемпотентность, удаление файла, parent."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from docling_core.types.doc.document import DoclingDocument
from docling_core.types.doc.labels import DocItemLabel
from qdrant_client import QdrantClient, models

from common.config import DEFAULT_CONFIG_PATH, load_app_config
from ingest.chunking import StructuralChunker
from ingest.embeddings import FakeEmbedder
from ingest.index import ChunkIndex
from ingest.metadata import ChunkPayload, DocumentMetadata, FileMetadata, child_payload, parent_payload
from ingest.parents import build_chunk_set
from ingest.tokens import WordTokenCounter

CONFIG = load_app_config(DEFAULT_CONFIG_PATH)
QDRANT = CONFIG.qdrant
CHUNKING = CONFIG.ingest.chunking


def _document_meta(doc_id: str, status: str = "active") -> DocumentMetadata:
    return DocumentMetadata(
        doc_id=doc_id,
        tessa_card_id=doc_id,
        card_type_name="OrderMKC",
        card_type_caption="Приказ",
        doc_kind="Приказ",
        doc_number="144",
        doc_date=None,
        doc_date_ts=None,
        doc_status=status,
        doc_status_name=None,
        state_id=6,
        state_name=None,
        approval_state=None,
        approval_state_name=None,
        department="Отдел охраны труда",
        department_id=None,
        author=None,
        relations=[],
        subject=None,
        comment=None,
        signed_by=None,
        direction_activity=[],
        approvers=[],
        responsible=[],
        validity_period=None,
        card_version=None,
        card_modified=None,
    )


def _file_meta(row_id: str, sha: str) -> FileMetadata:
    return FileMetadata(
        file_sha256=sha,
        file_name="приказ.docx",
        file_row_id=row_id,
        file_category="Документ",
        file_role="main",
        parse_route="native",
    )


def _payloads(
    doc_id: str, row_id: str, sha: str, texts: list[str], status: str = "active"
) -> tuple[list[ChunkPayload], list[ChunkPayload]]:
    doc = DoclingDocument(name="приказ")
    doc.add_heading("Тема", level=1)
    doc.add_heading("1. Раздел", level=2)
    for index, text in enumerate(texts, start=1):
        doc.add_text(label=DocItemLabel.TEXT, text=f"1.{index}. {text}")
    counter = WordTokenCounter()
    chunks = StructuralChunker(CHUNKING, counter).chunk(doc, ("Приказ №144",), file_sha256=sha)
    chunk_set = build_chunk_set(chunks, CHUNKING, counter, file_sha256=sha)
    meta, file = _document_meta(doc_id, status), _file_meta(row_id, sha)
    children = [
        child_payload(meta, file, chunk, CHUNKING.breadcrumb_separator) for chunk in chunk_set.children
    ]
    parents = [
        parent_payload(meta, file, parent, CHUNKING.breadcrumb_separator) for parent in chunk_set.parents
    ]
    return children, parents


def _index() -> tuple[ChunkIndex, FakeEmbedder]:
    embedder = FakeEmbedder(dense_dim=8)
    index = ChunkIndex(QdrantClient(":memory:"), QDRANT, embedder.dense_dim)
    index.ensure_collections()
    index.ensure_collections()  # повторный вызов ничего не ломает
    return index, embedder


def _write(
    index: ChunkIndex, embedder: FakeEmbedder, children: list[ChunkPayload], parents: list[ChunkPayload]
) -> int:
    index.delete_file(children[0].file_row_id)
    return index.upsert(children, embedder.encode([child.text for child in children]), parents)


def test_collections_have_named_vectors_and_payload_indexes() -> None:
    index, _ = _index()
    info: Any = index.client.get_collection(QDRANT.collection)
    assert (
        QDRANT.dense_vector in info.config.params.vectors
        and info.config.params.vectors[QDRANT.dense_vector].size == 8
    )
    assert QDRANT.sparse_vector in info.config.params.sparse_vectors
    # индексы payload встроенный клиент не показывает — их проверяет интеграционный тест на живом Qdrant
    assert index.client.collection_exists(QDRANT.parents_collection)


def test_upsert_is_idempotent_and_delete_removes_only_that_file() -> None:
    index, embedder = _index()
    doc_a, doc_b = str(uuid4()), str(uuid4())
    row_a, row_b = str(uuid4()), str(uuid4())
    children_a, parents_a = _payloads(
        doc_a, row_a, "a" * 64, ["Первый пункт приказа.", "Второй пункт приказа."]
    )
    children_b, parents_b = _payloads(
        doc_b, row_b, "b" * 64, ["Пункт другого документа."], status="cancelled"
    )
    written = _write(index, embedder, children_a, parents_a) + _write(index, embedder, children_b, parents_b)
    assert written == len(children_a) + len(parents_a) + len(children_b) + len(parents_b)
    before = (index.count(), index.count(parents=True))

    # AC-3.3: повторный прогон того же файла — те же точки, число не растёт
    _write(index, embedder, children_a, parents_a)
    assert (index.count(), index.count(parents=True)) == before
    # файл перечанкован короче — устаревшие точки не остаются
    shorter, shorter_parents = _payloads(doc_a, row_a, "a" * 64, ["Единственный пункт."])
    _write(index, embedder, shorter, shorter_parents)
    assert index.count_file(row_a) == len(shorter) == 1 and index.count_file(row_a, parents=True) == 1
    assert index.count_file(row_b) == len(children_b)
    assert index.file_row_ids() == {row_a, row_b}

    index.delete_file(row_b)
    assert index.count_file(row_b) == 0 and index.count_file(row_b, parents=True) == 0
    assert index.count() == 1 and index.file_row_ids() == {row_a}


def test_hybrid_query_with_status_filter_returns_child_and_its_parent() -> None:
    index, embedder = _index()
    doc_a, doc_b = str(uuid4()), str(uuid4())
    children_a, parents_a = _payloads(
        doc_a, str(uuid4()), "a" * 64, ["Отчёт сдаётся до пятого числа.", "Контроль за собой."]
    )
    children_b, parents_b = _payloads(
        doc_b, str(uuid4()), "b" * 64, ["Отчёт сдаётся до десятого числа."], status="cancelled"
    )
    _write(index, embedder, children_a, parents_a)
    _write(index, embedder, children_b, parents_b)

    query = embedder.encode(["отчёт сдаётся до пятого числа"])[0]
    only_active = models.Filter(
        must=[models.FieldCondition(key="doc_status", match=models.MatchValue(value="active"))]
    )
    found = index.client.query_points(
        QDRANT.collection,
        prefetch=[
            models.Prefetch(query=query.dense, using=QDRANT.dense_vector, limit=10, filter=only_active),
            models.Prefetch(
                query=models.SparseVector(indices=list(query.sparse), values=list(query.sparse.values())),
                using=QDRANT.sparse_vector,
                limit=10,
                filter=only_active,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=3,
        with_payload=True,
    ).points
    assert found and all(point.payload and point.payload["doc_status"] == "active" for point in found)
    top = ChunkPayload.model_validate(found[0].payload)
    assert top.doc_id == doc_a and "пятого" in top.body
    parent = index.get_parent(top.parent_id or "")
    assert parent is not None and parent.chunk_level == "parent" and top.chunk_id in parent.child_ids
    assert "Контроль за собой" in parent.body and index.get_parent(str(uuid4())) is None
