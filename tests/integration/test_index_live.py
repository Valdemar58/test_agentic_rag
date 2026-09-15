"""Индекс на живом Qdrant стенда (4.6): коллекции, индексы payload, гибридный запрос, идемпотентность.

Работает в отдельных тестовых коллекциях и удаляет их после себя; без стенда — скип.
"""

from __future__ import annotations

from collections.abc import Iterator
from uuid import uuid4

import httpx
import pytest
from docling_core.types.doc.document import DoclingDocument
from docling_core.types.doc.labels import DocItemLabel
from qdrant_client import QdrantClient, models

from common.config import DEFAULT_CONFIG_PATH, load_app_config
from common.settings import load_settings
from ingest.chunking import StructuralChunker
from ingest.embeddings import FakeEmbedder
from ingest.index import PAYLOAD_INDEXES, ChunkIndex
from ingest.metadata import ChunkPayload, DocumentMetadata, FileMetadata, child_payload, parent_payload
from ingest.parents import build_chunk_set
from ingest.tokens import WordTokenCounter

pytestmark = pytest.mark.integration

CONFIG = load_app_config(DEFAULT_CONFIG_PATH)
SUFFIX = uuid4().hex[:8]
QDRANT = CONFIG.qdrant.model_copy(
    update={"collection": f"test_documents_{SUFFIX}", "parents_collection": f"test_parents_{SUFFIX}"}
)


@pytest.fixture(scope="module")
def client() -> Iterator[QdrantClient]:
    url = load_settings().resolve_qdrant_url()
    try:
        httpx.get(f"{url}/readyz", timeout=3).raise_for_status()
    except (httpx.HTTPError, OSError) as exc:
        pytest.skip(f"Qdrant недоступен ({exc}); поднимите стенд")
    client = QdrantClient(url=url, timeout=int(CONFIG.qdrant.timeout_s))
    yield client
    for collection in (QDRANT.collection, QDRANT.parents_collection):
        if client.collection_exists(collection):
            client.delete_collection(collection)


def _payloads(doc_id: str, texts: list[str]) -> tuple[list[ChunkPayload], list[ChunkPayload]]:
    doc = DoclingDocument(name="приказ")
    doc.add_heading("Тема", level=1)
    doc.add_heading("1. Раздел", level=2)
    for index, text in enumerate(texts, start=1):
        doc.add_text(label=DocItemLabel.TEXT, text=f"1.{index}. {text}")
    counter = WordTokenCounter()
    sha = uuid4().hex * 2
    chunks = StructuralChunker(CONFIG.ingest.chunking, counter).chunk(doc, ("Приказ №1",), file_sha256=sha)
    chunk_set = build_chunk_set(chunks, CONFIG.ingest.chunking, counter, file_sha256=sha)
    meta = DocumentMetadata(
        doc_id=doc_id,
        tessa_card_id=doc_id,
        card_type_name="OrderMKC",
        card_type_caption="Приказ",
        doc_kind="Приказ",
        doc_number="1",
        doc_date=None,
        doc_date_ts=None,
        doc_status="active",
        doc_status_name=None,
        state_id=6,
        state_name=None,
        approval_state=None,
        approval_state_name=None,
        department=None,
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
    file = FileMetadata(
        file_sha256=sha,
        file_name="приказ.docx",
        file_row_id=str(uuid4()),
        file_category="Документ",
        file_role="main",
        parse_route="native",
    )
    separator = CONFIG.ingest.chunking.breadcrumb_separator
    return (
        [child_payload(meta, file, chunk, separator) for chunk in chunk_set.children],
        [parent_payload(meta, file, parent, separator) for parent in chunk_set.parents],
    )


def test_live_collections_indexes_hybrid_query_and_idempotency(client: QdrantClient) -> None:
    embedder = FakeEmbedder(dense_dim=8)
    index = ChunkIndex(client, QDRANT, embedder.dense_dim)
    index.ensure_collections()
    index.ensure_collections()
    schema = client.get_collection(QDRANT.collection).payload_schema
    assert set(PAYLOAD_INDEXES) <= set(schema)
    assert client.get_collection(QDRANT.parents_collection).payload_schema.keys() >= set(PAYLOAD_INDEXES)

    children, parents = _payloads(str(uuid4()), ["Отчёт сдаётся до пятого числа.", "Контроль за собой."])
    row_id = children[0].file_row_id
    embeddings = embedder.encode([child.text for child in children])
    for _ in range(2):
        index.delete_file(row_id)
        index.upsert(children, embeddings, parents)
    assert index.count_file(row_id) == len(children) and index.count_file(row_id, parents=True) == len(
        parents
    )
    assert index.file_row_ids() == {row_id}

    query = embedder.encode(["отчёт до пятого числа"])[0]
    active = models.Filter(
        must=[models.FieldCondition(key="doc_status", match=models.MatchValue(value="active"))]
    )
    found = client.query_points(
        QDRANT.collection,
        prefetch=[
            models.Prefetch(query=query.dense, using=QDRANT.dense_vector, limit=5, filter=active),
            models.Prefetch(
                query=models.SparseVector(indices=list(query.sparse), values=list(query.sparse.values())),
                using=QDRANT.sparse_vector,
                limit=5,
                filter=active,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=2,
        with_payload=True,
    ).points
    assert found and "пятого" in ChunkPayload.model_validate(found[0].payload).body
    parent = index.get_parent(str(found[0].payload["parent_id"]) if found[0].payload else "")
    assert parent is not None and parent.chunk_level == "parent"

    index.set_payload(row_id, {"doc_status": "cancelled"})
    changed, _ = client.scroll(QDRANT.collection, limit=10, with_payload=["doc_status"])
    assert {point.payload["doc_status"] for point in changed if point.payload} == {"cancelled"}
    index.delete_file(row_id)
    assert index.count() == 0 and index.count(parents=True) == 0
