"""Запись чанков в Qdrant (FR-3, §3 ТЗ): named vectors `dense` + `sparse`, parent-коллекция, идемпотентность.

- `documents`: child-чанки, у каждой точки dense-вектор bge-m3 и sparse-вектор (lexical weights),
  payload = `ChunkPayload`; индексы payload по полям фильтров (`doc_id`, `doc_kind`, `doc_status`,
  `doc_date_ts`, `department`, `chunk_kind`, `file_sha256`, `file_row_id`, `parent_id`).
- `documents_parents`: parent-разделы без векторов, только payload; агент получает parent по
  `parent_id` из найденного child.
- Идемпотентная запись (AC-3.3): id точек детерминированы (uuid5 от sha256 файла), перед upsert
  все точки файла (`file_row_id`) удаляются из обеих коллекций — повторный прогон не плодит
  дубликатов и не оставляет устаревших чанков.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any
from uuid import UUID

from qdrant_client import QdrantClient, models

from common.config import QdrantSettings
from ingest.embeddings import Embedding
from ingest.metadata import ChunkPayload

logger = logging.getLogger(__name__)

FILE_KEY = "file_row_id"
PAYLOAD_INDEXES: dict[str, models.PayloadSchemaType] = {
    "doc_id": models.PayloadSchemaType.KEYWORD,
    "doc_kind": models.PayloadSchemaType.KEYWORD,
    "doc_status": models.PayloadSchemaType.KEYWORD,
    "doc_date_ts": models.PayloadSchemaType.INTEGER,
    "department": models.PayloadSchemaType.KEYWORD,
    "chunk_kind": models.PayloadSchemaType.KEYWORD,
    "file_sha256": models.PayloadSchemaType.KEYWORD,
    FILE_KEY: models.PayloadSchemaType.KEYWORD,
    "parent_id": models.PayloadSchemaType.KEYWORD,
}


def _file_filter(file_row_id: UUID | str) -> models.Filter:
    return models.Filter(
        must=[models.FieldCondition(key=FILE_KEY, match=models.MatchValue(value=str(file_row_id)))]
    )


class ChunkIndex:
    def __init__(self, client: QdrantClient, settings: QdrantSettings, dense_dim: int) -> None:
        self._client = client
        self._settings = settings
        self._dense_dim = dense_dim

    @property
    def client(self) -> QdrantClient:
        return self._client

    def ensure_collections(self) -> None:
        """Создаёт коллекции и индексы payload, если их нет; существующие не трогает."""
        settings = self._settings
        if not self._client.collection_exists(settings.collection):
            self._client.create_collection(
                settings.collection,
                vectors_config={
                    settings.dense_vector: models.VectorParams(
                        size=self._dense_dim, distance=models.Distance[settings.distance.upper()]
                    )
                },
                sparse_vectors_config={settings.sparse_vector: models.SparseVectorParams()},
            )
            logger.info("Создана коллекция %s (dense %d, sparse)", settings.collection, self._dense_dim)
        if not self._client.collection_exists(settings.parents_collection):
            self._client.create_collection(settings.parents_collection, vectors_config={})
            logger.info("Создана коллекция %s (parent-разделы без векторов)", settings.parents_collection)
        for collection in (settings.collection, settings.parents_collection):
            existing = self._client.get_collection(collection).payload_schema
            for field, schema in PAYLOAD_INDEXES.items():
                if field not in existing:
                    self._client.create_payload_index(collection, field_name=field, field_schema=schema)

    def delete_file(self, file_row_id: UUID | str) -> None:
        for collection in (self._settings.collection, self._settings.parents_collection):
            self._client.delete(
                collection, points_selector=models.FilterSelector(filter=_file_filter(file_row_id))
            )

    def upsert(
        self,
        children: Sequence[ChunkPayload],
        embeddings: Sequence[Embedding],
        parents: Sequence[ChunkPayload],
    ) -> int:
        """Пишет child-точки с векторами и parent-точки; возвращает число записанных точек."""
        if len(children) != len(embeddings):
            raise ValueError(f"чанков {len(children)}, эмбеддингов {len(embeddings)}: должны совпадать")
        settings = self._settings
        child_points = [
            models.PointStruct(
                id=payload.chunk_id,
                vector={
                    settings.dense_vector: embedding.dense,
                    settings.sparse_vector: models.SparseVector(
                        indices=list(embedding.sparse.keys()), values=list(embedding.sparse.values())
                    ),
                },
                payload=payload.model_dump(),
            )
            for payload, embedding in zip(children, embeddings, strict=True)
        ]
        parent_points = [
            models.PointStruct(id=payload.chunk_id, vector={}, payload=payload.model_dump())
            for payload in parents
        ]
        batch = settings.upsert_batch_size
        for start in range(0, len(child_points), batch):
            self._client.upsert(settings.collection, points=child_points[start : start + batch], wait=True)
        for start in range(0, len(parent_points), batch):
            self._client.upsert(
                settings.parents_collection, points=parent_points[start : start + batch], wait=True
            )
        return len(child_points) + len(parent_points)

    def set_payload(self, file_row_id: UUID | str, fields: dict[str, Any]) -> None:
        """Обновляет поля payload у всех точек файла в обеих коллекциях (карточка изменилась, файл — нет)."""
        for collection in (self._settings.collection, self._settings.parents_collection):
            self._client.set_payload(
                collection,
                payload=fields,
                points=models.FilterSelector(filter=_file_filter(file_row_id)),
                wait=True,
            )

    def count(self, *, parents: bool = False) -> int:
        collection = self._settings.parents_collection if parents else self._settings.collection
        return int(self._client.count(collection, exact=True).count)

    def count_file(self, file_row_id: UUID | str, *, parents: bool = False) -> int:
        collection = self._settings.parents_collection if parents else self._settings.collection
        return int(self._client.count(collection, count_filter=_file_filter(file_row_id), exact=True).count)

    def file_row_ids(self) -> set[str]:
        """Все file_row_id с точками в коллекции child-чанков — для уборки исчезнувших файлов."""
        found: set[str] = set()
        offset: models.ExtendedPointId | None = None
        while True:
            points, offset = self._client.scroll(
                self._settings.collection,
                limit=self._settings.upsert_batch_size * 16,
                offset=offset,
                with_payload=[FILE_KEY],
                with_vectors=False,
            )
            found.update(str(point.payload[FILE_KEY]) for point in points if point.payload)
            if offset is None:
                return found

    def get_parent(self, parent_id: str) -> ChunkPayload | None:
        records = self._client.retrieve(self._settings.parents_collection, ids=[parent_id], with_payload=True)
        if not records or records[0].payload is None:
            return None
        return ChunkPayload.model_validate(records[0].payload)
