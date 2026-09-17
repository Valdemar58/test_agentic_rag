"""Коллекция глоссария в Qdrant (FR-5): записи «термин — определение» с dense- и sparse-векторами.

Одна точка — одна запись; id детерминирован (uuid5 от чанка и ключа термина), поэтому повторная сборка
переписывает те же точки, а исчезнувшие удаляются по разнице идентификаторов — как идемпотентная запись
чанков в `ChunkIndex` (AC-3.3). Термин ищется сначала точным совпадением ключа (индекс payload
`term_key`), и только если его нет — векторами: dense + sparse с серверным RRF, как в `hybrid_search`.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from qdrant_client import QdrantClient, models

from common.config import QdrantSettings
from ingest.embeddings import Embedding
from ingest.glossary import GlossaryRecord

logger = logging.getLogger(__name__)

TERM_FIELD = "term_key"
PAYLOAD_INDEXES: dict[str, models.PayloadSchemaType] = {
    TERM_FIELD: models.PayloadSchemaType.KEYWORD,
    "doc_id": models.PayloadSchemaType.KEYWORD,
    "doc_status": models.PayloadSchemaType.KEYWORD,
    "file_row_id": models.PayloadSchemaType.KEYWORD,
}


class GlossaryIndex:
    def __init__(self, client: QdrantClient, settings: QdrantSettings, dense_dim: int) -> None:
        self._client = client
        self._settings = settings
        self._dense_dim = dense_dim

    @property
    def collection(self) -> str:
        return self._settings.glossary_collection

    def exists(self) -> bool:
        return bool(self._client.collection_exists(self.collection))

    def ensure_collection(self) -> None:
        settings = self._settings
        if not self.exists():
            self._client.create_collection(
                self.collection,
                vectors_config={
                    settings.dense_vector: models.VectorParams(
                        size=self._dense_dim, distance=models.Distance[settings.distance.upper()]
                    )
                },
                sparse_vectors_config={settings.sparse_vector: models.SparseVectorParams()},
            )
            logger.info("Создана коллекция глоссария %s (dense %d, sparse)", self.collection, self._dense_dim)
        existing = self._client.get_collection(self.collection).payload_schema
        for field, schema in PAYLOAD_INDEXES.items():
            if field not in existing:
                self._client.create_payload_index(self.collection, field_name=field, field_schema=schema)

    def upsert(self, records: Sequence[GlossaryRecord], embeddings: Sequence[Embedding]) -> int:
        if len(records) != len(embeddings):
            raise ValueError(f"записей {len(records)}, эмбеддингов {len(embeddings)}: должны совпадать")
        settings = self._settings
        points = [
            models.PointStruct(
                id=record.record_id,
                vector={
                    settings.dense_vector: embedding.dense,
                    settings.sparse_vector: models.SparseVector(
                        indices=list(embedding.sparse.keys()), values=list(embedding.sparse.values())
                    ),
                },
                payload=record.model_dump(),
            )
            for record, embedding in zip(records, embeddings, strict=True)
        ]
        batch = settings.upsert_batch_size
        for start in range(0, len(points), batch):
            self._client.upsert(self.collection, points=points[start : start + batch], wait=True)
        return len(points)

    def prune(self, keep: set[str]) -> int:
        """Удаляет записи, которых нет в новой сборке; возвращает число удалённых точек."""
        if not self.exists():
            return 0
        stale = [record_id for record_id in self._record_ids() if record_id not in keep]
        for start in range(0, len(stale), self._settings.upsert_batch_size):
            self._client.delete(
                self.collection,
                points_selector=models.PointIdsList(
                    points=list(stale[start : start + self._settings.upsert_batch_size])
                ),
                wait=True,
            )
        return len(stale)

    def count(self) -> int:
        return int(self._client.count(self.collection, exact=True).count) if self.exists() else 0

    def by_term(self, term_key: str, limit: int) -> list[GlossaryRecord]:
        """Точное совпадение нормализованного термина."""
        if not self.exists():
            return []
        points, _ = self._client.scroll(
            self.collection,
            scroll_filter=models.Filter(
                must=[models.FieldCondition(key=TERM_FIELD, match=models.MatchValue(value=term_key))]
            ),
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        return [GlossaryRecord.model_validate(point.payload) for point in points if point.payload]

    def search(self, embedding: Embedding, limit: int) -> list[GlossaryRecord]:
        """Гибридный поиск по записям: dense + sparse prefetch → серверный RRF."""
        if not self.exists():
            return []
        settings = self._settings
        points = self._client.query_points(
            self.collection,
            prefetch=[
                models.Prefetch(query=embedding.dense, using=settings.dense_vector, limit=limit),
                models.Prefetch(
                    query=models.SparseVector(
                        indices=list(embedding.sparse.keys()), values=list(embedding.sparse.values())
                    ),
                    using=settings.sparse_vector,
                    limit=limit,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit,
            with_payload=True,
        ).points
        return [GlossaryRecord.model_validate(point.payload) for point in points if point.payload]

    def records(self) -> list[GlossaryRecord]:
        """Все записи коллекции (глоссарий мал: отчёт и проверки читают его целиком)."""
        if not self.exists():
            return []
        found: list[GlossaryRecord] = []
        offset: models.ExtendedPointId | None = None
        while True:
            points, offset = self._client.scroll(
                self.collection,
                limit=self._settings.upsert_batch_size * 16,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            found.extend(GlossaryRecord.model_validate(point.payload) for point in points if point.payload)
            if offset is None:
                return found

    def _record_ids(self) -> list[str]:
        ids: list[str] = []
        offset: models.ExtendedPointId | None = None
        while True:
            points, offset = self._client.scroll(
                self.collection,
                limit=self._settings.upsert_batch_size * 16,
                offset=offset,
                with_payload=False,
                with_vectors=False,
            )
            ids.extend(str(point.id) for point in points)
            if offset is None:
                return ids
