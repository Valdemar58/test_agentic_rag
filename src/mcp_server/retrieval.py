"""Гибридный поиск (FR-2.1): Qdrant prefetch dense + sparse → серверный RRF → rerank на CPU → top_k.

Фильтры — по payload чанков из инжеста (`doc_kind`, `doc_status`, `doc_date_ts`, `department`,
`doc_id`); без явного фильтра статуса берутся `retrieval.default_statuses` — только действующие
документы (AC-2.2). Фильтр передаётся в каждый prefetch: при fusion Qdrant применяет условия
именно там. Вместе с child-чанком агент получает текст parent-раздела (parent-child, FR-3).
"""

from __future__ import annotations

import datetime as dt
import logging
from typing import Any

from pydantic import BaseModel, Field
from qdrant_client import QdrantClient, models

from common.config import DocStatus, QdrantSettings, RetrievalSettings
from ingest.embeddings import Embedder
from ingest.metadata import ChunkPayload
from mcp_server.reranker import Reranker

logger = logging.getLogger(__name__)

KIND_FIELD = "doc_kind"
STATUS_FIELD = "doc_status"
DATE_FIELD = "doc_date_ts"
DEPARTMENT_FIELD = "department"
DOC_ID_FIELD = "doc_id"


class SearchFilters(BaseModel):
    """Фильтры поиска; пустое поле — без ограничения. Статус по умолчанию — только действующие."""

    doc_kinds: list[str] | None = Field(
        default=None,
        description="Виды документов как в СЭД: «Приказ», «Договорной документ», «Служебная записка», "
        "«Справка-обоснование», «Входящее письмо», «Исходящее письмо», «Иной документ»; "
        "регистр и часть названия допускаются («договор»)",
    )
    statuses: list[DocStatus] | None = Field(
        default=None,
        description="Статусы документов: active — действующие (по умолчанию), cancelled — отменённые, "
        "draft — проекты и документы в работе",
    )
    date_from: dt.date | None = Field(default=None, description="Дата документа не раньше (ГГГГ-ММ-ДД)")
    date_to: dt.date | None = Field(default=None, description="Дата документа не позже (ГГГГ-ММ-ДД)")
    departments: list[str] | None = Field(
        default=None, description="Подразделения-инициаторы, как в карточке; допускается часть названия"
    )
    doc_ids: list[str] | None = Field(default=None, description="Искать только внутри этих документов (ID)")


class SearchHit(BaseModel):
    """Найденный фрагмент с координатами в документе и текстом родительского раздела."""

    chunk_id: str = Field(description="ID фрагмента (для цитаты)")
    parent_id: str | None = Field(description="ID раздела-родителя (section_id для get_document_content)")
    doc_id: str = Field(description="ID документа (карточки СЭД)")
    score: float = Field(description="Оценка reranker'а, 0…1")
    fusion_rank: int = Field(description="Место после гибридного слияния dense + sparse (1 — лучшее)")
    doc_kind: str
    doc_number: str | None
    doc_date: str | None
    doc_status: DocStatus
    doc_status_name: str | None
    department: str | None
    subject: str | None
    file_name: str
    file_role: str = Field(
        description="main — основной текст, appendix — приложение, supplement — дополнение"
    )
    breadcrumbs: str = Field(description="Путь: документ → раздел → пункт")
    section_path: list[str]
    clause: str | None = Field(description="Номер пункта, если фрагмент — пункт")
    heading: str | None
    page_no: int | None
    text: str = Field(description="Текст фрагмента")
    context: str | None = Field(description="Текст раздела-родителя целиком, если он есть")


class SearchResult(BaseModel):
    query: str
    hits: list[SearchHit]
    candidates: int = Field(description="Сколько кандидатов вернул гибридный поиск до reranker'а")
    applied_filters: dict[str, Any] = Field(description="Фильтры, которые реально применились")
    notes: list[str] = Field(
        default_factory=list, description="Замечания: неизвестные значения фильтров и т.п."
    )


def _casefold(value: str) -> str:
    return " ".join(value.casefold().split())


def match_known_values(requested: list[str], known: list[str]) -> tuple[list[str], list[str]]:
    """Сопоставляет значения из запроса со значениями индекса: точно без регистра, затем по вхождению.

    Возвращает (значения для фильтра, неопознанные значения). Неопознанное значение остаётся в
    фильтре как есть — результат будет пустым, а в notes попадёт подсказка."""
    resolved: list[str] = []
    unknown: list[str] = []
    for raw in requested:
        wanted = _casefold(raw)
        if not wanted:
            continue
        exact = [item for item in known if _casefold(item) == wanted]
        partial = exact or [item for item in known if wanted in _casefold(item) or _casefold(item) in wanted]
        if partial:
            resolved.extend(item for item in partial if item not in resolved)
        else:
            unknown.append(raw)
            resolved.append(raw)
    return resolved, unknown


def _day_start_ts(day: dt.date) -> int:
    return int(dt.datetime(day.year, day.month, day.day, tzinfo=dt.UTC).timestamp())


class HybridSearcher:
    def __init__(
        self,
        client: QdrantClient,
        qdrant: QdrantSettings,
        retrieval: RetrievalSettings,
        embedder: Embedder,
        reranker: Reranker,
    ) -> None:
        self._client = client
        self._qdrant = qdrant
        self._retrieval = retrieval
        self._embedder = embedder
        self._reranker = reranker
        self._known: dict[str, list[str]] = {}

    def warm_up(self) -> None:
        """Прогоняет модели на коротком тексте: веса читаются с диска и инициализируются заранее."""
        self._embedder.encode(["прогрев"])
        self._reranker.score("прогрев", ["прогрев моделей"])
        for field in (KIND_FIELD, DEPARTMENT_FIELD):
            self.known_values(field, refresh=True)

    def known_values(self, field: str, *, refresh: bool = False) -> list[str]:
        """Различные значения поля в индексе (facet), кэшируются на время жизни сервера."""
        if refresh or field not in self._known:
            response = self._client.facet(
                self._qdrant.collection, key=field, limit=self._retrieval.known_values_limit, exact=True
            )
            self._known[field] = sorted(str(hit.value) for hit in response.hits)
        return self._known[field]

    def build_filter(self, filters: SearchFilters | None) -> tuple[models.Filter, dict[str, Any], list[str]]:
        """Qdrant-фильтр, применённые значения и замечания по неопознанным значениям."""
        filters = filters or SearchFilters()
        must: list[models.Condition] = []
        applied: dict[str, Any] = {}
        notes: list[str] = []

        statuses = list(filters.statuses or self._retrieval.default_statuses)
        must.append(models.FieldCondition(key=STATUS_FIELD, match=models.MatchAny(any=statuses)))
        applied["statuses"] = statuses

        for name, values, field in (
            ("doc_kinds", filters.doc_kinds, KIND_FIELD),
            ("departments", filters.departments, DEPARTMENT_FIELD),
        ):
            if not values:
                continue
            resolved, unknown = match_known_values(values, self.known_values(field))
            if unknown:
                known = ", ".join(f"«{item}»" for item in self.known_values(field)) or "нет значений"
                missing = ", ".join(f"«{item}»" for item in unknown)
                notes.append(f"{name}: не найдено в индексе {missing}; есть: {known}")
            if resolved:
                must.append(models.FieldCondition(key=field, match=models.MatchAny(any=resolved)))
                applied[name] = resolved

        if filters.date_from or filters.date_to:
            gte = _day_start_ts(filters.date_from) if filters.date_from else None
            lt = _day_start_ts(filters.date_to + dt.timedelta(days=1)) if filters.date_to else None
            must.append(models.FieldCondition(key=DATE_FIELD, range=models.Range(gte=gte, lt=lt)))
            applied["date_from"] = filters.date_from.isoformat() if filters.date_from else None
            applied["date_to"] = filters.date_to.isoformat() if filters.date_to else None

        if filters.doc_ids:
            must.append(
                models.FieldCondition(key=DOC_ID_FIELD, match=models.MatchAny(any=list(filters.doc_ids)))
            )
            applied["doc_ids"] = list(filters.doc_ids)
        return models.Filter(must=must), applied, notes

    def search(
        self, query: str, filters: SearchFilters | None = None, top_k: int | None = None
    ) -> SearchResult:
        """dense + sparse prefetch → RRF → rerank top-N → top_k; вместе с parent-разделами."""
        limit = min(top_k or self._retrieval.top_k, self._retrieval.max_top_k)
        limit = max(limit, 1)
        qdrant_filter, applied, notes = self.build_filter(filters)
        embedding = self._embedder.encode([query])[0]
        prefetch_limit = self._retrieval.prefetch_limit
        points = self._client.query_points(
            self._qdrant.collection,
            prefetch=[
                models.Prefetch(
                    query=embedding.dense,
                    using=self._qdrant.dense_vector,
                    limit=prefetch_limit,
                    filter=qdrant_filter,
                ),
                models.Prefetch(
                    query=models.SparseVector(
                        indices=list(embedding.sparse.keys()), values=list(embedding.sparse.values())
                    ),
                    using=self._qdrant.sparse_vector,
                    limit=prefetch_limit,
                    filter=qdrant_filter,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=self._retrieval.rerank_candidates,
            with_payload=True,
        ).points
        candidates = [ChunkPayload.model_validate(point.payload) for point in points if point.payload]
        scores = self._reranker.score(query, [candidate.text for candidate in candidates])
        ranked = sorted(
            zip(range(1, len(candidates) + 1), candidates, scores, strict=True), key=lambda item: -item[2]
        )[:limit]
        parents = self._parents([candidate.parent_id for _, candidate, _ in ranked])
        hits = [
            _hit(candidate, score, rank, parents.get(candidate.parent_id or ""))
            for rank, candidate, score in ranked
        ]
        logger.info("hybrid_search «%s»: кандидатов %d, отдано %d", query[:60], len(candidates), len(hits))
        return SearchResult(
            query=query, hits=hits, candidates=len(candidates), applied_filters=applied, notes=notes
        )

    def _parents(self, parent_ids: list[str | None]) -> dict[str, ChunkPayload]:
        wanted = sorted({parent_id for parent_id in parent_ids if parent_id})
        if not wanted or not self._retrieval.return_parent:
            return {}
        records = self._client.retrieve(self._qdrant.parents_collection, ids=list(wanted), with_payload=True)
        return {
            str(record.id): ChunkPayload.model_validate(record.payload)
            for record in records
            if record.payload
        }


def _hit(chunk: ChunkPayload, score: float, fusion_rank: int, parent: ChunkPayload | None) -> SearchHit:
    return SearchHit(
        chunk_id=chunk.chunk_id,
        parent_id=chunk.parent_id,
        doc_id=chunk.doc_id,
        score=round(score, 4),
        fusion_rank=fusion_rank,
        doc_kind=chunk.doc_kind,
        doc_number=chunk.doc_number,
        doc_date=chunk.doc_date,
        doc_status=chunk.doc_status,
        doc_status_name=chunk.doc_status_name,
        department=chunk.department,
        subject=chunk.subject,
        file_name=chunk.file_name,
        file_role=chunk.file_role,
        breadcrumbs=chunk.breadcrumbs,
        section_path=chunk.section_path,
        clause=chunk.clause,
        heading=chunk.heading,
        page_no=chunk.page_no,
        text=chunk.body,
        context=parent.body if parent is not None else None,
    )
