"""`get_document_content` (FR-2.3): полный текст документа или раздела из проиндексированного корпуса.

Текст берётся из parent-разделов коллекции `documents_parents` (те же разделы, что агент видит в
`context` результата поиска): без `section_id` — все разделы документа по порядку (основной файл,
затем приложения и дополнения, внутри файла — по позиции), частями по бюджету токенов;
с `section_id` — один раздел, причём принимается и id child-чанка (тогда отдаётся его раздел).
"""

from __future__ import annotations

import datetime as dt
import logging

from pydantic import BaseModel, Field
from qdrant_client import QdrantClient, models

from common.config import DocStatus, QdrantSettings, RetrievalSettings
from ingest.metadata import ChunkPayload, document_label

logger = logging.getLogger(__name__)

ROLE_ORDER = {"main": 0, "appendix": 1, "supplement": 2}
NOT_INDEXED_NOTE = (
    "текст документа отсутствует в индексе (файлы не проиндексированы или документ вне корпуса)"
)
OTHER_DOCUMENT_NOTE = (
    "раздел относится к документу {label} ({doc_id}), а не к запрошенному {requested}: текст приложения "
    "индексируется вместе с карточкой, к которой приложен файл. Ссылайся на документ, указанный здесь"
)


class ContentNotFoundError(Exception):
    """Раздел или документ не найден в индексе."""


class ContentSection(BaseModel):
    section_id: str = Field(description="ID раздела (parent-чанка)")
    breadcrumbs: str = Field(description="Путь: документ → файл → раздел")
    heading: str | None
    file_name: str
    file_role: str = Field(
        description="main — основной текст, appendix — приложение, supplement — дополнение"
    )
    page_no: int | None
    tokens: int
    text: str


class DocumentContent(BaseModel):
    doc_id: str
    label: str | None = Field(description="Документ: вид, номер, дата")
    doc_kind: str | None
    doc_number: str | None
    doc_date: str | None
    doc_status: DocStatus | None
    subject: str | None
    sections: list[ContentSection]
    total_sections: int = Field(description="Всего разделов у документа (или 1, если запрошен раздел)")
    offset: int = Field(description="Индекс первого возвращённого раздела")
    next_offset: int | None = Field(description="С какого раздела продолжать, если ответ обрезан")
    truncated: bool
    note: str | None = None


def _label(payload: ChunkPayload) -> str:
    date = dt.date.fromisoformat(payload.doc_date) if payload.doc_date else None
    return document_label(payload.doc_kind, payload.doc_number, date)


def _section(payload: ChunkPayload) -> ContentSection:
    return ContentSection(
        section_id=payload.chunk_id,
        breadcrumbs=payload.breadcrumbs,
        heading=payload.heading,
        file_name=payload.file_name,
        file_role=payload.file_role,
        page_no=payload.page_no,
        tokens=payload.tokens,
        text=payload.body,
    )


def _order_key(payload: ChunkPayload) -> tuple[int, str, int]:
    return (ROLE_ORDER.get(payload.file_role, len(ROLE_ORDER)), payload.file_name, payload.chunk_index)


class DocumentReader:
    def __init__(self, client: QdrantClient, qdrant: QdrantSettings, retrieval: RetrievalSettings) -> None:
        self._client = client
        self._qdrant = qdrant
        self._retrieval = retrieval

    def read(self, doc_id: str, section_id: str | None = None, offset: int = 0) -> DocumentContent:
        if section_id:
            return self._read_section(doc_id, section_id)
        parents = self._parents_of(doc_id)
        if not parents:
            return DocumentContent(
                doc_id=doc_id,
                label=None,
                doc_kind=None,
                doc_number=None,
                doc_date=None,
                doc_status=None,
                subject=None,
                sections=[],
                total_sections=0,
                offset=0,
                next_offset=None,
                truncated=False,
                note=NOT_INDEXED_NOTE,
            )
        start = min(max(offset, 0), len(parents) - 1)
        chosen: list[ChunkPayload] = []
        tokens = 0
        for payload in parents[start:]:
            over_budget = tokens + payload.tokens > self._retrieval.content_max_tokens
            if chosen and (over_budget or len(chosen) >= self._retrieval.content_max_sections):
                break
            chosen.append(payload)
            tokens += payload.tokens
        next_offset = start + len(chosen)
        truncated = next_offset < len(parents)
        first = parents[0]
        logger.info(
            "get_document_content %s: разделов %d, отдано %d с %d, токенов %d%s",
            doc_id,
            len(parents),
            len(chosen),
            start,
            tokens,
            " (обрезано)" if truncated else "",
        )
        return DocumentContent(
            doc_id=doc_id,
            label=_label(first),
            doc_kind=first.doc_kind,
            doc_number=first.doc_number,
            doc_date=first.doc_date,
            doc_status=first.doc_status,
            subject=first.subject,
            sections=[_section(payload) for payload in chosen],
            total_sections=len(parents),
            offset=start,
            next_offset=next_offset if truncated else None,
            truncated=truncated,
            note=None,
        )

    def _read_section(self, doc_id: str, section_id: str) -> DocumentContent:
        payload = self._parent_by_id(section_id)
        if payload is None:
            raise ContentNotFoundError(f"раздел {section_id} не найден в индексе")
        note = None
        if payload.doc_id != doc_id:
            # раздел найден, но у другого документа: отказ заставлял агента повторять вызов и тратить
            # бюджет (живой прогон 2026-09-25 — пять одинаковых ошибок подряд). Отдаём текст и говорим,
            # какому документу он принадлежит: ссылаться агент должен на него.
            note = OTHER_DOCUMENT_NOTE.format(label=_label(payload), doc_id=payload.doc_id, requested=doc_id)
            logger.info("get_document_content: раздел %s отдан из документа %s", section_id, payload.doc_id)
        return DocumentContent(
            doc_id=payload.doc_id,
            label=_label(payload),
            doc_kind=payload.doc_kind,
            doc_number=payload.doc_number,
            doc_date=payload.doc_date,
            doc_status=payload.doc_status,
            subject=payload.subject,
            sections=[_section(payload)],
            total_sections=1,
            offset=0,
            next_offset=None,
            truncated=False,
            note=note,
        )

    def _parent_by_id(self, section_id: str) -> ChunkPayload | None:
        """Раздел по id parent-чанка; id child-чанка тоже принимается — возвращается его раздел."""
        parent = self._retrieve(self._qdrant.parents_collection, section_id)
        if parent is not None:
            return parent
        child = self._retrieve(self._qdrant.collection, section_id)
        if child is None or not child.parent_id:
            return None
        return self._retrieve(self._qdrant.parents_collection, child.parent_id)

    def _retrieve(self, collection: str, point_id: str) -> ChunkPayload | None:
        try:
            records = self._client.retrieve(collection, ids=[point_id], with_payload=True)
        except (ValueError, TypeError):
            return None  # id не UUID — такой точки быть не может
        if not records or records[0].payload is None:
            return None
        return ChunkPayload.model_validate(records[0].payload)

    def _parents_of(self, doc_id: str) -> list[ChunkPayload]:
        found: list[ChunkPayload] = []
        offset: models.ExtendedPointId | None = None
        doc_filter = models.Filter(
            must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))]
        )
        while True:
            points, offset = self._client.scroll(
                self._qdrant.parents_collection,
                scroll_filter=doc_filter,
                limit=self._qdrant.upsert_batch_size,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            found.extend(ChunkPayload.model_validate(point.payload) for point in points if point.payload)
            if offset is None:
                break
        return sorted(found, key=_order_key)
