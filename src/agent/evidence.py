"""Реестр свидетельств (6.2): документы и фрагменты, полученные агентом от инструментов за сессию.

Документы и фрагменты получают короткие псевдонимы (D1, S1): LLM оперирует ими вместо UUID
(8B-модель искажает длинные идентификаторы, а псевдонимы ещё и короче в токенах); реестр переводит
псевдонимы обратно в UUID при вызове инструментов и в цитаты при ответе (6.5). Реестр живёт всю
сессию диалога и служит кэшем найденных документов (FR-6, 6.4): псевдонимы стабильны между вопросами.
"""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, Field

DOC_PREFIX = "D"
FRAGMENT_PREFIX = "S"
ALIAS_RE = re.compile(rf"^(?:{DOC_PREFIX}|{FRAGMENT_PREFIX})\d+$")
ID_ARGUMENTS = ("doc_id", "section_id")
FILTERS_ARGUMENT = "filters"
DOC_IDS_FILTER = "doc_ids"
STATUS_RU = {"active": "действует", "cancelled": "отменён", "draft": "проект"}
UNKNOWN_STATUS = "статус неизвестен"

FragmentKind = Literal["hit", "section"]


def status_text(doc_status: str | None) -> str:
    return STATUS_RU.get(doc_status or "", doc_status or UNKNOWN_STATUS)


class KnownDocument(BaseModel):
    alias: str
    doc_id: str
    label: str = Field(description="Документ: вид, номер, дата")
    doc_kind: str | None = None
    doc_number: str | None = None
    doc_date: str | None = None
    doc_status: str | None = None
    doc_status_name: str | None = None
    subject: str | None = None
    department: str | None = None
    card_text: str | None = Field(default=None, description="Сводка карточки, если её запрашивали")
    relations: list[str] = Field(default_factory=list, description="Связи текстом, если их запрашивали")
    last_used: int = 0

    @property
    def status(self) -> str:
        return status_text(self.doc_status)


class Fragment(BaseModel):
    alias: str
    chunk_id: str
    parent_id: str | None
    doc_id: str
    doc_alias: str
    kind: FragmentKind = Field(description="hit — фрагмент поиска, section — раздел из get_document_content")
    breadcrumbs: str
    clause: str | None = None
    page_no: int | None = None
    text: str
    context: str | None = Field(default=None, description="Текст раздела-родителя, если есть")
    score: float | None = None
    last_used: int = 0


class ToolCallRecord(BaseModel):
    name: str
    arguments: dict[str, Any] = Field(description="Аргументы после замены псевдонимов на ID")
    ok: bool
    summary: str = Field(description="Человекочитаемый итог для шагов UI")
    seconds: float
    query: str | None = Field(default=None, description="Поисковый запрос, если это был поиск")
    fragment_aliases: list[str] = Field(default_factory=list)
    document_aliases: list[str] = Field(default_factory=list)


class EvidenceSnapshot(BaseModel):
    """Часть реестра для сохранения с ответом: по ней диалог восстанавливается из БД (этап 7)."""

    documents: list[KnownDocument] = Field(default_factory=list)
    fragments: list[Fragment] = Field(default_factory=list)


def _alias_number(alias: str) -> int:
    return int(alias[1:])


class EvidenceRegistry:
    """Документы и фрагменты сессии с псевдонимами; вытеснение старых документов по LRU."""

    def __init__(self, max_documents: int) -> None:
        self._max_documents = max_documents
        self._documents: dict[str, KnownDocument] = {}
        self._fragments: dict[str, Fragment] = {}
        self._doc_aliases: dict[str, str] = {}
        self._fragment_aliases: dict[str, str] = {}
        self._doc_counter = 0
        self._fragment_counter = 0
        self._clock = 0

    def _tick(self) -> int:
        self._clock += 1
        return self._clock

    # ---------- регистрация ----------

    def register_document(
        self, doc_id: str, *, label: str | None = None, authoritative: bool = True, **fields: Any
    ) -> KnownDocument:
        """Регистрирует документ или обновляет известный; непустые поля перекрывают старые.

        `authoritative=False` — источник слабый (описание из секции связей): подпись не перекрывает
        ту, что пришла из поиска или карточки."""
        known = self._documents.get(doc_id)
        if known is None:
            self._doc_counter += 1
            known = KnownDocument(alias=f"{DOC_PREFIX}{self._doc_counter}", doc_id=doc_id, label=label or "")
            self._documents[doc_id] = known
            self._doc_aliases[known.alias] = doc_id
        elif label and (authoritative or not known.label):
            known.label = label
        for key, value in fields.items():
            if value is not None and value != "" and value != []:
                setattr(known, key, value)
        known.last_used = self._tick()
        return known

    def register_fragment(
        self,
        chunk_id: str,
        *,
        doc_id: str,
        kind: FragmentKind,
        breadcrumbs: str,
        text: str,
        parent_id: str | None = None,
        clause: str | None = None,
        page_no: int | None = None,
        context: str | None = None,
        score: float | None = None,
    ) -> Fragment:
        document = self._documents.get(doc_id)
        if document is None:
            document = self.register_document(doc_id)
        fragment = self._fragments.get(chunk_id)
        if fragment is None:
            self._fragment_counter += 1
            fragment = Fragment(
                alias=f"{FRAGMENT_PREFIX}{self._fragment_counter}",
                chunk_id=chunk_id,
                parent_id=parent_id,
                doc_id=doc_id,
                doc_alias=document.alias,
                kind=kind,
                breadcrumbs=breadcrumbs,
                clause=clause,
                page_no=page_no,
                text=text,
                context=context,
                score=score,
            )
            self._fragments[chunk_id] = fragment
            self._fragment_aliases[fragment.alias] = chunk_id
        else:
            fragment.text = text or fragment.text
            fragment.context = context or fragment.context
            fragment.score = score if score is not None else fragment.score
        fragment.last_used = self._tick()
        document.last_used = fragment.last_used
        return fragment

    # ---------- доступ ----------

    def document(self, doc_id: str) -> KnownDocument | None:
        return self._documents.get(doc_id)

    def fragment(self, chunk_id: str) -> Fragment | None:
        return self._fragments.get(chunk_id)

    def document_by_alias(self, alias: str) -> KnownDocument | None:
        doc_id = self._doc_aliases.get(alias.strip())
        return self._documents.get(doc_id) if doc_id else None

    def fragment_by_alias(self, alias: str) -> Fragment | None:
        chunk_id = self._fragment_aliases.get(alias.strip())
        return self._fragments.get(chunk_id) if chunk_id else None

    def documents(self) -> list[KnownDocument]:
        return sorted(self._documents.values(), key=lambda item: _alias_number(item.alias))

    def fragments(self) -> list[Fragment]:
        return sorted(self._fragments.values(), key=lambda item: _alias_number(item.alias))

    def fragments_of(self, doc_id: str) -> list[Fragment]:
        return [fragment for fragment in self.fragments() if fragment.doc_id == doc_id]

    def touch(self, doc_id: str) -> None:
        document = self._documents.get(doc_id)
        if document is not None:
            document.last_used = self._tick()

    # ---------- псевдонимы в аргументах инструментов ----------

    def resolve(self, value: Any) -> Any:
        """Псевдоним D#/S# → UUID документа или фрагмента; всё остальное — как есть."""
        if not isinstance(value, str) or not ALIAS_RE.match(value.strip()):
            return value
        alias = value.strip()
        if alias.startswith(DOC_PREFIX):
            return self._doc_aliases.get(alias, value)
        return self._fragment_aliases.get(alias, value)

    def resolve_arguments(self, arguments: dict[str, Any]) -> dict[str, Any]:
        resolved = dict(arguments)
        for key in ID_ARGUMENTS:
            if key in resolved:
                resolved[key] = self.resolve(resolved[key])
        filters = resolved.get(FILTERS_ARGUMENT)
        if isinstance(filters, dict) and isinstance(filters.get(DOC_IDS_FILTER), list):
            filters = dict(filters)
            filters[DOC_IDS_FILTER] = [self.resolve(item) for item in filters[DOC_IDS_FILTER]]
            resolved[FILTERS_ARGUMENT] = filters
        return resolved

    # ---------- снимок и восстановление (возобновление диалога из БД) ----------

    def snapshot(self, document_aliases: list[str], fragment_aliases: list[str]) -> EvidenceSnapshot:
        """Документы и фрагменты с указанными псевдонимами (плюс документы фрагментов)."""
        fragments = [
            fragment for alias in fragment_aliases if (fragment := self.fragment_by_alias(alias)) is not None
        ]
        wanted = list(document_aliases) + [fragment.doc_alias for fragment in fragments]
        documents: dict[str, KnownDocument] = {}
        for alias in wanted:
            document = self.document_by_alias(alias)
            if document is not None:
                documents.setdefault(document.alias, document)
        return EvidenceSnapshot(
            documents=[item.model_copy() for item in documents.values()],
            fragments=[item.model_copy() for item in fragments],
        )

    def restore(self, snapshot: EvidenceSnapshot) -> None:
        """Возвращает документы и фрагменты в реестр с их прежними псевдонимами.

        Уже известный документ (по ID) обновляется непустыми полями; конфликт псевдонима с другим
        документом невозможен при восстановлении по порядку ходов, а на всякий случай запись
        с занятым псевдонимом пропускается."""
        for document in snapshot.documents:
            known = self._documents.get(document.doc_id)
            if known is not None:
                self.register_document(
                    document.doc_id,
                    label=document.label,
                    **document.model_dump(exclude={"alias", "doc_id", "label"}),
                )
                continue
            if document.alias in self._doc_aliases:
                continue
            self._documents[document.doc_id] = document.model_copy()
            self._doc_aliases[document.alias] = document.doc_id
            self._doc_counter = max(self._doc_counter, _alias_number(document.alias))
        for fragment in snapshot.fragments:
            if fragment.chunk_id in self._fragments or fragment.alias in self._fragment_aliases:
                continue
            if fragment.doc_id not in self._documents:
                continue
            self._fragments[fragment.chunk_id] = fragment.model_copy()
            self._fragment_aliases[fragment.alias] = fragment.chunk_id
            self._fragment_counter = max(self._fragment_counter, _alias_number(fragment.alias))
        self._clock = max([self._clock, *(d.last_used for d in self._documents.values())])

    # ---------- кэш сессии ----------

    def trim(self) -> list[str]:
        """Вытесняет давно не использованные документы сверх лимита вместе с их фрагментами."""
        overflow = len(self._documents) - self._max_documents
        if overflow <= 0:
            return []
        victims = sorted(self._documents.values(), key=lambda item: item.last_used)[:overflow]
        evicted: list[str] = []
        for document in victims:
            for fragment in self.fragments_of(document.doc_id):
                del self._fragments[fragment.chunk_id]
                del self._fragment_aliases[fragment.alias]
            del self._documents[document.doc_id]
            del self._doc_aliases[document.alias]
            evicted.append(document.alias)
        return evicted
