"""Parent-child (FR-3): индексируются child-чанки (пункт/абзац), агенту возвращается parent — раздел.

Parent собирается из подряд идущих child-чанков файла с одним `section_key` (при
`ingest.parent_level: section`) или из всех чанков файла (`document`). Раздел длиннее
`parent_max_tokens` делится на несколько parent-окон. У каждого child проставляется `parent_id`;
parent хранит список `child_ids`. Идентификаторы детерминированы (uuid5 от sha256 файла), поэтому
повторный инжест даёт те же id (AC-3.3). Эквивалент AutoMergingRetriever: поиск идёт по child,
а по `parent_id` из payload Qdrant возвращается раздел целиком.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, replace
from typing import Literal

from common.config import ChunkingSettings
from ingest.chunking import CHUNK_NAMESPACE, Chunk
from ingest.tokens import TokenCounter

ParentLevel = Literal["section", "document"]
PARENT_BODY_SEPARATOR = "\n\n"


@dataclass(frozen=True)
class ParentChunk:
    chunk_id: str
    ordinal: int
    breadcrumbs: tuple[str, ...]
    section_key: str
    heading: str | None
    body: str
    text: str
    tokens: int
    page_no: int | None
    child_ids: tuple[str, ...]
    part: int
    parts: int


@dataclass(frozen=True)
class ChunkSet:
    children: list[Chunk]
    parents: list[ParentChunk]

    def parent_of(self, child: Chunk) -> ParentChunk | None:
        return next((parent for parent in self.parents if parent.chunk_id == child.parent_id), None)


def _section_crumbs(chunk: Chunk) -> tuple[str, ...]:
    """Крошки раздела без пункта: у структурных чанков последняя крошка — «п. N» при наличии clause."""
    if chunk.clause is not None and chunk.breadcrumbs and chunk.breadcrumbs[-1].endswith(chunk.clause):
        return chunk.breadcrumbs[:-1]
    return chunk.breadcrumbs


def _group_key(chunk: Chunk, level: ParentLevel) -> tuple[str, ...]:
    if level == "document":
        return ()
    return _section_crumbs(chunk)


def build_chunk_set(
    chunks: list[Chunk],
    settings: ChunkingSettings,
    counter: TokenCounter,
    *,
    file_sha256: str,
    level: ParentLevel = "section",
) -> ChunkSet:
    """Разбивает child-чанки на разделы и окна, создаёт parent-чанки и связывает их с детьми."""
    if not chunks:
        return ChunkSet(children=[], parents=[])
    separator_tokens = counter.count(PARENT_BODY_SEPARATOR)

    groups: list[list[Chunk]] = []
    for chunk in chunks:
        if groups and _group_key(groups[-1][-1], level) == _group_key(chunk, level):
            groups[-1].append(chunk)
        else:
            groups.append([chunk])

    windows: list[list[Chunk]] = []
    for group in groups:
        prefix_tokens = counter.count(settings.breadcrumb_separator.join(_section_crumbs(group[0])))
        current: list[Chunk] = []
        current_tokens = prefix_tokens
        for chunk in group:
            body_tokens = counter.count(chunk.body)
            if current and current_tokens + separator_tokens + body_tokens > settings.parent_max_tokens:
                windows.append(current)
                current, current_tokens = [], prefix_tokens
            current.append(chunk)
            current_tokens += body_tokens + (separator_tokens if len(current) > 1 else 0)
        if current:
            windows.append(current)

    parts_by_key: dict[tuple[str, ...], int] = {}
    for window in windows:
        key = _group_key(window[0], level)
        parts_by_key[key] = parts_by_key.get(key, 0) + 1

    parents: list[ParentChunk] = []
    children: list[Chunk] = []
    part_counter: dict[tuple[str, ...], int] = {}
    for ordinal, window in enumerate(windows):
        key = _group_key(window[0], level)
        part_counter[key] = part_counter.get(key, 0) + 1
        crumbs = _section_crumbs(window[0]) if level == "section" else window[0].breadcrumbs[:1]
        prefix = settings.breadcrumb_separator.join(crumbs)
        body = PARENT_BODY_SEPARATOR.join(chunk.body for chunk in window)
        text = f"{prefix}\n{body}" if prefix else body
        parent_id = str(uuid.uuid5(CHUNK_NAMESPACE, f"{file_sha256}:parent:{ordinal}"))
        parents.append(
            ParentChunk(
                chunk_id=parent_id,
                ordinal=ordinal,
                breadcrumbs=crumbs,
                section_key=window[0].section_key if level == "section" else "",
                heading=window[0].heading if level == "section" else None,
                body=body,
                text=text,
                tokens=counter.count(text),
                page_no=window[0].page_no,
                child_ids=tuple(chunk.chunk_id for chunk in window),
                part=part_counter[key],
                parts=parts_by_key[key],
            )
        )
        children.extend(replace(chunk, parent_id=parent_id) for chunk in window)
    return ChunkSet(children=children, parents=parents)
