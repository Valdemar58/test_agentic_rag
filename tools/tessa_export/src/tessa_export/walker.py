"""Рекурсивный обход связей от seed-списка (§8.1.2–8.1.4 ТЗ).

BFS с множеством посещённых карточек: циклы не зацикливают и не дублируют документы,
глубина считается от ближайшего seed, при срабатывании max_depth / max_docs обход
останавливается корректно, а все непройденные связи попадают в реестр с причиной.
Исключённые правилами документы фиксируются без содержимого, их связи дальше не обходятся.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Literal
from uuid import UUID

from tessa_export.config import Direction, ExcludeRule, TraversalSettings
from tessa_export.models import (
    CardAccessError,
    CardNotFoundError,
    CardSnapshot,
    GatewayConnectionError,
    GatewayError,
    LinkInfo,
    TessaGateway,
)

logger = logging.getLogger(__name__)

SkipReason = Literal["max_depth", "max_docs", "excluded", "error"]
ErrorKind = Literal["not_found", "access", "connection", "other"]


@dataclass(frozen=True)
class EntryPath:
    """Как документ попал в сет: seed или по связи от другого документа."""

    kind: Literal["seed", "link"]
    depth: int
    via_card_id: UUID | None = None
    relation: str | None = None
    direction: Direction | None = None


@dataclass
class VisitedCard:
    snapshot: CardSnapshot
    depth: int
    entry_paths: list[EntryPath] = field(default_factory=list)


@dataclass
class ExcludedCard:
    """Исключённый документ: только идентификация и причина, без содержимого."""

    card_id: UUID
    reason: str
    depth: int
    type_name: str | None
    doc_type_title: str | None
    entry_paths: list[EntryPath] = field(default_factory=list)


@dataclass
class FetchError:
    card_id: UUID
    depth: int
    kind: ErrorKind
    message: str
    entry_paths: list[EntryPath] = field(default_factory=list)


@dataclass(frozen=True)
class SkippedLink:
    """Связь, по которой обход не пошёл, с причиной."""

    from_card_id: UUID | None
    to_card_id: UUID
    direction: Direction | None
    relation: str | None
    reason: SkipReason
    detail: str = ""


@dataclass(frozen=True)
class LinkEdge:
    """Ребро графа связей. Типизированная исходящая связь важнее входящей без типа."""

    from_card_id: UUID
    to_card_id: UUID
    relation_type: str | None
    reverse_type: str | None
    relation_type_id: UUID | None
    source: Direction


@dataclass
class WalkResult:
    seed_ids: list[UUID]
    cards: dict[UUID, VisitedCard] = field(default_factory=dict)
    excluded: dict[UUID, ExcludedCard] = field(default_factory=dict)
    errors: list[FetchError] = field(default_factory=list)
    skipped_links: list[SkippedLink] = field(default_factory=list)
    edges: list[LinkEdge] = field(default_factory=list)
    dangling_edges: list[LinkEdge] = field(default_factory=list)

    @property
    def attempted(self) -> int:
        return len(self.cards) + len(self.excluded) + len(self.errors)


@dataclass
class _Node:
    depth: int
    entry_paths: list[EntryPath]


def _field_matches(snapshot: CardSnapshot, rule: ExcludeRule) -> bool:
    """Критерий по полю карточки: any_of — значение в списке, none_of — значения в списке нет."""
    assert rule.field is not None  # noqa: S101 — вызывается только при заданном field
    section, name = rule.field.split(".", 1)
    value = snapshot.section_field(section, name)
    wanted = {item.casefold() for item in rule.values}
    text = None if value is None else str(value).casefold()
    if rule.values_mode == "any_of":
        return text is not None and text in wanted
    return text is None or text not in wanted


def match_exclusion(snapshot: CardSnapshot, rules: list[ExcludeRule], *, is_seed: bool = False) -> str | None:
    """Возвращает подпись сработавшего правила или None.

    Явный card_ids действует всегда; критерии по типу/виду/полю к seed-карточке применяются
    только при applies_to_seed (seed выбран заказчиком явно и по умолчанию защищён)."""
    for index, rule in enumerate(rules, start=1):
        label = rule.reason or f"правило {index}"
        if snapshot.card_id in rule.card_ids:
            return label
        if is_seed and not rule.applies_to_seed:
            continue
        if snapshot.type_name and snapshot.type_name in rule.card_type_names:
            return label
        title = snapshot.common_text("DocTypeTitle")
        if title and any(title.casefold() == item.casefold() for item in rule.doc_type_titles):
            return label
        if rule.field and _field_matches(snapshot, rule):
            return label
    return None


def classify_error(exc: GatewayError) -> ErrorKind:
    if isinstance(exc, CardNotFoundError):
        return "not_found"
    if isinstance(exc, CardAccessError):
        return "access"
    if isinstance(exc, GatewayConnectionError):
        return "connection"
    return "other"


class Walker:
    def __init__(
        self,
        gateway: TessaGateway,
        traversal: TraversalSettings,
        exclude_rules: list[ExcludeRule] | None = None,
    ) -> None:
        self._gateway = gateway
        self._traversal = traversal
        self._rules = list(exclude_rules or [])

    def walk(self, seed_ids: list[UUID]) -> WalkResult:
        result = WalkResult(seed_ids=list(seed_ids))
        nodes: dict[UUID, _Node] = {}
        queue: deque[UUID] = deque()
        edges: dict[tuple[UUID, UUID], LinkEdge] = {}

        for card_id in seed_ids:
            if card_id in nodes:
                nodes[card_id].entry_paths.append(EntryPath(kind="seed", depth=0))
                continue
            if len(nodes) >= self._traversal.max_docs:
                result.skipped_links.append(
                    SkippedLink(None, card_id, None, None, "max_docs", "seed сверх лимита max_docs")
                )
                continue
            nodes[card_id] = _Node(depth=0, entry_paths=[EntryPath(kind="seed", depth=0)])
            queue.append(card_id)

        while queue:
            card_id = queue.popleft()
            node = nodes[card_id]
            logger.info(
                "Карточка %s (глубина %d, %d/%d)", card_id, node.depth, len(nodes), self._traversal.max_docs
            )
            try:
                snapshot = self._gateway.get_card(card_id)
            except GatewayError as exc:
                logger.warning("Карточка %s не получена: %s", card_id, exc)
                result.errors.append(
                    FetchError(card_id, node.depth, classify_error(exc), str(exc), node.entry_paths)
                )
                continue

            is_seed = any(path.kind == "seed" for path in node.entry_paths)
            exclusion = match_exclusion(snapshot, self._rules, is_seed=is_seed)
            if exclusion is not None:
                logger.info("Карточка %s исключена: %s", card_id, exclusion)
                result.excluded[card_id] = ExcludedCard(
                    card_id=card_id,
                    reason=exclusion,
                    depth=node.depth,
                    type_name=snapshot.type_name,
                    doc_type_title=snapshot.common_text("DocTypeTitle"),
                    entry_paths=node.entry_paths,
                )
                for direction, links in self._links(snapshot):
                    for item in links:
                        result.skipped_links.append(
                            SkippedLink(
                                card_id, item.doc_id, direction, item.ref_type_name, "excluded", exclusion
                            )
                        )
                continue

            result.cards[card_id] = VisitedCard(
                snapshot=snapshot, depth=node.depth, entry_paths=node.entry_paths
            )

            for direction, links in self._links(snapshot):
                for item in links:
                    self._merge_edge(edges, card_id, item, direction)
                    self._enqueue(result, nodes, queue, card_id, node.depth, item, direction)

        self._split_edges(result, edges)
        return result

    def _links(self, snapshot: CardSnapshot) -> list[tuple[Direction, list[LinkInfo]]]:
        pairs: list[tuple[Direction, list[LinkInfo]]] = []
        for direction in self._traversal.directions:
            links = snapshot.outgoing if direction == "outgoing" else snapshot.incoming
            pairs.append((direction, links))
        return pairs

    def _enqueue(
        self,
        result: WalkResult,
        nodes: dict[UUID, _Node],
        queue: deque[UUID],
        card_id: UUID,
        depth: int,
        item: LinkInfo,
        direction: Direction,
    ) -> None:
        neighbor = item.doc_id
        if neighbor == card_id:
            return
        path = EntryPath(
            kind="link",
            depth=depth + 1,
            via_card_id=card_id,
            relation=item.ref_type_name,
            direction=direction,
        )
        if neighbor in nodes:
            nodes[neighbor].entry_paths.append(path)
            return
        if depth + 1 > self._traversal.max_depth:
            result.skipped_links.append(
                SkippedLink(
                    card_id, neighbor, direction, item.ref_type_name, "max_depth", f"глубина {depth + 1}"
                )
            )
            return
        if len(nodes) >= self._traversal.max_docs:
            result.skipped_links.append(
                SkippedLink(
                    card_id,
                    neighbor,
                    direction,
                    item.ref_type_name,
                    "max_docs",
                    f"лимит {self._traversal.max_docs}",
                )
            )
            return
        nodes[neighbor] = _Node(depth=depth + 1, entry_paths=[path])
        queue.append(neighbor)

    @staticmethod
    def _merge_edge(
        edges: dict[tuple[UUID, UUID], LinkEdge], card_id: UUID, item: LinkInfo, direction: Direction
    ) -> None:
        if direction == "outgoing":
            edge = LinkEdge(
                card_id,
                item.doc_id,
                item.ref_type_name,
                item.ref_type_reverse_name,
                item.ref_type_id,
                "outgoing",
            )
        else:
            edge = LinkEdge(item.doc_id, card_id, None, None, None, "incoming")
        key = (edge.from_card_id, edge.to_card_id)
        existing = edges.get(key)
        if existing is None or (existing.relation_type is None and edge.relation_type is not None):
            edges[key] = edge

    @staticmethod
    def _split_edges(result: WalkResult, edges: dict[tuple[UUID, UUID], LinkEdge]) -> None:
        in_set = set(result.cards)
        for edge in edges.values():
            if edge.from_card_id in in_set and edge.to_card_id in in_set:
                result.edges.append(edge)
            else:
                result.dangling_edges.append(edge)
