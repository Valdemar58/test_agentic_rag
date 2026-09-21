"""Перечень приказов из представления Тессы (запрос заказчика 2026-09-18, вне ТЗ).

В API Тессы нет запроса «дай все карточки вида X»: списки отдаются только представлениями
(`views/get-data`), и какое представление показывает приказы — знает рабочий контур заказчика.
Поэтому алиас, фильтры и имена колонок берутся из конфига (секция `orders`), а найти их помогает
команда `tessa-export views`.

Два предохранителя против особенностей представлений: строки дедуплицируются по ID документа
(представление без пагинации отдаёт весь набор на каждой странице), а обход прекращается по
`max_pages`/`max_documents`. Отбор по состоянию маршрута применяется здесь только предварительно —
окончательное решение принимается по полю карточки `DocumentCommonInfo.StateID` правилами
исключения (`state_rules`), потому что колонки состояния в представлении может не быть.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

from tessa_export.config import ConfigError, ExcludeRule, OrdersSettings
from tessa_export.models import ViewSource

logger = logging.getLogger(__name__)

STATE_SECTION_FIELD = "DocumentCommonInfo.StateID"
# Названия состояний маршрута, которые встречаются в правилах отбора (таблица состояний — в конфиге)
STATE_NAMES = {5: "Отменён", 6: "Зарегистрировано", 8: "Подписан", 17: "Аннулирован"}

NO_VIEW_ALIAS = (
    "не задан orders.view_alias: команда не знает, из какого представления брать приказы.\n"
    "Посмотрите доступные представления командой `tessa-export views --config config.yaml`, "
    "выберите нужное и впишите его алиас (а при необходимости фильтры) в секцию orders конфига"
)


class OrdersError(ConfigError):
    """Перечень приказов получить не удалось: настройки представления не совпали с ответом."""


@dataclass(frozen=True)
class OrderRow:
    """Строка представления, отобранная к выгрузке."""

    card_id: UUID
    state_id: int | None = None
    label: str = ""


@dataclass
class OrdersListing:
    rows: list[OrderRow] = field(default_factory=list)
    pages: int = 0
    received: int = 0
    unique: int = 0
    duplicates: int = 0
    without_id: int = 0
    excluded_by_state: int = 0
    excluded_by_match: int = 0
    truncated: bool = False
    reported_total: int | None = None

    @property
    def card_ids(self) -> list[UUID]:
        return [row.card_id for row in self.rows]

    @property
    def incomplete(self) -> bool:
        """Тесса сообщила больше строк, чем мы прочитали: список неполный."""
        return self.reported_total is not None and self.unique < self.reported_total

    def summary_lines(self) -> list[str]:
        total = f" из {self.reported_total} по данным Тессы" if self.reported_total else ""
        lines = [
            f"Представление: прочитано строк {self.unique}{total} за {self.pages} стр., "
            f"к выгрузке отобрано {len(self.rows)}"
        ]
        details = []
        if self.excluded_by_state:
            details.append(f"по состоянию {self.excluded_by_state}")
        if self.excluded_by_match:
            details.append(f"по фильтру строк {self.excluded_by_match}")
        if self.duplicates:
            details.append(f"повторов {self.duplicates}")
        if self.without_id:
            details.append(f"без ID документа {self.without_id}")
        if details:
            lines.append("  отсеяно: " + ", ".join(details))
        if self.incomplete:
            lines.append(
                f"  ВНИМАНИЕ: представление сообщает {self.reported_total} строк, прочитано "
                f"{self.unique} — список неполный. Поднимите orders.max_pages/max_documents"
            )
        elif self.truncated:
            lines.append("  ВНИМАНИЕ: сработал предохранитель max_documents/max_pages, список неполный")
        return lines


def _to_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _to_uuid(value: Any) -> UUID | None:
    if value is None:
        return None
    try:
        return UUID(str(value))
    except ValueError:
        return None


def _require_columns(columns: list[str], settings: OrdersSettings) -> None:
    """Имена колонок из конфига сверяются с ответом представления до разбора строк."""
    known = set(columns)
    wanted = {settings.id_column}
    if settings.state_column:
        wanted.add(settings.state_column)
    wanted.update(settings.match)
    missing = sorted(name for name in wanted if name not in known)
    if missing:
        raise OrdersError(
            f"в представлении «{settings.view_alias}» нет колонок {', '.join(missing)}; "
            f"есть: {', '.join(columns) or '(ни одной)'}. Поправьте orders.id_column / "
            "orders.state_column / orders.match в конфиге"
        )


def _matches(row: dict[str, Any], patterns: dict[str, list[str]]) -> bool:
    """Все заданные колонки должны совпасть хотя бы с одним своим регэкспом."""
    for column, expressions in patterns.items():
        text = str(row.get(column) or "")
        if not any(re.search(expression, text, re.IGNORECASE) for expression in expressions):
            return False
    return True


def _label(row: dict[str, Any], columns: list[str], settings: OrdersSettings) -> str:
    """Короткая подпись строки для лога: первые текстовые колонки, кроме служебных."""
    skip = {settings.id_column, settings.state_column}
    parts = [str(row[name]) for name in columns[:4] if name not in skip and row.get(name)]
    return " · ".join(parts)[:120]


def collect_orders(source: ViewSource, settings: OrdersSettings) -> OrdersListing:
    """Читает представление постранично и отбирает строки к выгрузке."""
    if not settings.view_alias:
        raise OrdersError(NO_VIEW_ALIAS)
    listing = OrdersListing()
    seen: set[UUID] = set()
    sorting = (settings.sort_column, settings.sort_descending) if settings.sort_column else None
    for page in range(1, settings.max_pages + 1):
        # PageOffset у Тессы — номер первой строки окна, а не номер страницы: с page_offset=2
        # возвращаются строки 2…201, то есть окно сдвигается на одну строку (проверено на живом
        # представлении заказчика 2026-09-21). Смещение считается в строках.
        result = source.view_page(
            settings.view_alias,
            settings.parameters,
            subset=settings.subset,
            sorting=sorting,
            page_offset=(page - 1) * settings.page_limit + 1,
            page_limit=settings.page_limit,
            with_count=page == 1,
        )
        if page == 1:
            _require_columns(result.columns, settings)
            listing.reported_total = result.row_count or None
        if not result.rows:
            break
        listing.pages = page
        listing.received += len(result.rows)
        fresh = 0
        for row in result.rows:
            card_id = _to_uuid(row.get(settings.id_column))
            if card_id is None:
                listing.without_id += 1
                continue
            if card_id in seen:
                listing.duplicates += 1
                continue
            seen.add(card_id)
            fresh += 1
            if settings.match and not _matches(row, settings.match):
                listing.excluded_by_match += 1
                continue
            state_id = _to_int(row.get(settings.state_column)) if settings.state_column else None
            if state_id is not None and not _state_allowed(state_id, settings):
                listing.excluded_by_state += 1
                continue
            listing.rows.append(
                OrderRow(card_id=card_id, state_id=state_id, label=_label(row, result.columns, settings))
            )
        listing.unique = len(seen)
        logger.info(
            "Представление «%s», страница %d: строк %d, новых %d, прочитано всего %d, отобрано %d",
            settings.view_alias,
            page,
            len(result.rows),
            fresh,
            listing.unique,
            len(listing.rows),
        )
        if len(listing.rows) >= settings.max_documents:
            listing.rows = listing.rows[: settings.max_documents]
            listing.truncated = True
            break
        # представление без пагинации отдаёт тот же набор на каждой странице — повтор значит конец
        if fresh == 0 or len(result.rows) < settings.page_limit:
            break
    else:
        listing.truncated = True
    listing.unique = len(seen)
    return listing


def _state_allowed(state_id: int, settings: OrdersSettings) -> bool:
    if settings.include_state_ids and state_id not in settings.include_state_ids:
        return False
    return state_id not in settings.exclude_state_ids


def _state_names(states: list[int]) -> str:
    return ", ".join(
        f"{state} {STATE_NAMES[state]}" if state in STATE_NAMES else str(state) for state in states
    )


def state_rules(settings: OrdersSettings) -> list[ExcludeRule]:
    """Правила отбора по состоянию маршрута: проверяются уже на полученной карточке.

    Белый список `include_state_ids` даёт правило «исключить всё, кроме перечисленного» — карточка
    без поля StateID тоже исключается, потому что её состояние не подтверждено.
    """
    rules: list[ExcludeRule] = []
    if settings.include_state_ids:
        allowed = _state_names(settings.include_state_ids)
        rules.append(
            ExcludeRule(
                reason=f"состояние маршрута не из списка выгрузки: {allowed}",
                field=STATE_SECTION_FIELD,
                values=[str(state) for state in settings.include_state_ids],
                values_mode="none_of",
                applies_to_seed=True,
            )
        )
    if settings.exclude_state_ids:
        rules.append(
            ExcludeRule(
                reason=f"состояние маршрута: {_state_names(settings.exclude_state_ids)}",
                field=STATE_SECTION_FIELD,
                values=[str(state) for state in settings.exclude_state_ids],
                applies_to_seed=True,
            )
        )
    return rules
