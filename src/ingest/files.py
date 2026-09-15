"""Правило файлов карточки: какие файлы документа индексируются и в какой роли.

Спецификация — `docs/chunk_metadata_mapping.md`, п. 3.1 (О5, [ТРЕБУЕТ ПОДТВЕРЖДЕНИЯ]); параметры —
`ingest.files` в `configs/app.yaml`. Порядок для файлов одной карточки:

1. неразрешённый формат, имя с запрещённым началом («Для печати…») или служебная категория
   («Подписи ЭП», «Получено из Диадока») → пропуск с причиной;
2. основной файл (role=main, один на документ) — первый подходящий кандидат по списку конфига:
   docx «Документ» → docx для ЭДО → pdf подписанный/«Документ» → любой docx → любой pdf;
3. остальные: pdf в категориях основного текста при docx-оригинале — дубликат, пропуск;
   «Приложение» и прочие категории → appendix; «Дополнительные сведения» → supplement только у типов
   карточек из конфига (приказы), иначе пропуск.

Между карточками файл с одинаковым sha256 индексируется один раз: владелец — экземпляр с лучшей
ролью (main > appendix > supplement), при равенстве — первый по порядку манифеста; остальные
пропускаются, а владелец получает список карточек `also_in` (для метаданных чанка).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Literal
from uuid import UUID

from common.config import FileRulesSettings, IngestSettings, MainFileCandidate
from ingest.corpus import Corpus, CorpusDocument, CorpusFile

FileRole = Literal["main", "appendix", "supplement"]
ROLE_PRIORITY: dict[FileRole, int] = {"main": 0, "appendix": 1, "supplement": 2}


@dataclass(frozen=True)
class FilePlan:
    """Решение по файлу: роль в индексе или причина пропуска."""

    file: CorpusFile
    role: FileRole | None
    skip_reason: str | None = None
    also_in: tuple[UUID, ...] = ()

    @property
    def indexed(self) -> bool:
        return self.role is not None


def _skip(file: CorpusFile, reason: str) -> FilePlan:
    return FilePlan(file=file, role=None, skip_reason=reason)


def _matches(file: CorpusFile, candidate: MainFileCandidate) -> bool:
    if file.extension != candidate.extension.lower():
        return False
    return candidate.any_category or file.category in candidate.categories


def _forbidden_prefix(file: CorpusFile, rules: FileRulesSettings) -> str | None:
    name = file.name.casefold()
    return next((p for p in rules.never_index_name_prefixes if name.startswith(p.casefold())), None)


def plan_document_files(
    document: CorpusDocument, rules: FileRulesSettings, extensions: Sequence[str]
) -> list[FilePlan]:
    """Роли файлов одной карточки в порядке манифеста (без учёта дублей между карточками)."""
    allowed = {ext.lower() for ext in extensions}
    plans: dict[UUID, FilePlan] = {}
    eligible: list[CorpusFile] = []
    for file in document.files:
        prefix = _forbidden_prefix(file, rules)
        if file.extension not in allowed:
            plans[file.row_id] = _skip(file, f"формат {file.extension} не входит в ingest.extensions")
        elif prefix is not None:
            plans[file.row_id] = _skip(file, f"копия для печати (имя начинается с «{prefix}»)")
        elif file.category in rules.never_index_categories:
            plans[file.row_id] = _skip(file, f"категория «{file.category}» не индексируется")
        else:
            eligible.append(file)

    main: CorpusFile | None = None
    for candidate in rules.main_candidates:
        main = next((file for file in eligible if _matches(file, candidate)), None)
        if main is not None:
            break

    supplement_types = ", ".join(rules.supplement_card_types) or "—"
    for file in eligible:
        if file is main:
            plans[file.row_id] = FilePlan(file=file, role="main")
        elif file.category in rules.main_text_categories:
            if (
                rules.skip_pdf_duplicate_of_docx_main
                and file.extension == "pdf"
                and main is not None
                and main.extension == "docx"
            ):
                plans[file.row_id] = _skip(
                    file, f"pdf-копия основного текста при docx-оригинале «{main.name}»"
                )
            else:
                plans[file.row_id] = FilePlan(file=file, role="appendix")
        elif file.category in rules.appendix_categories:
            plans[file.row_id] = FilePlan(file=file, role="appendix")
        elif file.category in rules.supplement_categories:
            if document.card.type_name in rules.supplement_card_types:
                plans[file.row_id] = FilePlan(file=file, role="supplement")
            else:
                plans[file.row_id] = _skip(
                    file, f"«{file.category}» индексируются только у типов {supplement_types}"
                )
        else:
            plans[file.row_id] = FilePlan(file=file, role="appendix")
    return [plans[file.row_id] for file in document.files]


def plan_corpus_files(corpus: Corpus, settings: IngestSettings) -> dict[UUID, list[FilePlan]]:
    """Роли файлов всех документов корпуса с разрешением дублей по sha256 между карточками."""
    plans = {
        document.card_id: plan_document_files(document, settings.files, settings.extensions)
        for document in corpus.documents
    }
    if not settings.files.skip_cross_card_duplicates:
        return plans

    occurrences: dict[str, list[tuple[UUID, int]]] = defaultdict(list)
    for card_id, card_plans in plans.items():
        for index, plan in enumerate(card_plans):
            if plan.indexed:
                occurrences[plan.file.sha256].append((card_id, index))
    for items in occurrences.values():
        if len(items) < 2:
            continue
        ranked = sorted(items, key=lambda item: ROLE_PRIORITY[plans[item[0]][item[1]].role or "supplement"])
        owner_card, owner_index = ranked[0]
        others = [item for item in items if item != (owner_card, owner_index)]
        owner = plans[owner_card][owner_index]
        plans[owner_card][owner_index] = replace(owner, also_in=tuple(card for card, _ in others))
        for card_id, index in others:
            plans[card_id][index] = _skip(
                plans[card_id][index].file, f"дубль по sha256 файла «{owner.file.name}» карточки {owner_card}"
            )
    return plans
