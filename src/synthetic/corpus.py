"""Синтетический корпус в формате экспорта Тессы (§9 ТЗ): карточки по схеме CardData, файлы,
`links_graph.json`, `manifest.json`, отчёт валидации — тем же кодом, что и реальный экспорт.

Сценарий повторяет структуру реального корпуса заказчика: типы карточек и категории Тессы,
категории файлов («Документ», «Приложение», «Для печати_…», «Файлы для отправки по ЭДО»,
«Дополнительные сведения»), справочник статусов и состояний, типы связей («в отмену»/«отменено»,
«дополнение»/«дополнено», «Приказ»/«Документ-основание», «Основной договор»/«Доп. соглашение»,
«запрос»/«ответ»). Есть пара противоречащих приказов (отменённый и отменяющий), приказ с
положением и разделом «Термины и определения», инструкция с таблицей, договор с приложением и
доп. соглашением, служебные записки, письмо-скан без текстового слоя и проект приказа.
Все данные синтетические.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

from synthetic import texts
from synthetic.documents import (
    DocumentText,
    docx_bytes,
    pdf_bytes,
    scan_image_bytes,
    scan_pdf_bytes,
    xlsx_bytes,
)
from tessa_export.config import CANCELLED_STATUS_ID, ExportConfig
from tessa_export.fake import FakeGateway, link, make_file, make_snapshot, stable_uuid
from tessa_export.models import FileInfo, LinkInfo
from tessa_export.runner import RunSummary, run_export
from tessa_export.sample_files import docx_with_misdeclared_altchunk_bytes
from tessa_export.storage import MANIFEST_NAME, export_dir

logger = logging.getLogger(__name__)

# Реальные значения справочника статусов заказчика (по экспорту 2026-09-14)
ACTIVE_ORDER_STATUS = UUID("1229fb49-3d80-47c9-9816-3c5467954679")  # «Действующий»
ACTIVE_CONTRACT_STATUS = UUID("f9e512aa-6ae3-4d77-855f-60d4e11e6de3")  # «Действует»
SOURCE = "синтетический корпус (scripts/make_synthetic_corpus.py)"
ARCHIVE_NAME = "synthetic_corpus.zip"
README_NAME = "README_SYNTHETIC.md"
README_TEXT = (
    "# ДАННЫЕ СИНТЕТИЧЕСКИЕ\n\n"
    "Корпус сгенерирован `scripts/make_synthetic_corpus.py` для разработки до получения реального "
    "экспорта. Организация, документы, номера, ФИО и связи вымышлены. Метрики M1–M8 на этих "
    "данных не считаются (правило проекта).\n"
)

# Категории файлов Тессы
DOC = "Документ"
APPENDIX = "Приложение"
EXTRA = "Дополнительные сведения"
EDO = "Файлы для отправки по ЭДО"


def card_id(name: str) -> UUID:
    return stable_uuid("synthetic", name)


ORDER_144, ORDER_109, ORDER_173, ORDER_150, ORDER_151, ORDER_DRAFT_160 = (
    card_id(name)
    for name in ("order_144", "order_109", "order_173", "order_150", "order_151", "order_draft_160")
)
CONTRACT_D1, CONTRACT_D1_1 = card_id("contract_d1"), card_id("contract_d1_1")
MEMO_9, MEMO_12 = card_id("memo_9"), card_id("memo_12")
LETTER_IN_33, LETTER_OUT_40 = card_id("letter_in_33"), card_id("letter_out_40")
JUSTIFICATION_5 = card_id("justification_5")

SEED: list[UUID] = [
    ORDER_144,
    ORDER_109,
    ORDER_150,
    CONTRACT_D1,
    MEMO_12,
    LETTER_IN_33,
    JUSTIFICATION_5,
    ORDER_DRAFT_160,
]


def _date(year: int, month: int, day: int) -> datetime:
    return datetime(year, month, day, tzinfo=UTC)


def _incoming(*ids: UUID) -> list[LinkInfo]:
    return [link(item, ref_type_name=None, ref_type_reverse_name=None) for item in ids]


@dataclass(frozen=True)
class CardFiles:
    """Файлы карточки и их содержимое."""

    files: list[FileInfo]
    contents: dict[str, bytes]


def _files(card: UUID, *items: tuple[str, str | None, bytes | None]) -> CardFiles:
    files: list[FileInfo] = []
    contents: dict[str, bytes] = {}
    for name, category, content in items:
        if content is None:
            files.append(make_file(card, name, category=category, is_virtual=True, size=-1))
            continue
        files.append(make_file(card, name, category=category, size=len(content)))
        contents[name] = content
    return CardFiles(files, contents)


def _order_files(
    card: UUID, number: str, date_text: str, text: DocumentText, *extra: tuple[str, str | None, bytes | None]
) -> CardFiles:
    """Типовой набор файлов приказа: оригинал docx, копия «Для печати», регистрационный скан pdf."""
    return _files(
        card,
        (f"ДокШаблон Приказ №{number}.docx", DOC, docx_bytes(text)),
        (f"Для печати_Приказ №{number}.docx", DOC, docx_with_misdeclared_altchunk_bytes(text.lines()[:6])),
        (f"{number} от {date_text} {text.title}.pdf", DOC, scan_pdf_bytes(text)),
        ("Лист согласования.html", None, None),
        *extra,
    )


def build_gateway() -> FakeGateway:
    """Все карточки сценария в фейковом шлюзе."""
    gateway = FakeGateway()

    order_144 = _order_files(
        ORDER_144,
        "144",
        "15.01.2026",
        texts.ORDER_144,
        ("Приложение - Положение об охране труда.docx", APPENDIX, docx_bytes(texts.REGULATION_OT)),
    )
    gateway.add(
        make_snapshot(
            ORDER_144,
            number="144",
            doc_date=_date(2026, 1, 15),
            subject=texts.ORDER_144.title,
            status_id=ACTIVE_ORDER_STATUS,
            status_name="Действующий",
            department="Отдел охраны труда",
            outgoing=[
                link(
                    MEMO_9,
                    ref_type_name="Документ-основание",
                    ref_type_reverse_name="Приказ",
                    doc_type_name="Служебная записка",
                )
            ],
            incoming=_incoming(ORDER_151, MEMO_9),
            files=order_144.files,
        ),
        order_144.contents,
    )

    order_109 = _order_files(ORDER_109, "109", "10.02.2025", texts.ORDER_109)
    gateway.add(
        make_snapshot(
            ORDER_109,
            number="109",
            doc_date=_date(2025, 2, 10),
            subject=texts.ORDER_109.title,
            status_id=CANCELLED_STATUS_ID,
            status_name="Отмененный",
            department="Канцелярия",
            incoming=_incoming(ORDER_173),
            files=order_109.files,
            extra_fields={"Comment": "Отменен приказом от 27.08.2026 № 173."},
        ),
        order_109.contents,
    )

    order_173 = _order_files(ORDER_173, "173", "27.08.2026", texts.ORDER_173)
    gateway.add(
        make_snapshot(
            ORDER_173,
            number="173",
            doc_date=_date(2026, 8, 27),
            subject=texts.ORDER_173.title,
            status_id=ACTIVE_ORDER_STATUS,
            status_name="Действующий",
            department="Канцелярия",
            outgoing=[link(ORDER_109, ref_type_name="в отмену", ref_type_reverse_name="отменено")],
            files=order_173.files,
        ),
        order_173.contents,
    )

    order_150 = _order_files(
        ORDER_150,
        "150",
        "20.05.2024",
        texts.ORDER_150,
        (
            "Приложение - Инструкция по действиям при пожаре.docx",
            APPENDIX,
            docx_bytes(texts.INSTRUCTION_FIRE),
        ),
        ("Скан приложения.jpg", EXTRA, scan_image_bytes(texts.INSTRUCTION_FIRE)),
    )
    gateway.add(
        make_snapshot(
            ORDER_150,
            number="150",
            doc_date=_date(2024, 5, 20),
            subject=texts.ORDER_150.title,
            status_id=ACTIVE_ORDER_STATUS,
            status_name="Действующий",
            department="Отдел охраны труда",
            files=order_150.files,
        ),
        order_150.contents,
    )

    order_151 = _order_files(ORDER_151, "151", "03.03.2026", texts.ORDER_151)
    gateway.add(
        make_snapshot(
            ORDER_151,
            number="151",
            doc_date=_date(2026, 3, 3),
            subject=texts.ORDER_151.title,
            status_id=ACTIVE_ORDER_STATUS,
            status_name="Действующий",
            department="Отдел охраны труда",
            outgoing=[link(ORDER_144, ref_type_name="дополнение", ref_type_reverse_name="дополнено")],
            files=order_151.files,
        ),
        order_151.contents,
    )

    draft = _files(ORDER_DRAFT_160, ("ДокШаблон Приказ проект.docx", DOC, docx_bytes(texts.ORDER_DRAFT_160)))
    gateway.add(
        make_snapshot(
            ORDER_DRAFT_160,
            number="",
            doc_date=_date(2026, 9, 1),
            subject=texts.ORDER_DRAFT_160.title,
            status_id=None,
            status_name=None,
            state_id=1,
            state_name="$KrStates_Doc_Active",
            department="Отдел закупок",
            files=draft.files,
            extra_fields={"SecondaryFullNumber": "П-160"},
        ),
        draft.contents,
    )

    contract = _files(
        CONTRACT_D1,
        ("Договор поставки Д-1.pdf", DOC, pdf_bytes(texts.CONTRACT_D1)),
        ("Договор поставки Д-1.docx", EDO, docx_bytes(texts.CONTRACT_D1)),
        (
            "Спецификация к договору Д-1.xlsx",
            APPENDIX,
            xlsx_bytes("Спецификация", texts.CONTRACT_SPECIFICATION),
        ),
    )
    gateway.add(
        make_snapshot(
            CONTRACT_D1,
            type_name="ContractMKC",
            type_caption="Договорной документ",
            number="Д-1",
            doc_date=_date(2023, 11, 1),
            subject=texts.CONTRACT_D1.title,
            status_id=ACTIVE_CONTRACT_STATUS,
            status_name="Действует",
            state_id=8,
            state_name="$KrStates_Doc_Signed",
            department="Отдел закупок",
            incoming=_incoming(CONTRACT_D1_1, JUSTIFICATION_5),
            files=contract.files,
            extra_fields={"ValidityPeriod": "до 31.12.2026 с автоматической пролонгацией"},
        ),
        contract.contents,
    )

    supplement = _files(CONTRACT_D1_1, ("Доп. соглашение Д-1-1.pdf", DOC, pdf_bytes(texts.CONTRACT_D1_1)))
    gateway.add(
        make_snapshot(
            CONTRACT_D1_1,
            type_name="ContractMKC",
            type_caption="Договорной документ",
            number="Д-1/1",
            doc_date=_date(2024, 6, 15),
            subject=texts.CONTRACT_D1_1.title,
            status_id=ACTIVE_CONTRACT_STATUS,
            status_name="Действует",
            state_id=8,
            state_name="$KrStates_Doc_Signed",
            department="Отдел закупок",
            outgoing=[
                link(
                    CONTRACT_D1,
                    ref_type_name="Основной договор",
                    ref_type_reverse_name="Доп. соглашение",
                    doc_type_name="Договорной документ",
                )
            ],
            files=supplement.files,
            extra_fields={"ValidityPeriod": "до 31.12.2026"},
        ),
        supplement.contents,
    )

    memo_9 = _files(MEMO_9, ("СЗ-9 О выделении средств защиты.docx", DOC, docx_bytes(texts.MEMO_9)))
    gateway.add(
        make_snapshot(
            MEMO_9,
            type_name="InhouseDocumentMKC",
            type_caption="Служебная записка",
            number="СЗ-9",
            doc_date=_date(2025, 12, 20),
            subject=texts.MEMO_9.title,
            status_id=None,
            status_name=None,
            state_id=6,
            department="Отдел охраны труда",
            outgoing=[link(ORDER_144, ref_type_name="Приказ", ref_type_reverse_name="Документ-основание")],
            incoming=_incoming(ORDER_144),
            files=memo_9.files,
        ),
        memo_9.contents,
    )

    memo_12 = _files(MEMO_12, ("СЗ-12 Об итогах инструктажей.docx", DOC, docx_bytes(texts.MEMO_12)))
    gateway.add(
        make_snapshot(
            MEMO_12,
            type_name="InhouseDocumentMKC",
            type_caption="Служебная записка",
            number="СЗ-12",
            doc_date=_date(2026, 2, 5),
            subject=texts.MEMO_12.title,
            status_id=None,
            status_name=None,
            state_id=12,
            state_name="Исполнено",
            department="Отдел охраны труда",
            files=memo_12.files,
        ),
        memo_12.contents,
    )

    letter_in = _files(LETTER_IN_33, ("Вх-33 скан письма.pdf", DOC, scan_pdf_bytes(texts.LETTER_IN_33)))
    gateway.add(
        make_snapshot(
            LETTER_IN_33,
            type_name="IncomingMKC",
            type_caption="Входящее письмо",
            number="Вх-33",
            doc_date=_date(2026, 4, 10),
            subject=texts.LETTER_IN_33.title,
            status_id=None,
            status_name=None,
            state_id=11,
            state_name="На исполнении",
            department="Канцелярия",
            incoming=_incoming(LETTER_OUT_40),
            files=letter_in.files,
        ),
        letter_in.contents,
    )

    letter_out = _files(LETTER_OUT_40, ("Исх-40 ответ.docx", DOC, docx_bytes(texts.LETTER_OUT_40)))
    gateway.add(
        make_snapshot(
            LETTER_OUT_40,
            type_name="OutgoingMKC",
            type_caption="Исходящее письмо",
            number="Исх-40",
            doc_date=_date(2026, 4, 25),
            subject=texts.LETTER_OUT_40.title,
            status_id=None,
            status_name=None,
            state_id=8,
            state_name="$KrStates_Doc_Signed",
            department="Канцелярия",
            outgoing=[
                link(
                    LETTER_IN_33,
                    ref_type_name="запрос",
                    ref_type_reverse_name="ответ",
                    doc_type_name="Входящее письмо",
                )
            ],
            files=letter_out.files,
        ),
        letter_out.contents,
    )

    justification = _files(
        JUSTIFICATION_5,
        ("Справка-обоснование закупки СИЗ.xlsx", DOC, xlsx_bytes("Обоснование", texts.JUSTIFICATION_TABLE)),
        ("Подпись.sig", None, b"synthetic-signature"),
    )
    gateway.add(
        make_snapshot(
            JUSTIFICATION_5,
            type_name="BackgroundJustificationMKC",
            type_caption="Справка-обоснование",
            number="СО-5",
            doc_date=_date(2023, 10, 15),
            subject="Обоснование закупки средств индивидуальной защиты на 2024 год",
            status_id=None,
            status_name=None,
            state_id=6,
            department="Отдел закупок",
            outgoing=[
                link(
                    CONTRACT_D1,
                    ref_type_name="Договор",
                    ref_type_reverse_name="Справка-обоснование",
                    doc_type_name="Договорной документ",
                )
            ],
            files=justification.files,
        ),
        justification.contents,
    )
    return gateway


def export_config(output_dir: Path) -> ExportConfig:
    return ExportConfig.model_validate(
        {
            "tessa": {"base_url": "http://synthetic.invalid"},
            "external": {"tessa_sdk_path": "/nonexistent", "card_service_path": "/nonexistent"},
            "output_dir": str(output_dir),
            "archive_name": ARCHIVE_NAME,
            "traversal": {"max_depth": 2, "max_docs": 100},
        }
    )


class RealCorpusPresentError(Exception):
    """В каталоге лежит корпус не с пометкой «синтетический» — перезапись запрещена без --force."""


def existing_manifest_is_real(output_dir: Path) -> bool:
    manifest_path = export_dir(output_dir) / MANIFEST_NAME
    if not manifest_path.is_file():
        return False
    return '"synthetic": true' not in manifest_path.read_text(encoding="utf-8")


def generate_corpus(output_dir: Path, *, force: bool = False) -> RunSummary:
    """Собирает корпус в `output_dir/export/` и архив; отказывается затирать реальный экспорт."""
    if not force and existing_manifest_is_real(output_dir):
        raise RealCorpusPresentError(
            f"в {output_dir} уже лежит корпус без пометки synthetic (похоже, реальный экспорт); "
            "укажите другой каталог или --force"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    gateway = build_gateway()
    summary = run_export(export_config(output_dir), SEED, gateway, synthetic=True, source=SOURCE)
    (output_dir / README_NAME).write_text(README_TEXT, encoding="utf-8")
    logger.info(
        "Синтетический корпус: %d документов, %d файлов, %s",
        summary.documents,
        summary.files_downloaded,
        summary.overall,
    )
    return summary
