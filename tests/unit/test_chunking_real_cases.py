"""Случаи из ручной проверки реального корпуса (AC-3.2): даты и суммы не номера, «голый» номер,
заголовки-пункты договоров, длинные крошки, широкие строки таблиц."""

from __future__ import annotations

from docling_core.types.doc.document import DoclingDocument
from docling_core.types.doc.items.table.table_data import TableCell, TableData
from docling_core.types.doc.labels import DocItemLabel

from common.config import DEFAULT_CONFIG_PATH, load_app_config
from ingest.chunking import StructuralChunker, bare_number, number_of
from ingest.tokens import WordTokenCounter

SETTINGS = load_app_config(DEFAULT_CONFIG_PATH).ingest.chunking
ROOT = ("Документ",)
SHA = "f" * 64


def chunker() -> StructuralChunker:
    return StructuralChunker(SETTINGS, WordTokenCounter())


def test_number_detection_ignores_dates_sums_and_years() -> None:
    assert number_of("1.1. Утвердить положение.") == "1.1"
    assert number_of("1.1 Утвердить положение.") == "1.1"
    assert number_of("3. Контроль оставляю за собой.") == "3"
    assert number_of("3) Контроль оставляю за собой.") == "3"
    assert number_of("26 702 руб.") is None
    assert number_of("04.09.2026") is None
    assert number_of("2026 г. был удачным") is None
    assert number_of("1.6.") is None and bare_number("1.6.") == "1.6" and bare_number("3)") == "3"
    assert bare_number("26") is None and bare_number("04.09.2026") is None


def test_bare_number_paragraph_is_attached_to_the_next_paragraph() -> None:
    doc = DoclingDocument(name="договор")
    doc.add_heading("Договор", level=1)
    doc.add_heading("1. Термины и определения", level=2)
    doc.add_text(label=DocItemLabel.TEXT, text="1.6.")
    doc.add_text(label=DocItemLabel.TEXT, text="Мастер-каталог — приложение к Договору с перечнем товаров.")
    doc.add_text(label=DocItemLabel.TEXT, text="26 702 руб.")
    chunks = chunker().chunk(doc, ROOT, file_sha256=SHA)
    clause = next(chunk for chunk in chunks if chunk.clause == "1.6")
    assert clause.body.startswith("1.6. Мастер-каталог")
    assert not any(chunk.body.strip() == "1.6." for chunk in chunks)
    assert all(chunk.clause != "26" for chunk in chunks)


def test_clause_headings_and_long_headings_in_breadcrumbs() -> None:
    doc = DoclingDocument(name="договор")
    doc.add_heading("Договор поставки", level=1)
    doc.add_heading("2 СРОКИ И УСЛОВИЯ ПОСТАВКИ", level=2)
    doc.add_heading(
        "2.6 Поставщик гарантирует качество и комплектность поставляемого Товара в течение всего срока",
        level=3,
    )
    doc.add_text(label=DocItemLabel.TEXT, text="Товар должен поставляться в полной комплектации.")
    doc.add_heading("2.7 Гарантийный срок", level=3)
    doc.add_text(label=DocItemLabel.TEXT, text="Гарантийный срок составляет 12 месяцев.")
    long_heading = (
        "В случае поставки товара ненадлежащего качества покупатель вправе по своему выбору потребовать"
    )
    doc.add_heading(f"{long_heading} замены", level=2)
    doc.add_text(label=DocItemLabel.TEXT, text="соразмерно уменьшить стоимость.")
    chunks = chunker().chunk(doc, ROOT, file_sha256=SHA)
    first = chunks[0]
    assert first.breadcrumbs == (ROOT[0], "Раздел 2 СРОКИ И УСЛОВИЯ ПОСТАВКИ", "п. 2.6")
    assert chunks[1].breadcrumbs == (ROOT[0], "Раздел 2 СРОКИ И УСЛОВИЯ ПОСТАВКИ", "п. 2.7 Гарантийный срок")
    long_crumb = chunks[2].breadcrumbs[1]
    assert long_crumb.endswith("…") and len(long_crumb.split()) == SETTINGS.breadcrumb_max_words
    assert all(
        len(crumb.split()) <= SETTINGS.breadcrumb_max_words for chunk in chunks for crumb in chunk.breadcrumbs
    )


def _wide_table(rows: int, cols: int, words_per_cell: int) -> TableData:
    cells = [
        TableCell(
            text=(f"колонка{c}" if r == 0 else " ".join(f"ячейка{r}_{c}_{w}" for w in range(words_per_cell))),
            start_row_offset_idx=r,
            end_row_offset_idx=r + 1,
            start_col_offset_idx=c,
            end_col_offset_idx=c + 1,
            column_header=r == 0,
        )
        for r in range(rows)
        for c in range(cols)
    ]
    return TableData(table_cells=cells, num_rows=rows, num_cols=cols)


def test_wide_table_rows_become_column_value_records_within_limit() -> None:
    doc = DoclingDocument(name="каталог")
    doc.add_heading("Приложение 1 Мастер-каталог", level=1)
    doc.add_table(data=_wide_table(rows=4, cols=40, words_per_cell=20))  # строка ≈ 800+ слов
    chunks = chunker().chunk(doc, ROOT, file_sha256=SHA)
    tables = [chunk for chunk in chunks if chunk.kind == "table"]
    assert tables and all(chunk.tokens <= SETTINGS.max_tokens for chunk in tables)
    records = [chunk for chunk in tables if "колонка0: ячейка1_0_0" in chunk.body]
    assert records, [chunk.body[:80] for chunk in tables]
    assert all(": " in line for chunk in records for line in chunk.body.splitlines())
    # все ячейки всех строк попали в записи
    body = "\n".join(chunk.body for chunk in tables)
    assert all(f"ячейка{r}_{c}_0" in body for r in range(1, 4) for c in (0, 39))


def test_narrow_table_still_split_by_rows_with_header() -> None:
    doc = DoclingDocument(name="таблица")
    doc.add_heading("Спецификация", level=1)
    doc.add_table(data=_wide_table(rows=120, cols=3, words_per_cell=6))
    chunks = chunker().chunk(doc, ROOT, file_sha256=SHA)
    tables = [chunk for chunk in chunks if chunk.kind == "table"]
    assert len(tables) >= 3 and all(chunk.tokens <= SETTINGS.max_tokens for chunk in tables)
    assert all(chunk.body.splitlines()[0].startswith("| колонка0") for chunk in tables)
