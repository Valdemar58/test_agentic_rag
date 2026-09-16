"""Чанкер (4.3): структура и хлебные крошки, пункты из маркеров и текста, таблицы, фоллбэк, перекрытие."""

from __future__ import annotations

from docling_core.types.doc.base import BoundingBox
from docling_core.types.doc.common.reference import ProvenanceItem
from docling_core.types.doc.document import DoclingDocument
from docling_core.types.doc.items.table.table_data import TableCell, TableData
from docling_core.types.doc.labels import DocItemLabel

from common.config import DEFAULT_CONFIG_PATH, load_app_config
from ingest.chunking import Chunk, StructuralChunker
from ingest.parents import build_chunk_set
from ingest.tokens import WordTokenCounter

SETTINGS = load_app_config(DEFAULT_CONFIG_PATH).ingest.chunking
ROOT = ("Приказ №144 от 15.01.2026",)
SEP = SETTINGS.breadcrumb_separator
SHA = "a" * 64


def chunker() -> StructuralChunker:
    return StructuralChunker(SETTINGS, WordTokenCounter())


def _table(rows: list[list[str]]) -> TableData:
    cells = [
        TableCell(
            text=text,
            start_row_offset_idx=r,
            end_row_offset_idx=r + 1,
            start_col_offset_idx=c,
            end_col_offset_idx=c + 1,
            column_header=r == 0,
        )
        for r, row in enumerate(rows)
        for c, text in enumerate(row)
    ]
    return TableData(table_cells=cells, num_rows=len(rows), num_cols=len(rows[0]))


def _prov(page: int) -> ProvenanceItem:
    return ProvenanceItem(page_no=page, bbox=BoundingBox(l=0, t=0, r=10, b=10), charspan=(0, 0))


def order_docx_like() -> DoclingDocument:
    """Как Docling читает docx приказа: h1 — тема, h2 — разделы, пункты — абзацы с номером в тексте."""
    doc = DoclingDocument(name="приказ")
    doc.add_heading("Об утверждении Положения об охране труда", level=1)
    doc.add_text(label=DocItemLabel.TEXT, text="В целях обеспечения безопасных условий труда ПРИКАЗЫВАЮ:")
    doc.add_heading("1. Утверждение", level=2)
    doc.add_text(
        label=DocItemLabel.TEXT, text="1.1. Утвердить Положение об охране труда и ввести его в действие."
    )
    doc.add_text(label=DocItemLabel.TEXT, text="1.2. Признать утратившим силу прежнее Положение.")
    doc.add_heading("3. Контроль", level=2)
    doc.add_text(
        label=DocItemLabel.TEXT, text="3.1. Контроль за исполнением приказа возложить на заместителя."
    )
    doc.add_text(
        label=DocItemLabel.TEXT, text="3.2. Отдел кадров направляет копию приказа во все подразделения."
    )
    return doc


def test_structural_breadcrumbs_title_excluded_sections_and_clauses() -> None:
    chunks = chunker().chunk(order_docx_like(), ROOT, file_sha256=SHA)
    assert all(chunk.strategy == "structural" for chunk in chunks)
    by_clause = {chunk.clause: chunk for chunk in chunks}
    preamble = next(chunk for chunk in chunks if chunk.clause is None)
    assert preamble.breadcrumbs == ROOT and "Об утверждении" not in preamble.text
    assert preamble.text.startswith(f"{ROOT[0]}\nВ целях")
    clause = by_clause["3.2"]
    assert clause.breadcrumbs == (ROOT[0], "Раздел 3. Контроль", "п. 3.2")
    expected_prefix = SEP.join((ROOT[0], "Раздел 3. Контроль", "п. 3.2"))
    assert (
        clause.text == f"{expected_prefix}\n3.2. Отдел кадров направляет копию приказа во все подразделения."
    )
    assert clause.heading == "3. Контроль" and clause.section_key == "Раздел 3. Контроль"
    assert clause.kind == "text" and clause.tokens == len(clause.text.split())
    assert by_clause["1.1"].section_key == "Раздел 1. Утверждение"
    assert [chunk.ordinal for chunk in chunks] == list(range(len(chunks)))
    assert clause.item_refs == ("#/texts/7",)


def test_pdf_style_list_markers_and_numbered_section_titles() -> None:
    """Нативный pdf: разделы и пункты — ListItem с номером в marker и текстом без номера."""
    doc = DoclingDocument(name="приказ")
    doc.add_heading("О порядке отчётности", level=1)
    group = doc.add_list_group()
    doc.add_list_item("Утверждение", enumerated=True, marker="1.", parent=group)
    doc.add_list_item("Утвердить порядок отчётности.", enumerated=True, marker="1.1.", parent=group)
    doc.add_list_item("Ответственные", enumerated=True, marker="2.", parent=group)
    doc.add_list_item(
        "Назначить ответственным начальника отдела охраны труда и обеспечить контроль исполнения.",
        enumerated=True,
        marker="2.1.",
        parent=group,
    )
    chunks = chunker().chunk(doc, ROOT, file_sha256=SHA)
    clause_21 = next(chunk for chunk in chunks if chunk.clause == "2.1")
    assert clause_21.breadcrumbs == (ROOT[0], "Раздел 2. Ответственные", "п. 2.1")
    assert clause_21.body.startswith("2.1. Назначить")
    assert next(chunk for chunk in chunks if chunk.clause == "1.1").section_key == "Раздел 1. Утверждение"


def test_flat_numbered_order_without_section_titles() -> None:
    """Сквозная нумерация без заголовков: длинный «1. …» — пункт, а не раздел."""
    doc = DoclingDocument(name="приказ")
    doc.add_heading("О назначении ответственных", level=1)
    doc.add_text(
        label=DocItemLabel.TEXT, text="1. Назначить ответственным за пожарную безопасность начальника отдела."
    )
    doc.add_text(label=DocItemLabel.TEXT, text="2. Контроль оставляю за собой.")
    chunks = chunker().chunk(doc, ROOT, file_sha256=SHA)
    assert [chunk.breadcrumbs for chunk in chunks] == [(ROOT[0], "п. 1"), (ROOT[0], "п. 2")]
    assert all(chunk.section_key == "" for chunk in chunks)


def test_vlm_headers_with_hashes_get_levels_from_hashes() -> None:
    """dots.mocr: все заголовки уровня 1, уровень читается из «#»/«##» в тексте."""
    doc = DoclingDocument(name="скан")
    doc.add_heading("# Инструкция по действиям при пожаре", level=1, prov=_prov(1))
    doc.add_text(label=DocItemLabel.TEXT, text="Приложение 1 к приказу от 20.05.2024 № 150", prov=_prov(1))
    doc.add_heading("## 2. Эвакуация", level=1, prov=_prov(1))
    doc.add_text(label=DocItemLabel.TEXT, text="2.1. Покинуть помещение по ближайшему выходу.", prov=_prov(2))
    doc.add_text(label=DocItemLabel.PAGE_FOOTER, text="стр. 2", prov=_prov(2))
    root = (ROOT[0], "Приложение «Инструкция.jpg»")
    chunks = chunker().chunk(doc, root, file_sha256=SHA)
    assert chunks[0].breadcrumbs == root and chunks[0].page_no == 1
    clause = chunks[1]
    assert clause.breadcrumbs == (*root, "Раздел 2. Эвакуация", "п. 2.1") and clause.page_no == 2
    assert not any("стр. 2" in chunk.body for chunk in chunks)


def test_small_paragraphs_merge_and_long_paragraph_splits_with_overlap() -> None:
    doc = DoclingDocument(name="положение")
    doc.add_heading("Положение об охране труда", level=1)
    doc.add_heading("2. Термины и определения", level=2)
    for term in ("СИЗ — средства индивидуальной защиты.", "ОТ — охрана труда.", "ЛНА — локальный акт."):
        doc.add_text(label=DocItemLabel.TEXT, text=term)
    doc.add_heading("3. Обязанности", level=2)
    long_text = " ".join(
        f"Предложение номер {i} описывает обязанность работника номер {i}." for i in range(160)
    )
    doc.add_text(label=DocItemLabel.TEXT, text=f"3.1. {long_text}")
    chunks = chunker().chunk(doc, ROOT, file_sha256=SHA)
    terms = [chunk for chunk in chunks if chunk.section_key == "Раздел 2. Термины и определения"]
    assert len(terms) == 1 and terms[0].body.count("\n") == 2
    parts = [chunk for chunk in chunks if chunk.clause == "3.1"]
    assert len(parts) >= 3
    assert all(chunk.tokens <= SETTINGS.max_tokens for chunk in parts)
    assert all(chunk.breadcrumbs == (ROOT[0], "Раздел 3. Обязанности", "п. 3.1") for chunk in parts)
    for previous, following in zip(parts, parts[1:], strict=False):
        overlap = following.body.split()[: SETTINGS.overlap_tokens]
        assert " ".join(overlap) in previous.body
    assert len({chunk.chunk_id for chunk in chunks}) == len(chunks)


def test_tables_are_separate_markdown_chunks_bound_to_section() -> None:
    doc = DoclingDocument(name="положение")
    doc.add_heading("Положение", level=1)
    doc.add_heading("4. Периодичность инструктажей", level=2)
    doc.add_text(label=DocItemLabel.TEXT, text="4.1. Виды инструктажей приведены в таблице.")
    doc.add_table(
        data=_table([["Вид", "Периодичность"], ["Вводный", "при приёме"], ["Повторный", "раз в полгода"]])
    )
    doc.add_table(data=_table([["№", "Описание"], *[[str(i), f"строка {i} " * 30] for i in range(60)]]))
    chunks = chunker().chunk(doc, ROOT, file_sha256=SHA)
    tables = [chunk for chunk in chunks if chunk.kind == "table"]
    assert tables and tables[0].body.startswith("|") and "Вводный" in tables[0].body
    assert (
        tables[0].breadcrumbs == (ROOT[0], "Раздел 4. Периодичность инструктажей")
        and tables[0].clause is None
    )
    assert tables[0].item_refs == ("#/tables/0",)
    assert not any("Вводный" in chunk.body for chunk in chunks if chunk.kind == "text")
    big = [chunk for chunk in tables if "Описание" in chunk.body]
    assert len(big) >= 3 and all(chunk.tokens <= SETTINGS.max_tokens for chunk in big)
    assert all(chunk.body.splitlines()[0] == big[0].body.splitlines()[0] for chunk in big)


def test_fallback_fixed_windows_without_structure() -> None:
    doc = DoclingDocument(name="письмо")
    doc.add_heading("Ответ на запрос", level=1)
    for i in range(12):
        doc.add_text(label=DocItemLabel.TEXT, text=" ".join(f"Абзац {i} слово {j}." for j in range(50)))
    chunks = chunker().chunk(doc, ROOT, file_sha256=SHA)
    assert chunks and all(chunk.strategy == "fixed" for chunk in chunks)
    assert all(chunk.breadcrumbs == (ROOT[0], "Ответ на запрос") for chunk in chunks)
    assert all(chunk.tokens <= SETTINGS.max_tokens for chunk in chunks) and len(chunks) >= 2
    for previous, following in zip(chunks, chunks[1:], strict=False):
        assert " ".join(following.body.split()[: SETTINGS.overlap_tokens]) in previous.body
    assert "".join(chunk.body for chunk in chunks).count("Абзац 11") >= 1


def test_chunk_ids_are_deterministic_per_file() -> None:
    first = chunker().chunk(order_docx_like(), ROOT, file_sha256=SHA)
    second = chunker().chunk(order_docx_like(), ROOT, file_sha256=SHA)
    other = chunker().chunk(order_docx_like(), ROOT, file_sha256="b" * 64)
    assert [chunk.chunk_id for chunk in first] == [chunk.chunk_id for chunk in second]
    assert set(chunk.chunk_id for chunk in first).isdisjoint(chunk.chunk_id for chunk in other)
    assert all(isinstance(chunk, Chunk) for chunk in first)


def rules_with_appendices() -> DoclingDocument:
    """Как Docling читает docx правил без стилей заголовков: оглавление с табуляцией, нумерованные разделы
    списком, «Приложение № N» обычными абзацами, перечень должностей — нумерованный список."""
    doc = DoclingDocument(name="правила")
    doc.add_heading("Правила внутреннего распорядка", level=1)
    doc.add_text(label=DocItemLabel.TEXT, text="6.\tРежим рабочего времени\t14")
    doc.add_text(label=DocItemLabel.TEXT, text="Приложение № 1\t25")
    doc.add_text(label=DocItemLabel.TEXT, text="Приложение № 2\t26")
    group = doc.add_list_group()
    doc.add_list_item("Заключительные положения", enumerated=True, marker="10.", parent=group)
    doc.add_list_item(
        "За нарушение требований работники несут ответственность.",
        enumerated=True,
        marker="10.1.",
        parent=group,
    )
    doc.add_list_item(
        "Правила обязательны для всех работников.", enumerated=True, marker="10.2.", parent=group
    )
    doc.add_text(label=DocItemLabel.TEXT, text="Приложение № 1")
    doc.add_text(
        label=DocItemLabel.TEXT, text="к Правилам, утверждённым приказом от «___» ______ 202_ г. № ___"
    )
    doc.add_text(label=DocItemLabel.TEXT, text="Перечень должностей (профессий) работников Общества,")
    doc.add_text(label=DocItemLabel.TEXT, text="которым установлен суммированный учет рабочего времени")
    positions = doc.add_list_group()
    for index in range(1, 13):
        doc.add_list_item(
            f"Оператор установки {index} категории производственного отдела.",
            enumerated=True,
            marker=f"{index}.",
            parent=positions,
        )
    doc.add_text(label=DocItemLabel.TEXT, text="Приложение № 2")
    doc.add_text(
        label=DocItemLabel.TEXT, text="к Правилам, утверждённым приказом от «___» ______ 202_ г. № ___"
    )
    doc.add_text(label=DocItemLabel.TEXT, text="Режим работы в холодное время года на открытой территории")
    doc.add_table(data=_table([["Температура", "Перерыв"], ["−20", "10 минут"], ["−30", "15 минут"]]))
    doc.add_text(label=DocItemLabel.TEXT, text="Приложение № 3")
    doc.add_text(label=DocItemLabel.TEXT, text="Форма расчетного листка")
    doc.add_text(label=DocItemLabel.TEXT, text="Приложение № 4")
    doc.add_text(label=DocItemLabel.TEXT, text="Форма заявления на отпуск")
    doc.add_text(label=DocItemLabel.TEXT, text="Прошу предоставить ежегодный оплачиваемый отпуск.")
    return doc


def test_appendices_become_sections_lists_stay_together_and_toc_is_skipped() -> None:
    """Реальный корпус (2026-09-16): приложение № 1 лежало внутри раздела 10, каждая должность — свой чанк
    «п. N», строки оглавления стали ложными заголовками; перечень не находился поиском и не цитировался."""
    chunks = chunker().chunk(rules_with_appendices(), ROOT, file_sha256=SHA)
    texts = "\n".join(chunk.text for chunk in chunks)
    assert "\t" not in texts and "Режим рабочего времени 14" not in texts, "оглавление пропущено"
    assert not any(crumb.endswith(" 14") for chunk in chunks for crumb in chunk.breadcrumbs)
    clause = next(chunk for chunk in chunks if chunk.clause == "10.1")
    assert clause.breadcrumbs == (ROOT[0], "Раздел 10. Заключительные положения", "п. 10.1")

    appendix = [
        chunk for chunk in chunks if chunk.breadcrumbs[1:2] and "Приложение № 1" in chunk.breadcrumbs[1]
    ]
    assert len(appendix) == 1, [chunk.breadcrumbs for chunk in chunks]
    listing = appendix[0]
    crumb = listing.breadcrumbs[1]
    assert crumb.startswith("Приложение № 1. Перечень должностей (профессий) работников Общества")
    assert crumb.endswith("…") and len(listing.breadcrumbs) == 2 and listing.clause is None
    assert listing.heading is not None and listing.heading.endswith("суммированный учет рабочего времени")
    assert listing.body.startswith("к Правилам") and "1. Оператор установки 1" in listing.body
    assert "12. Оператор установки 12" in listing.body, "весь перечень — один чанк"
    assert listing.section_key == crumb and not any(chunk.clause in {"1", "12"} for chunk in chunks)

    table = next(chunk for chunk in chunks if chunk.kind == "table")
    assert table.breadcrumbs == (
        ROOT[0],
        "Приложение № 2. Режим работы в холодное время года на открытой территории",
    )
    # приложение без текста («Форма…») не съедает заголовок следующего приложения
    form = next(chunk for chunk in chunks if chunk.body.startswith("Прошу предоставить"))
    assert form.breadcrumbs == (ROOT[0], "Приложение № 4. Форма заявления на отпуск")
    assert not any("листка Приложение" in crumb for chunk in chunks for crumb in chunk.breadcrumbs)

    parents = build_chunk_set(chunks, SETTINGS, WordTokenCounter(), file_sha256=SHA).parents
    parent_crumbs = [parent.breadcrumbs[1] for parent in parents if len(parent.breadcrumbs) > 1]
    assert crumb in parent_crumbs and "Раздел 10. Заключительные положения" in parent_crumbs
    section_10 = next(
        parent for parent in parents if parent.breadcrumbs[1:] == ("Раздел 10. Заключительные положения",)
    )
    assert "Оператор" not in section_10.body, "приложение — отдельный parent, а не хвост раздела 10"


def test_appendix_paragraph_at_the_start_of_a_file_is_not_a_section() -> None:
    doc = DoclingDocument(name="вложение")
    doc.add_heading("Инструкция", level=1)
    doc.add_text(label=DocItemLabel.TEXT, text="Приложение № 2 к приказу от 20.05.2024 № 150")
    doc.add_text(label=DocItemLabel.TEXT, text="1.1. Инструкция устанавливает порядок действий.")
    chunks = chunker().chunk(doc, ROOT, file_sha256=SHA)
    assert chunks[0].breadcrumbs == ROOT and chunks[0].body.startswith("Приложение № 2 к приказу")
    long_appendix = DoclingDocument(name="договор")
    long_appendix.add_heading("Договор", level=1)
    long_appendix.add_text(label=DocItemLabel.TEXT, text="1.1. Предмет договора описан ниже.")
    long_appendix.add_text(
        label=DocItemLabel.TEXT,
        text="Приложение № 3 к настоящему договору содержит перечень товаров, цены, сроки поставки и "
        "условия приёмки, согласованные сторонами при подписании.",
    )
    chunks = chunker().chunk(long_appendix, ROOT, file_sha256=SHA)
    assert not any("Приложение" in crumb for chunk in chunks for crumb in chunk.breadcrumbs), (
        "длинный абзац — текст"
    )


def test_split_fixed_respects_limit_and_overlap_on_plain_text() -> None:
    text = " ".join(f"Слово{i}." for i in range(1000))
    windows = chunker().split_fixed(text, 100)
    assert all(len(window.split()) <= 100 for window in windows) and len(windows) >= 10
    assert windows[1].split()[0] in windows[0].split()
    long_sentence = " ".join(f"слово{i}" for i in range(300))
    pieces = chunker().split_fixed(long_sentence, 120)
    assert all(len(piece.split()) <= 120 for piece in pieces) and len(pieces) >= 3
