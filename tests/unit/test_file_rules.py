"""Правило файлов карточки (4.2, О5): основной файл, приложения, дополнения, пропуски, дубли."""

from __future__ import annotations

from pathlib import Path
from uuid import UUID, uuid4

from common.config import DEFAULT_CONFIG_PATH, load_app_config
from ingest.cards import CardRecord
from ingest.corpus import Corpus, CorpusDocument, CorpusFile
from ingest.files import plan_corpus_files, plan_document_files
from tessa_export.manifest import DocumentEntry

SETTINGS = load_app_config(DEFAULT_CONFIG_PATH).ingest
DOC = "Документ"
APPENDIX = "Приложение"
EXTRA = "Дополнительные сведения"
EDO = "Файлы для отправки по ЭДО"
SIGNED = "Подписанные документы"


def _file(card_id: UUID, name: str, category: str | None, sha: str = "") -> CorpusFile:
    return CorpusFile(
        card_id=card_id,
        row_id=uuid4(),
        name=name,
        extension=name.rsplit(".", 1)[-1].lower(),
        category=category,
        relative_path=f"files/{card_id}/{name}",
        path=Path(f"files/{card_id}/{name}"),
        sha256=sha or uuid4().hex * 2,
        size=100,
        has_text_layer=None,
        page_count=None,
        duplicate_of=None,
        smoke_note=None,
    )


def _document(type_name: str, files: list[CorpusFile]) -> CorpusDocument:
    card_id = files[0].card_id
    entry = DocumentEntry(
        card_id=card_id,
        type_name=type_name,
        type_caption=type_name,
        doc_type_title=type_name,
        doc_kind=type_name,
        coverage_kinds=[],
        number="1",
        doc_date=None,
        subject="тест",
        department=None,
        status_id=None,
        status_name=None,
        state_id=6,
        doc_status="active",
        is_cancelled=False,
        state_name=None,
        approval_state=None,
        depth=0,
        entry_paths=[],
        card_path=f"cards/{card_id}.json",
        card_raw_path=f"cards_raw/{card_id}.json",
        files=[],
    )
    return CorpusDocument(
        card_id=card_id, entry=entry, card=CardRecord(id=card_id, type_name=type_name), files=files
    )


def _roles(document: CorpusDocument) -> dict[str, str]:
    return {
        plan.file.name: plan.role or f"skip: {plan.skip_reason}"
        for plan in plan_document_files(document, SETTINGS.files, SETTINGS.extensions)
    }


def test_order_typical_set_main_appendix_supplement_and_skips() -> None:
    card = uuid4()
    roles = _roles(
        _document(
            "OrderMKC",
            [
                _file(card, "ДокШаблон Приказ №144.docx", DOC),
                _file(card, "Для печати_Приказ №144.docx", DOC),
                _file(card, "144 от 15.01.2026 Об утверждении.pdf", DOC),
                _file(card, "Приложение - Положение.docx", APPENDIX),
                _file(card, "Скан протокола.jpg", EXTRA),
                _file(card, "Подпись.sig", "Подписи ЭП"),
                _file(card, "Архив подписей.zip", "Подписи ЭП"),
            ],
        )
    )
    assert roles["ДокШаблон Приказ №144.docx"] == "main"
    assert roles["Приложение - Положение.docx"] == "appendix"
    assert roles["Скан протокола.jpg"] == "supplement"
    assert roles["Для печати_Приказ №144.docx"].startswith("skip: копия для печати")
    assert "pdf-копия основного текста" in roles["144 от 15.01.2026 Об утверждении.pdf"]
    assert roles["Подпись.sig"].startswith("skip: формат sig")
    assert roles["Архив подписей.zip"].startswith("skip: формат zip")


def test_contract_set_edo_docx_is_main_signed_pdf_skipped_supplement_not_for_contracts() -> None:
    card = uuid4()
    roles = _roles(
        _document(
            "ContractMKC",
            [
                _file(card, "Договор Д-1 подписанный.pdf", SIGNED),
                _file(card, "Договор Д-1.docx", EDO),
                _file(card, "Спецификация.xlsx", APPENDIX),
                _file(card, "Устав контрагента.pdf", EXTRA),
                _file(card, "УПД.xml", "Получено из Диадока"),
            ],
        )
    )
    assert roles["Договор Д-1.docx"] == "main"
    assert "pdf-копия" in roles["Договор Д-1 подписанный.pdf"]
    assert roles["Спецификация.xlsx"] == "appendix"
    assert "только у типов OrderMKC" in roles["Устав контрагента.pdf"]
    assert "не индексируется" in roles["УПД.xml"] or roles["УПД.xml"].startswith("skip: формат xml")


def test_pdf_becomes_main_when_no_docx_and_other_pdfs_stay_appendix() -> None:
    card = uuid4()
    roles = _roles(
        _document(
            "IncomingMKC",
            [
                _file(card, "Для печати_письмо.docx", DOC),
                _file(card, "Вх-33 скан.pdf", DOC),
                _file(card, "Вх-33 приложение.pdf", DOC),
                _file(card, "без категории.pdf", None),
            ],
        )
    )
    assert roles["Вх-33 скан.pdf"] == "main"
    assert roles["Вх-33 приложение.pdf"] == "appendix"
    assert roles["без категории.pdf"] == "appendix"
    assert roles["Для печати_письмо.docx"].startswith("skip")


def test_any_docx_fallback_and_uncategorized_files() -> None:
    card = uuid4()
    roles = _roles(
        _document("OtherDocumentMKC", [_file(card, "текст.docx", None), _file(card, "план.xlsx", None)])
    )
    assert roles == {"текст.docx": "main", "план.xlsx": "appendix"}


def test_cross_card_duplicates_keep_best_role_then_manifest_order() -> None:
    shared = "ab" * 32
    contract, justification, memo = uuid4(), uuid4(), uuid4()
    contract_doc = _document(
        "ContractMKC",
        [_file(contract, "Договор.docx", EDO), _file(contract, "Договор.pdf", APPENDIX, shared)],
    )
    justification_doc = _document(
        "BackgroundJustificationMKC", [_file(justification, "Договор.pdf", DOC, shared)]
    )
    memo_doc = _document("InhouseDocumentMKC", [_file(memo, "Копия договора.pdf", APPENDIX, shared)])
    corpus = Corpus(
        export_dir=Path("."),
        manifest=None,  # type: ignore[arg-type]
        links_graph=None,  # type: ignore[arg-type]
        documents=[contract_doc, justification_doc, memo_doc],
        issues=[],
        not_downloaded=0,
    )
    plans = plan_corpus_files(corpus, SETTINGS)
    # владелец — карточка, где файл основной (справка-обоснование), остальные — дубли
    owner = plans[justification][0]
    assert owner.role == "main" and set(owner.also_in) == {contract, memo}
    assert plans[contract][1].role is None and "дубль по sha256" in (plans[contract][1].skip_reason or "")
    assert plans[memo][0].role is None and str(justification) in (plans[memo][0].skip_reason or "")
    assert plans[contract][0].role == "main"

    # при равных ролях владелец — первый по порядку документов
    first, second = uuid4(), uuid4()
    corpus.documents = [
        _document("OrderMKC", [_file(first, "a.docx", DOC), _file(first, "общий.xlsx", APPENDIX, shared)]),
        _document("OrderMKC", [_file(second, "b.docx", DOC), _file(second, "общий.xlsx", APPENDIX, shared)]),
    ]
    plans = plan_corpus_files(corpus, SETTINGS)
    assert plans[first][1].role == "appendix" and plans[first][1].also_in == (second,)
    assert plans[second][1].role is None
