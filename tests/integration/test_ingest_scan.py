"""AC-3.4: скан из корпуса распознаётся dots.mocr и его текст попадает в индекс.

Нужен поднятый профиль ingest (vllm-dots); индекс — встроенный Qdrant, эмбеддер — фейк:
проверяется путь «скан → VLM → чанки → точки с текстом», а не качество векторов.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
from qdrant_client import QdrantClient

from common.config import DEFAULT_CONFIG_PATH, load_app_config
from common.settings import load_settings
from ingest.corpus import load_corpus
from ingest.embeddings import FakeEmbedder
from ingest.files import plan_corpus_files
from ingest.index import ChunkIndex
from ingest.parsing import DocumentParser
from ingest.pipeline import IngestPipeline
from ingest.router import choose_route
from ingest.tokens import WordTokenCounter
from synthetic import texts

pytestmark = [pytest.mark.integration, pytest.mark.gpu]

CONFIG = load_app_config(DEFAULT_CONFIG_PATH)
REAL_CORPUS = Path("data/final_2026-09-15")


@pytest.fixture(scope="module")
def parser(tmp_path_factory: pytest.TempPathFactory) -> DocumentParser:
    settings = load_settings()
    vlm = DocumentParser(
        CONFIG, vlm_base_url=settings.resolve_vlm_base_url(CONFIG), work_dir=tmp_path_factory.mktemp("w")
    )
    try:
        httpx.get(vlm.vlm_endpoint.replace("/chat/completions", "/models"), timeout=3).raise_for_status()
    except (httpx.HTTPError, OSError) as exc:
        pytest.skip(f"vllm-dots недоступен ({exc}); поднимите профиль ingest")
    return vlm


def _pipeline(parser: DocumentParser) -> tuple[IngestPipeline, ChunkIndex]:
    embedder = FakeEmbedder()
    index = ChunkIndex(QdrantClient(":memory:"), CONFIG.qdrant, embedder.dense_dim)
    index.ensure_collections()
    return IngestPipeline(
        CONFIG, parser=parser, embedder=embedder, index=index, counter=WordTokenCounter()
    ), index


def _bodies(index: ChunkIndex) -> list[str]:
    points, _ = index.client.scroll(
        CONFIG.qdrant.collection, limit=1000, with_payload=["body", "parse_route"]
    )
    assert all(point.payload and point.payload["parse_route"] == "vlm" for point in points)
    return [str(point.payload["body"]) for point in points if point.payload]


def test_synthetic_scan_pdf_text_reaches_index(parser: DocumentParser) -> None:
    corpus = load_corpus(CONFIG.paths.corpus_dir_absolute)
    if not corpus.synthetic:
        pytest.skip("в paths.corpus_dir не синтетический корпус (scripts/make_synthetic_corpus.py)")
    plans = plan_corpus_files(corpus, CONFIG.ingest)
    pipeline, index = _pipeline(parser)
    scans = [
        (document, plan)
        for document in corpus.documents
        for plan in plans[document.card_id]
        if plan.indexed and choose_route(plan.file, CONFIG.ingest).route == "vlm"
    ]
    assert scans, "в синтетическом корпусе нет файлов, уходящих в VLM"
    outcomes = [
        pipeline.process_file(pipeline.document_metadata(corpus, document), plan) for document, plan in scans
    ]
    assert all(outcome.indexed and outcome.route == "vlm" for outcome in outcomes), [
        o.reason for o in outcomes
    ]
    text = "\n".join(_bodies(index)).lower()
    # письмо-скан и скан приложения (инструкция при пожаре) — фразы из синтетических текстов
    assert "101" in text and "эвакуац" in text
    assert any(word in text for word in texts.LETTER_IN_33.lines()[1].lower().split()[:3])


def test_real_scan_pdf_from_corpus_is_recognised(parser: DocumentParser) -> None:
    if not (REAL_CORPUS / "export").is_dir():
        pytest.skip("реальный корпус не распакован в data/final_2026-09-15 (данные вне git)")
    corpus = load_corpus(REAL_CORPUS)
    plans = plan_corpus_files(corpus, CONFIG.ingest)
    candidates = [
        (document, plan)
        for document in corpus.documents
        for plan in plans[document.card_id]
        if plan.indexed
        and plan.file.extension == "pdf"
        and choose_route(plan.file, CONFIG.ingest).route == "vlm"
    ]
    assert candidates
    document, plan = min(candidates, key=lambda item: item[1].file.size)
    pipeline, index = _pipeline(parser)
    outcome = pipeline.process_file(pipeline.document_metadata(corpus, document), plan)
    assert outcome.indexed and outcome.route == "vlm" and outcome.chunks >= 1, outcome.reason
    bodies = _bodies(index)
    assert sum(len(body) for body in bodies) > 200
    assert any(
        any(char.isalpha() and char.lower() in "абвгдежзийклмнопрстуфхцчшщыэюя" for char in body)
        for body in bodies
    )
