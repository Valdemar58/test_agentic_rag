"""bge-m3 (4.6): dense совпадает с sentence-transformers, sparse — веса по токенам; нужны веса в models/."""

from __future__ import annotations

import math

import pytest

from common.config import DEFAULT_CONFIG_PATH, load_app_config
from common.model_store import model_status
from ingest.embeddings import BgeM3Embedder, resolve_device

pytestmark = pytest.mark.integration

CONFIG = load_app_config(DEFAULT_CONFIG_PATH)
TEXTS = [
    "Руководителям подразделений предоставлять ежемесячный отчёт до 5 числа.",
    "Ежемесячный отчёт подразделения сдаётся руководителем до пятого числа месяца.",
    "Тренировки по эвакуации при пожаре проводить не реже одного раза в год.",
]


def _cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True)) / (
        math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    )


@pytest.fixture(scope="module")
def embedder() -> BgeM3Embedder:
    if not model_status(CONFIG.models.embedding, CONFIG.models.dir_absolute).present:
        pytest.skip("веса bge-m3 не скачаны (scripts/download_models.py --only embedding)")
    return BgeM3Embedder(CONFIG.models.local_path(CONFIG.models.embedding), CONFIG.embedding, "cpu")


def test_dense_matches_sentence_transformers_and_ranks_paraphrase_higher(embedder: BgeM3Embedder) -> None:
    from sentence_transformers import SentenceTransformer

    ours = embedder.encode(TEXTS)
    reference = SentenceTransformer(str(CONFIG.models.local_path(CONFIG.models.embedding)), device="cpu")
    expected = reference.encode(TEXTS, normalize_embeddings=True)
    for embedding, vector in zip(ours, expected, strict=True):
        assert len(embedding.dense) == CONFIG.embedding.dense_dim
        assert _cosine(embedding.dense, [float(x) for x in vector]) > 0.999
        assert abs(math.sqrt(sum(x * x for x in embedding.dense)) - 1.0) < 1e-3
    assert _cosine(ours[0].dense, ours[1].dense) > _cosine(ours[0].dense, ours[2].dense) + 0.15


def test_sparse_weights_cover_content_tokens_and_share_terms(embedder: BgeM3Embedder) -> None:
    first, second, other = embedder.encode(TEXTS)
    tokenizer = embedder._load()[0]
    special = {tokenizer.cls_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id, tokenizer.unk_token_id}
    for embedding in (first, second, other):
        assert embedding.sparse and all(weight > 0 for weight in embedding.sparse.values())
        assert not (set(embedding.sparse) & special)
    shared = set(first.sparse) & set(second.sparse)
    unrelated = set(first.sparse) & set(other.sparse)
    assert len(shared) > len(unrelated)
    report_ids = set(tokenizer.encode("отчёт", add_special_tokens=False))
    assert report_ids & set(first.sparse)


def test_resolve_device_falls_back_to_cpu_without_cuda() -> None:
    import torch

    assert resolve_device("cpu") == "cpu"
    assert resolve_device("cuda") == ("cuda" if torch.cuda.is_available() else "cpu")
