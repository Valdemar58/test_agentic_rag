"""Эмбеддинги BAAI/bge-m3: dense и sparse (lexical weights) одним проходом через transformers.

План Б из журнала решений (FlagEmbedding не используется): модель XLM-R из локального `models/`,
dense — CLS-вектор последнего слоя с нормировкой, sparse — голова `sparse_linear.pt`
(Linear 1024→1) поверх скрытых состояний токенов: вес токена = relu(W·h + b), по одинаковым
токенам берётся максимум, служебные токены (CLS, EOS, PAD, UNK) выбрасываются. Это тот же
алгоритм, что в `BGEM3FlagModel.encode(return_sparse=True)`. Устройство — из конфига: при инжесте
GPU после выгрузки VLM, в рантайме CPU (§2 ТЗ); если CUDA запрошена, но недоступна — CPU с
предупреждением в логе.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from common.config import EmbeddingSettings

logger = logging.getLogger(__name__)

SPARSE_HEAD_FILE = "sparse_linear.pt"


@dataclass(frozen=True)
class Embedding:
    dense: list[float]
    sparse: dict[int, float]


class Embedder(Protocol):
    @property
    def dense_dim(self) -> int: ...

    def encode(self, texts: list[str]) -> list[Embedding]: ...


class FakeEmbedder:
    """Детерминированные векторы по хэшу слов — для тестов без весов модели."""

    def __init__(self, dense_dim: int = 8) -> None:
        self._dim = dense_dim

    @property
    def dense_dim(self) -> int:
        return self._dim

    def encode(self, texts: list[str]) -> list[Embedding]:
        result: list[Embedding] = []
        for text in texts:
            dense = [0.0] * self._dim
            sparse: dict[int, float] = {}
            for word in text.lower().split():
                code = sum(ord(char) for char in word)
                dense[code % self._dim] += 1.0
                sparse[code % 10007] = max(sparse.get(code % 10007, 0.0), 1.0 + len(word) / 10)
            norm = sum(value * value for value in dense) ** 0.5 or 1.0
            result.append(Embedding(dense=[value / norm for value in dense], sparse=sparse))
        return result


def resolve_device(requested: str) -> str:
    import torch

    if requested == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA запрошена для эмбеддингов, но недоступна в этой сборке torch — считаю на CPU")
        return "cpu"
    return requested


class BgeM3Embedder:
    def __init__(self, model_dir: Path, settings: EmbeddingSettings, device: str) -> None:
        self._model_dir = model_dir
        self._settings = settings
        self._device = device
        self._loaded: tuple[Any, Any, Any] | None = None

    @property
    def dense_dim(self) -> int:
        return self._settings.dense_dim

    @property
    def device(self) -> str:
        return self._device

    def _load(self) -> tuple[Any, Any, Any]:
        if self._loaded is None:
            import torch
            from transformers import AutoModel, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(str(self._model_dir), local_files_only=True)
            dtype = torch.float16 if self._device.startswith("cuda") else torch.float32
            model = AutoModel.from_pretrained(str(self._model_dir), local_files_only=True, dtype=dtype)
            model.to(self._device).eval()
            state = torch.load(self._model_dir / SPARSE_HEAD_FILE, map_location=self._device)
            sparse_head = torch.nn.Linear(model.config.hidden_size, 1)
            sparse_head.load_state_dict(state)
            sparse_head.to(self._device, dtype=dtype).eval()
            self._loaded = (tokenizer, model, sparse_head)
            logger.info("bge-m3 загружена из %s на %s", self._model_dir, self._device)
        return self._loaded

    def encode(self, texts: list[str]) -> list[Embedding]:
        import torch

        tokenizer, model, sparse_head = self._load()
        special_ids = {
            token_id
            for token_id in (
                tokenizer.cls_token_id,
                tokenizer.eos_token_id,
                tokenizer.pad_token_id,
                tokenizer.unk_token_id,
            )
            if token_id is not None
        }
        result: list[Embedding] = []
        batch_size = self._settings.batch_size
        with torch.inference_mode():
            for start in range(0, len(texts), batch_size):
                batch = texts[start : start + batch_size]
                encoded = tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self._settings.max_length,
                    return_tensors="pt",
                ).to(self._device)
                hidden = model(**encoded).last_hidden_state
                dense = hidden[:, 0]
                if self._settings.normalize:
                    dense = torch.nn.functional.normalize(dense, dim=-1)
                weights = torch.relu(sparse_head(hidden)).squeeze(-1)
                dense_list = dense.float().cpu().tolist()
                weights_list = weights.float().cpu().tolist()
                ids_list = encoded["input_ids"].cpu().tolist()
                mask_list = encoded["attention_mask"].cpu().tolist()
                for row, (ids, mask, token_weights) in enumerate(
                    zip(ids_list, mask_list, weights_list, strict=True)
                ):
                    sparse: dict[int, float] = {}
                    for token_id, present, weight in zip(ids, mask, token_weights, strict=True):
                        if not present or token_id in special_ids or weight <= 0:
                            continue
                        if weight > sparse.get(token_id, 0.0):
                            sparse[token_id] = float(weight)
                    result.append(Embedding(dense=dense_list[row], sparse=sparse))
        return result


def build_embedder(model_dir: Path, settings: EmbeddingSettings, *, ingest: bool) -> BgeM3Embedder:
    requested = settings.ingest_device if ingest else settings.runtime_device
    return BgeM3Embedder(model_dir, settings, resolve_device(requested))
