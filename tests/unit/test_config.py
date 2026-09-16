"""Единый конфиг configs/app.yaml: пример валиден, ошибки читаемы."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from common.config import (
    CONFIG_PATH_ENV,
    DEFAULT_CONFIG_PATH,
    ROOT,
    ConfigError,
    config_path,
    load_app_config,
)


def _example() -> dict[str, object]:
    return dict(yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8")))


def _write(tmp_path: Path, data: dict[str, object]) -> Path:
    path = tmp_path / "app.yaml"
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return path


def test_example_config_is_valid_and_budget_is_12_gib() -> None:
    config = load_app_config(DEFAULT_CONFIG_PATH)
    assert config.gpu.vram_budget_gib == 12.0
    assert config.gpu.vllm_memory_gib <= config.gpu.vram_budget_gib
    assert config.vllm.qwen.tool_call_parser == "hermes"
    assert config.models.dir_absolute == ROOT / "models"
    assert config.models.local_path(config.models.dots) == ROOT / "models" / "DotsMOCR"
    assert len(config.models.all_sources()) == 6
    assert all(len(source.revision) == 40 for source in config.models.all_sources())
    # Docling ищет модели в artifacts_path по repo_id с «/» → «--»
    assert config.models.docling_artifacts_dir == ROOT / "models"
    assert (
        config.models.local_path(config.models.docling_layout).name == "docling-project--docling-layout-heron"
    )


def test_docling_model_name_must_follow_docling_layout(tmp_path: Path) -> None:
    data = _example()
    models = dict(data["models"])  # type: ignore[call-overload]
    models["docling_layout"] = {**models["docling_layout"], "local_name": "layout"}
    data["models"] = models
    with pytest.raises(ConfigError, match="так ищет Docling"):
        load_app_config(_write(tmp_path, data))
    data = _example()
    ingest = dict(data["ingest"])  # type: ignore[call-overload]
    files = dict(ingest["files"])
    files["main_candidates"] = [{"extension": "docx"}]
    ingest["files"] = files
    data["ingest"] = ingest
    with pytest.raises(ConfigError, match="any_category"):
        load_app_config(_write(tmp_path, data))


def test_example_config_matches_tz_requirements() -> None:
    config = load_app_config(DEFAULT_CONFIG_PATH)
    assert config.agent.max_tool_calls == 8  # FR-1
    assert config.retrieval.rerank_candidates == 20  # FR-2: rerank топ-20
    assert config.retrieval.default_statuses == ["active"]  # AC-2.2
    assert (config.ingest.chunking.max_tokens, config.ingest.chunking.overlap_tokens) == (512, 64)  # FR-3
    assert config.embedding.runtime_device == "cpu" and config.reranker.device == "cpu"  # §2
    # N9 (решение 2026-09-16): размышления выключены в цикле инструментов, включены в разборе и ответе
    assert config.agent.thinking.tool_loop is False and config.agent.thinking.rewrite is True
    assert config.agent.thinking.answer is True
    assert config.agent.llm_options("tool_loop").enable_thinking is False
    assert config.agent.llm_options("answer").max_tokens == config.agent.llm.thinking_max_tokens
    assert config.eval.first_signal_budget_s == 5.0  # NFR-2
    assert config.paths.corpus_dir_absolute == ROOT / "data" / "corpus"
    assert config.eval.golden_set_absolute == ROOT / "eval" / "golden_set.yaml"


def test_cross_field_rules(tmp_path: Path) -> None:
    data = _example()
    retrieval = dict(data["retrieval"])  # type: ignore[call-overload]
    retrieval["top_k"] = 50
    data["retrieval"] = retrieval
    with pytest.raises(ConfigError, match="top_k ≤ max_top_k"):
        load_app_config(_write(tmp_path, data))
    data = _example()
    ingest = dict(data["ingest"])  # type: ignore[call-overload]
    ingest["chunking"] = {**ingest["chunking"], "overlap_tokens": 512}
    data["ingest"] = ingest
    with pytest.raises(ConfigError, match="overlap_tokens"):
        load_app_config(_write(tmp_path, data))


def test_vllm_memory_above_budget_is_rejected(tmp_path: Path) -> None:
    data = _example()
    gpu = dict(data["gpu"])  # type: ignore[call-overload]
    gpu["vllm_memory_gib"] = 13
    data["gpu"] = gpu
    with pytest.raises(ConfigError, match="больше бюджета"):
        load_app_config(_write(tmp_path, data))


def test_unknown_field_and_bad_revision_are_reported(tmp_path: Path) -> None:
    data = _example()
    data["unexpected"] = 1
    with pytest.raises(ConfigError, match="unexpected"):
        load_app_config(_write(tmp_path, data))
    data = _example()
    models = dict(data["models"])  # type: ignore[call-overload]
    models["qwen"] = {**models["qwen"], "revision": "main"}
    data["models"] = models
    with pytest.raises(ConfigError, match="models.qwen.revision"):
        load_app_config(_write(tmp_path, data))


def test_missing_and_invalid_yaml(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="не найден"):
        load_app_config(tmp_path / "nope.yaml")
    broken = tmp_path / "broken.yaml"
    broken.write_text("gpu: [unclosed", encoding="utf-8")
    with pytest.raises(ConfigError, match="YAML"):
        load_app_config(broken)
    scalar = tmp_path / "scalar.yaml"
    scalar.write_text("just text", encoding="utf-8")
    with pytest.raises(ConfigError, match="словарь"):
        load_app_config(scalar)


def test_config_path_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(CONFIG_PATH_ENV, raising=False)
    assert config_path() == DEFAULT_CONFIG_PATH
    monkeypatch.setenv(CONFIG_PATH_ENV, "/tmp/other.yaml")
    assert config_path() == Path("/tmp/other.yaml")
