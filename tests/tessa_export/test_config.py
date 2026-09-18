"""Конфиг и seed-файл экспорт-скрипта: валидация и понятные ошибки (§8.1 ТЗ)."""

from __future__ import annotations

from pathlib import Path
from uuid import UUID

import pytest

from tessa_export.config import (
    CANCELLED_STATUS_ID,
    ConfigError,
    ExcludeRule,
    ExportConfig,
    load_config,
    load_seed,
)

TOOL_DIR = Path(__file__).resolve().parents[2] / "tools" / "tessa_export"

MINIMAL_CONFIG = """
tessa:
  base_url: https://tessa.local/
external:
  tessa_sdk_path: /opt/external/tessa_sdk
  card_service_path: /opt/external/robot_skills
"""


def _write(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_example_config_is_valid() -> None:
    config = load_config(TOOL_DIR / "config.example.yaml")
    assert config.traversal.max_depth == 2
    assert config.traversal.max_docs == 400  # первый запуск с запасом, затем ручной отбор
    assert config.traversal.directions == ["outgoing", "incoming"]
    # решение заказчика 2026-09-15 (сценарий B): бухгалтерские типы карточек исключены, seed защищён
    excluded_types = {name for rule in config.exclude_rules for name in rule.card_type_names}
    assert excluded_types == {"PrimaryDocumentMKC", "IncomingEDO", "ActReconciliationMKC"}
    assert all(rule.applies_to_seed is False for rule in config.exclude_rules)
    assert "pdf" in config.files.allowed_extensions
    assert "doc" not in config.files.allowed_extensions
    # у заказчика нет CA сервера Тессы: проверка сертификата выключена и в примере, и по умолчанию
    assert config.tessa.verify_tls is False and config.tessa.ca_bundle is None
    assert config.status.cancelled_status_ids == [CANCELLED_STATUS_ID]
    assert 5 in config.status.cancelled_state_ids and 6 in config.status.active_state_ids
    assert "Инструкции" in config.coverage.coverage_kind_map


def test_seed_file_from_tz_has_30_unique_ids() -> None:
    seed = load_seed(TOOL_DIR / "seed_cards.yaml")
    assert len(seed) == 30
    assert len({card.id for card in seed}) == 30


def test_minimal_config_defaults_and_relative_paths(tmp_path: Path) -> None:
    path = _write(tmp_path, "config.yaml", MINIMAL_CONFIG)
    config = load_config(path)
    assert config.tessa.base_url == "https://tessa.local"
    assert config.seed_file == tmp_path / "seed_cards.yaml"
    assert config.output_dir == tmp_path / "output"
    assert config.tessa.tessa_version == "4.2"
    assert config.tessa.verify_tls is False


def test_missing_required_field_gives_readable_error(tmp_path: Path) -> None:
    path = _write(tmp_path, "config.yaml", "tessa:\n  verify_tls: true\n")
    with pytest.raises(ConfigError) as exc_info:
        load_config(path)
    assert "tessa.base_url" in str(exc_info.value)


def test_external_paths_are_optional(tmp_path: Path) -> None:
    """Пути к внешнему коду можно не задавать: пакеты ставятся в окружение из внутреннего индекса."""
    path = _write(tmp_path, "config.yaml", "tessa:\n  base_url: https://tessa.local\n")
    config = load_config(path)
    assert config.external.tessa_sdk_path is None
    assert config.external.card_service_path is None


def test_unknown_key_is_rejected(tmp_path: Path) -> None:
    path = _write(tmp_path, "config.yaml", MINIMAL_CONFIG + "traversal:\n  max_deep: 3\n")
    with pytest.raises(ConfigError) as exc_info:
        load_config(path)
    assert "max_deep" in str(exc_info.value)


def test_bad_url_is_rejected(tmp_path: Path) -> None:
    path = _write(tmp_path, "config.yaml", MINIMAL_CONFIG.replace("https://tessa.local/", "tessa.local"))
    with pytest.raises(ConfigError) as exc_info:
        load_config(path)
    assert "http" in str(exc_info.value)


def test_missing_config_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigError) as exc_info:
        load_config(tmp_path / "nope.yaml")
    assert "не найден" in str(exc_info.value)


def test_credentials_from_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config = load_config(_write(tmp_path, "config.yaml", MINIMAL_CONFIG))
    monkeypatch.delenv("TESSA_USERNAME", raising=False)
    monkeypatch.delenv("TESSA_PASSWORD", raising=False)
    with pytest.raises(ConfigError) as exc_info:
        config.resolve_credentials()
    assert "TESSA_USERNAME" in str(exc_info.value)
    monkeypatch.setenv("TESSA_USERNAME", "DOMAIN\\user")
    monkeypatch.setenv("TESSA_PASSWORD", "secret")
    assert config.resolve_credentials() == ("DOMAIN\\user", "secret")


def test_seed_plain_list_and_comments(tmp_path: Path) -> None:
    plain = _write(tmp_path, "plain.yaml", "- 982256c5-b088-4adc-b90c-87a0e2c5fb28\n")
    assert load_seed(plain)[0].id == UUID("982256c5-b088-4adc-b90c-87a0e2c5fb28")
    rich = _write(
        tmp_path,
        "rich.yaml",
        "cards:\n  - id: 982256c5-b088-4adc-b90c-87a0e2c5fb28\n    comment: приказ\n",
    )
    assert load_seed(rich)[0].comment == "приказ"


def test_seed_rejects_bad_id_duplicates_and_empty(tmp_path: Path) -> None:
    with pytest.raises(ConfigError) as exc_info:
        load_seed(_write(tmp_path, "bad.yaml", "- not-a-uuid\n"))
    assert "элемент 1" in str(exc_info.value)
    with pytest.raises(ConfigError) as exc_info:
        load_seed(
            _write(
                tmp_path,
                "dup.yaml",
                "- 982256c5-b088-4adc-b90c-87a0e2c5fb28\n- 982256c5-b088-4adc-b90c-87a0e2c5fb28\n",
            )
        )
    assert "повторяющиеся" in str(exc_info.value)
    with pytest.raises(ConfigError):
        load_seed(_write(tmp_path, "empty.yaml", "cards: []\n"))


def test_exclude_rule_requires_criteria() -> None:
    with pytest.raises(ValueError):
        ExcludeRule(reason="пусто")
    with pytest.raises(ValueError):
        ExcludeRule(field="DocumentCommonInfo.StatusID")
    rule = ExcludeRule(field="DocumentCommonInfo.StatusID", values=[str(CANCELLED_STATUS_ID)])
    assert rule.values == [str(CANCELLED_STATUS_ID)]


def test_extensions_are_normalized() -> None:
    config = ExportConfig.model_validate(
        {
            "tessa": {"base_url": "http://t"},
            "external": {"tessa_sdk_path": "/a", "card_service_path": "/b"},
            "files": {"allowed_extensions": [".PDF", "Docx"]},
        }
    )
    assert config.files.allowed_extensions == ["pdf", "docx"]
