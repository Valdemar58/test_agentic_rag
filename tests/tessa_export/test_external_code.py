"""Подключение внешнего кода: каталоги репозиториев или пакеты, установленные в окружение."""

from __future__ import annotations

import os
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

from tessa_export import external
from tessa_export.config import ExternalCodeSettings


@pytest.fixture(autouse=True)
def _isolated_environment(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Свои sys.path и окружение: CLI-тесты подхватывают .env репозитория в os.environ."""
    monkeypatch.setattr(os, "environ", dict(os.environ))
    monkeypatch.delenv(external.SDK_PATH_ENV, raising=False)
    monkeypatch.delenv(external.CARD_SERVICE_PATH_ENV, raising=False)
    saved = list(sys.path)
    yield
    sys.path[:] = saved
    external._attached.clear()


def _repo(root: Path, package: str) -> Path:
    (root / "src" / package).mkdir(parents=True, exist_ok=True)
    (root / "src" / package / "__init__.py").write_text("", encoding="utf-8")
    return root


def test_paths_are_added_to_sys_path(tmp_path: Path) -> None:
    sdk = _repo(tmp_path / "tessa_sdk", "tessa_client")
    service = _repo(tmp_path / "robot_skills", "robot_skills")

    external.attach_external_code(ExternalCodeSettings(tessa_sdk_path=sdk, card_service_path=service))

    assert str(sdk / "src") in sys.path
    assert str(service / "src") in sys.path


def test_installed_packages_need_no_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    """Поставка колёсами из внутреннего индекса: путей нет, sys.path не трогаем."""
    monkeypatch.setattr(external, "package_installed", lambda package: True)
    before = list(sys.path)

    external.attach_external_code(ExternalCodeSettings())

    assert sys.path == before


def test_missing_everything_names_both_sources(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(external, "package_installed", lambda package: False)
    with pytest.raises(external.ExternalCodeError) as exc_info:
        external.attach_external_code(ExternalCodeSettings())
    message = str(exc_info.value)
    assert "external.tessa_sdk_path" in message and "не задан" in message
    assert "tessa_client не установлен в окружение" in message

    with pytest.raises(external.ExternalCodeError) as exc_info:
        external.attach_external_code(
            ExternalCodeSettings(tessa_sdk_path=tmp_path, card_service_path=tmp_path)
        )
    assert str(tmp_path) in str(exc_info.value)


def test_installed_package_covers_a_wrong_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Путь задан по ошибке (например, из Docker-примера), но пакет стоит в окружении — работаем."""
    monkeypatch.setattr(external, "package_installed", lambda package: package == "robot_skills")
    sdk = _repo(tmp_path / "tessa_sdk", "tessa_client")

    external.attach_external_code(
        ExternalCodeSettings(tessa_sdk_path=sdk, card_service_path=Path("/opt/external/robot_skills"))
    )

    assert str(sdk / "src") in sys.path


def test_env_variables_are_used_when_config_has_no_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Те же TESSA_SDK_PATH и CARD_SERVICE_PATH, что читает остальной проект (в том числе из .env)."""
    monkeypatch.setattr(external, "package_installed", lambda package: False)
    sdk = _repo(tmp_path / "tessa_sdk", "tessa_client")
    service = _repo(tmp_path / "robot_skills", "robot_skills")
    monkeypatch.setenv(external.SDK_PATH_ENV, str(sdk))
    monkeypatch.setenv(external.CARD_SERVICE_PATH_ENV, str(service))

    external.attach_external_code(ExternalCodeSettings())

    assert str(sdk / "src") in sys.path
    assert str(service / "src") in sys.path


def test_env_variable_covers_a_stale_path_from_the_docker_example(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Случай заказчика: в конфиге остался путь контейнера, верный каталог указан в .env."""
    monkeypatch.setattr(external, "package_installed", lambda package: package == "tessa_client")
    service = _repo(tmp_path / "robot_skills", "robot_skills")
    monkeypatch.setenv(external.CARD_SERVICE_PATH_ENV, str(service))

    external.attach_external_code(ExternalCodeSettings(card_service_path=Path("/opt/external/robot_skills")))

    assert str(service / "src") in sys.path


def test_error_names_config_env_and_package(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(external, "package_installed", lambda package: False)
    monkeypatch.delenv(external.SDK_PATH_ENV, raising=False)
    monkeypatch.delenv(external.CARD_SERVICE_PATH_ENV, raising=False)

    with pytest.raises(external.ExternalCodeError) as exc_info:
        external.attach_external_code(ExternalCodeSettings())

    message = str(exc_info.value)
    assert "external.tessa_sdk_path" in message
    assert external.SDK_PATH_ENV in message
    assert "не установлен в окружение" in message
