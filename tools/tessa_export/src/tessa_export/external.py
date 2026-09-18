"""Подключение SDK Тессы и схем сервиса карточек (§8.0 ТЗ: код не копируется в пакет).

Два способа. Либо каталоги репозиториев в конфиге (`external.tessa_sdk_path`,
`external.card_service_path`) — тогда их подкаталоги src добавляются в sys.path; так работает
Docker-образ, где они лежат в /opt/external. Либо пакеты `tessa_client` и `robot_skills` уже
установлены в окружение (например, колёсами из внутреннего devpi) — тогда пути в конфиге можно
не задавать. Путь имеет приоритет; если по нему пакета нет, а установленный есть, берётся
установленный.
"""

from __future__ import annotations

import sys
from importlib.util import find_spec
from pathlib import Path

from tessa_export.config import ExternalCodeSettings

SDK_PACKAGE = "tessa_client"
CARD_SERVICE_PACKAGE = "robot_skills"


class ExternalCodeError(RuntimeError):
    """Внешний код не найден: ни по указанному пути, ни среди установленных пакетов."""


# Каталоги, которые этот модуль сам добавил в sys.path: пакет оттуда — не «установленный»
_attached: set[Path] = set()


def package_installed(package: str) -> bool:
    """True, если пакет стоит в окружении (колесом), а не подхвачен из каталога по пути."""
    try:
        spec = find_spec(package)
    except (ImportError, ValueError):
        return False
    if spec is None:
        return False
    origin = Path(spec.origin) if spec.origin else None
    return origin is None or not any(origin.is_relative_to(entry) for entry in _attached)


def _source_dir(label: str, root: Path | None, package: str, hint: str) -> Path | None:
    """Каталог src для sys.path или None, если пакет уже установлен в окружение."""
    source_dir = None if root is None else root / "src"
    if source_dir is not None and (source_dir / package / "__init__.py").is_file():
        return source_dir
    if package_installed(package):
        return None
    where = "не задан" if root is None else f"= {root}, но пакета {package} там нет"
    raise ExternalCodeError(
        f"{hint} не найден: параметр {label} в конфиге {where}, "
        f"и пакет {package} не установлен в окружение (например, из внутреннего индекса пакетов)."
    )


def attach_external_code(settings: ExternalCodeSettings) -> None:
    """Проверяет источники внешнего кода и добавляет каталоги src в sys.path. Идемпотентно."""
    sources = [
        _source_dir("external.tessa_sdk_path", settings.tessa_sdk_path, SDK_PACKAGE, "SDK Тессы"),
        _source_dir(
            "external.card_service_path",
            settings.card_service_path,
            CARD_SERVICE_PACKAGE,
            "Сервис карточек (схемы)",
        ),
    ]
    for source in sources:
        if source is None:
            continue
        entry = str(source)
        _attached.add(source)
        if entry not in sys.path:
            sys.path.append(entry)
