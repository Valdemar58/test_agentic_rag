"""Единая точка проверок (NFR-6): ruff check, ruff format --check, mypy, pytest.

Кроссплатформенный эквивалент `make check`. Запуск: `uv run python scripts/check.py`.
Выполняет все проверки, даже если одна упала, и возвращает ненулевой код при любой ошибке.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CHECKS: list[tuple[str, list[str]]] = [
    ("ruff check", [sys.executable, "-m", "ruff", "check"]),
    ("ruff format --check", [sys.executable, "-m", "ruff", "format", "--check"]),
    ("mypy", [sys.executable, "-m", "mypy"]),
    ("pytest", [sys.executable, "-m", "pytest"]),
]


def main() -> int:
    failed: list[str] = []
    for name, command in CHECKS:
        print(f"\n=== {name} ===", flush=True)
        result = subprocess.run(command, cwd=ROOT)
        if result.returncode != 0:
            failed.append(name)
    print("\n=== итог ===")
    if failed:
        print("ПРОВАЛЕНО: " + ", ".join(failed))
        return 1
    print("Все проверки прошли.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
