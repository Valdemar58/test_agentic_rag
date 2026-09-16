"""AC-9.2: в коде приложения нет сырых SQL-запросов и `text()` — доступ к БД только через ORM (FR-9).

Проверяются `src/`, `alembic/`, `mocks/`, `scripts/`: импорт `text` из SQLAlchemy, вызовы `text(...)`,
`execute`/`exec_driver_sql` со строковым SQL и любые строковые константы, похожие на SQL-операторы.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

from common.config import ROOT

SCANNED_DIRS = ("src", "alembic", "mocks", "scripts")
SQL_STATEMENT_RE = re.compile(
    r"^\s*(select\b.*\bfrom\b|insert\s+into\b|update\s+\w+\s+set\b|delete\s+from\b|"
    r"create\s+(table|index|extension)\b|alter\s+table\b|drop\s+(table|index)\b|truncate\b)",
    re.IGNORECASE | re.DOTALL,
)
RAW_EXECUTORS = ("execute", "exec_driver_sql", "executemany", "scalar", "scalars")


def _name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def sql_violations(source: str, filename: str = "<source>") -> list[str]:
    """Нарушения в порядке строк файла: «путь:строка: описание»."""
    tree = ast.parse(source, filename=filename)
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("sqlalchemy"):
            if any(alias.name == "text" for alias in node.names):
                found.append((node.lineno, "импорт text из SQLAlchemy"))
        elif isinstance(node, ast.Call):
            name = _name(node.func)
            if name == "text" and isinstance(node.func, ast.Attribute):
                base = _name(node.func.value)
                if base in ("sqlalchemy", "sa"):
                    found.append((node.lineno, f"вызов {base}.text()"))
            elif name == "text" and isinstance(node.func, ast.Name):
                found.append((node.lineno, "вызов text()"))
            elif name in RAW_EXECUTORS and node.args:
                first = node.args[0]
                if isinstance(first, ast.Constant) and isinstance(first.value, str):
                    found.append((node.lineno, f"{name}() со строкой SQL"))
        elif (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and SQL_STATEMENT_RE.match(node.value)
        ):
            found.append((node.lineno, f"строка похожа на SQL: {node.value[:40]!r}"))
    return [f"{filename}:{lineno}: {message}" for lineno, message in sorted(found, key=lambda item: item[0])]


def _python_files() -> list[Path]:
    files: list[Path] = []
    for directory in SCANNED_DIRS:
        files += sorted((ROOT / directory).rglob("*.py"))
    return files


def test_application_code_has_no_raw_sql() -> None:
    files = _python_files()
    assert len(files) > 50, "ожидались исходники приложения"
    violations: list[str] = []
    for path in files:
        violations += sql_violations(path.read_text(encoding="utf-8"), str(path.relative_to(ROOT)))
    assert violations == []


def test_detector_catches_raw_sql_samples() -> None:
    sample = (
        "from sqlalchemy import text\n"
        "import sqlalchemy as sa\n"
        "def f(session):\n"
        "    session.execute(text('SELECT 1'))\n"
        "    session.execute(sa.text('DELETE FROM t'))\n"
        "    session.execute('INSERT INTO t VALUES (1)')\n"
        "    query = 'select id from users where x = 1'\n"
    )
    found = sql_violations(sample, "sample.py")
    assert [line.split(": ", 1)[1] for line in found] == [
        "импорт text из SQLAlchemy",
        "вызов text()",
        "вызов sa.text()",
        "строка похожа на SQL: 'DELETE FROM t'",
        "execute() со строкой SQL",
        "строка похожа на SQL: 'INSERT INTO t VALUES (1)'",
        "строка похожа на SQL: 'select id from users where x = 1'",
    ]
    assert [line.split(":")[1] for line in found] == ["1", "4", "5", "5", "6", "6", "7"]
    assert sql_violations("x = select(User).where(User.id == 1)\nname = 'Отчёт сдаётся до пятого'") == []
