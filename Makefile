.PHONY: check lint format typecheck test

# Единая точка проверок (NFR-6). Кроссплатформенный эквивалент: uv run python scripts/check.py
check: lint format typecheck test

lint:
	uv run ruff check

format:
	uv run ruff format --check

typecheck:
	uv run mypy

test:
	uv run pytest
