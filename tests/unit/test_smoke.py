"""Smoke-тест каркаса: пакеты из src импортируются, pytest настроен."""

import common


def test_common_package_importable() -> None:
    assert common.__doc__
