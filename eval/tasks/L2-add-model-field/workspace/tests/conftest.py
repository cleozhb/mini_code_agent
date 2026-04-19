"""测试前自动清空内存存储，保证测试之间相互独立."""

from __future__ import annotations

import pytest

from app.api import reset_store


@pytest.fixture(autouse=True)
def _clean_store() -> None:
    reset_store()
    yield
    reset_store()
