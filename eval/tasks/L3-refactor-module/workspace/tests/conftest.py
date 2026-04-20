"""每个测试前后清空 session 存储，保证独立."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _clean_sessions() -> None:
    # 重构前/后：reset_sessions 都必须能用（从 auth 或 session 里任一处 import 到都行）
    try:
        from app.session import reset_sessions  # type: ignore
    except ImportError:
        from app.auth import reset_sessions  # type: ignore
    reset_sessions()
    yield
    reset_sessions()
