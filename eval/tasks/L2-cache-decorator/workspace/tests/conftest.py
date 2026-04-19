"""每个测试跑之前清一下 app.cache 里的状态，避免相互污染."""

from __future__ import annotations

import pytest

from app import cache as cache_mod


@pytest.fixture(autouse=True)
def _clear_between_tests() -> None:
    # clear_cache 可能不存在（初始状态下没实现），容错一下
    fn = getattr(cache_mod, "clear_cache", None)
    if callable(fn):
        fn()
    yield
    fn = getattr(cache_mod, "clear_cache", None)
    if callable(fn):
        fn()
