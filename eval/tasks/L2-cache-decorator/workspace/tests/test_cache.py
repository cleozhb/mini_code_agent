"""缓存行为测试 —— 这批测试编码了"给三个读方法加缓存 + 可失效"的预期行为.

初始状态下 `app/cache.py` 还是空壳，所以这些测试全部会失败；改动完成后应全绿。
"""

from __future__ import annotations

import pytest

from app.backend import Backend
from app.cache import cached, clear_cache  # noqa: F401 — 装饰器在 Client 内部用到
from app.client import Client


# ---- 基础：同样入参第二次不打 backend ----------------------------------------


def test_get_user_cached_on_same_uid():
    backend = Backend()
    c = Client(backend)
    r1 = c.get_user(1)
    r2 = c.get_user(1)
    assert r1 == r2
    assert backend.calls["get_user"] == 1, (
        f"同样 uid 应该命中缓存，实际打了 {backend.calls['get_user']} 次 backend"
    )


def test_get_profile_cached_on_same_uid():
    backend = Backend()
    c = Client(backend)
    c.get_profile(42)
    c.get_profile(42)
    assert backend.calls["get_profile"] == 1


def test_list_orders_cached_on_same_uid():
    backend = Backend()
    c = Client(backend)
    c.list_orders(7)
    c.list_orders(7)
    assert backend.calls["list_orders"] == 1


# ---- 不同入参：key 不同就应各自算一次 -----------------------------------------


def test_different_uids_do_not_share_cache():
    backend = Backend()
    c = Client(backend)
    c.get_user(1)
    c.get_user(2)
    c.get_user(1)  # 再打 1 应命中
    assert backend.calls["get_user"] == 2


# ---- 失效：clear_cache 后能重新打 backend ------------------------------------


def test_clear_cache_forces_refresh():
    backend = Backend()
    c = Client(backend)
    c.get_user(1)
    assert backend.calls["get_user"] == 1
    clear_cache()
    c.get_user(1)
    assert backend.calls["get_user"] == 2, "clear_cache 后应该重新打 backend"


def test_clear_cache_clears_all_three_methods():
    backend = Backend()
    c = Client(backend)
    c.get_user(1)
    c.get_profile(1)
    c.list_orders(1)
    assert backend.calls == {"get_user": 1, "get_profile": 1, "list_orders": 1}
    clear_cache()
    c.get_user(1)
    c.get_profile(1)
    c.list_orders(1)
    assert backend.calls == {"get_user": 2, "get_profile": 2, "list_orders": 2}


# ---- 返回值保真：缓存不能改返回值的内容 ---------------------------------------


def test_cached_return_value_preserves_shape():
    backend = Backend()
    c = Client(backend)
    assert c.get_user(5) == {"id": 5, "name": "user-5"}
    assert c.get_profile(5) == {"uid": 5, "bio": "bio-5"}
    assert c.list_orders(5) == [
        {"uid": 5, "order_id": 0},
        {"uid": 5, "order_id": 1},
        {"uid": 5, "order_id": 2},
    ]


@pytest.mark.parametrize("uid", [0, 1, 100, -1])
def test_cached_across_various_uids(uid):
    backend = Backend()
    c = Client(backend)
    a = c.get_user(uid)
    b = c.get_user(uid)
    assert a == b
    assert backend.calls["get_user"] == 1
