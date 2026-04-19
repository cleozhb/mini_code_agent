"""模拟后端服务：业务方不关心它怎么实现，我们只在乎它被调了几次.

真实环境下这里会走 HTTP 或数据库；测试里只关心 `calls` 计数。
"""

from __future__ import annotations


class Backend:
    """记录每个方法被调用次数的小桩."""

    def __init__(self) -> None:
        self.calls: dict[str, int] = {
            "get_user": 0,
            "get_profile": 0,
            "list_orders": 0,
        }

    def get_user(self, uid: int) -> dict:
        self.calls["get_user"] += 1
        return {"id": uid, "name": f"user-{uid}"}

    def get_profile(self, uid: int) -> dict:
        self.calls["get_profile"] += 1
        return {"uid": uid, "bio": f"bio-{uid}"}

    def list_orders(self, uid: int) -> list[dict]:
        self.calls["list_orders"] += 1
        return [{"uid": uid, "order_id": i} for i in range(3)]
