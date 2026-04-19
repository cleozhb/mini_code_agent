"""面向业务侧的 Client：包一层 Backend，未来要给热点方法加缓存.

当前三个读方法每次都穿透到 backend，高并发下压力大；目标是给它们加缓存。
"""

from __future__ import annotations

from .backend import Backend


class Client:
    """薄封装：三个读方法 get_user / get_profile / list_orders."""

    def __init__(self, backend: Backend) -> None:
        self.backend = backend

    def get_user(self, uid: int) -> dict:
        return self.backend.get_user(uid)

    def get_profile(self, uid: int) -> dict:
        return self.backend.get_profile(uid)

    def list_orders(self, uid: int) -> list[dict]:
        return self.backend.list_orders(uid)
