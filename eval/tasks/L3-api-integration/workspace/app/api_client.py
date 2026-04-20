"""后端服务的 API 客户端。

已实现：`get_user` —— 带 4xx → ApiError、5xx → 重试 MAX_RETRIES 次的行为。
未实现：`create_item` / `list_items`，需要按 get_user 的模式补上。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from app.http import HttpClient, HttpResponse


# 最多重试次数（= 总请求次数 - 1）。_SLEEP 单列成模块级变量，测试可以打补丁
# 规避真实睡眠。
MAX_RETRIES: int = 3
_SLEEP = time.sleep


class ApiError(Exception):
    """非 2xx 响应（4xx 直接抛，5xx 在重试耗尽后抛）."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(f"[{status}] {message}")
        self.status = status
        self.message = message


@dataclass
class User:
    id: int
    name: str


@dataclass
class Item:
    id: int
    name: str
    qty: int


class ApiClient:
    def __init__(self, http: HttpClient) -> None:
        self.http = http

    # ------------------------------------------------------------------
    # 内部：带重试的请求。2xx 返回；4xx 立刻抛 ApiError；
    # 5xx 重试 MAX_RETRIES 次（指数退避 0.01 → 0.02 → 0.04 …），
    # 还是 5xx 就抛 ApiError。
    # ------------------------------------------------------------------
    def _request_with_retry(
        self,
        method: str,
        path: str,
        json: Any = None,
    ) -> HttpResponse:
        delay = 0.01
        for attempt in range(MAX_RETRIES + 1):
            resp = self.http.request(method, path, json)
            if 200 <= resp.status < 300:
                return resp
            if 400 <= resp.status < 500:
                raise ApiError(resp.status, f"{resp.body}")
            # 5xx
            if attempt < MAX_RETRIES:
                _SLEEP(delay)
                delay *= 2
                continue
            raise ApiError(resp.status, f"server error after retries: {resp.body}")
        # 理论上不会走到这里
        raise RuntimeError("unreachable")

    # ------------------------------------------------------------------
    # 已实现端点
    # ------------------------------------------------------------------
    def get_user(self, uid: int) -> User:
        resp = self._request_with_retry("GET", f"/users/{uid}")
        body = resp.body
        return User(id=body["id"], name=body["name"])

    # ------------------------------------------------------------------
    # 待实现端点
    # ------------------------------------------------------------------
    def create_item(self, name: str, qty: int) -> Item:
        """POST /items，body = {"name": name, "qty": qty}，响应 201 返回 Item.

        约定：
          - 复用 `_request_with_retry` 的错误处理 + 重试模式（不要自己重写）
          - 4xx 原样抛 ApiError
          - 5xx 重试 MAX_RETRIES 次后再抛
          - 2xx（201 也算）→ 解析 body 里的 id / name / qty 返回 Item
        """
        raise NotImplementedError("TODO: 补完 create_item")

    def list_items(self) -> list[Item]:
        """GET /items，响应 200 返回 Item 列表.

        响应 body 是 list[dict]：`[{"id": int, "name": str, "qty": int}, ...]`
        """
        raise NotImplementedError("TODO: 补完 list_items")
