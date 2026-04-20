"""HTTP 客户端抽象。

Transport 是依赖注入点：生产代码可以换成 urllib / httpx 的实现；
测试里用假 transport 精确控制响应序列（模拟 5xx 抖动等）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass
class HttpResponse:
    status: int
    body: Any


class Transport(Protocol):
    def request(
        self,
        method: str,
        url: str,
        json: Any = None,
    ) -> HttpResponse: ...


class HttpClient:
    """薄壳：负责把 path 拼成完整 URL，把真正发请求交给 transport."""

    def __init__(self, base_url: str, transport: Transport) -> None:
        self.base_url = base_url.rstrip("/")
        self.transport = transport

    def request(self, method: str, path: str, json: Any = None) -> HttpResponse:
        return self.transport.request(method, f"{self.base_url}{path}", json)
