"""FakeTransport + no-sleep fixture，精确控制响应序列 & 不真的睡。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from app.http import HttpClient, HttpResponse


@dataclass
class FakeTransport:
    """按 (method, url) 查表返回响应；每调一次弹出队列首项。

    没有预设响应的 (method, url) 组合默认返回 200 + body=None（避免"忘设响应"
    时在 test 内部以 KeyError 炸掉、误导排查）。同时全部调用记在 `calls` 里
    供断言。
    """

    responses: dict[tuple[str, str], list[HttpResponse]] = field(default_factory=dict)
    calls: list[tuple[str, str, Any]] = field(default_factory=list)

    def request(self, method: str, url: str, json: Any = None) -> HttpResponse:
        self.calls.append((method, url, json))
        queue = self.responses.get((method, url))
        if queue:
            return queue.pop(0)
        return HttpResponse(200, None)


@pytest.fixture
def transport() -> FakeTransport:
    return FakeTransport()


@pytest.fixture
def client(transport: FakeTransport):
    from app.api_client import ApiClient

    return ApiClient(HttpClient(base_url="", transport=transport))


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """重试测试不真的睡，保证 CI 跑得快 + 确定性."""
    import app.api_client as m

    monkeypatch.setattr(m, "_SLEEP", lambda _d: None)
