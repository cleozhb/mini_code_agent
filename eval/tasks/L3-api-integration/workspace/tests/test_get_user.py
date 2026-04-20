"""已实现端点的行为冒烟（示范 4xx/5xx 的处理模式）."""

from __future__ import annotations

import pytest

from app.api_client import ApiError, User
from app.http import HttpResponse


def test_get_user_ok(client, transport) -> None:
    transport.responses[("GET", "/users/1")] = [
        HttpResponse(200, {"id": 1, "name": "Alice"})
    ]
    assert client.get_user(1) == User(id=1, name="Alice")


def test_get_user_4xx_raises_no_retry(client, transport) -> None:
    transport.responses[("GET", "/users/404")] = [HttpResponse(404, "not found")]
    with pytest.raises(ApiError) as ei:
        client.get_user(404)
    assert ei.value.status == 404
    assert len(transport.calls) == 1


def test_get_user_5xx_retries_then_succeeds(client, transport) -> None:
    transport.responses[("GET", "/users/7")] = [
        HttpResponse(503, "down"),
        HttpResponse(500, "still down"),
        HttpResponse(200, {"id": 7, "name": "G"}),
    ]
    assert client.get_user(7).id == 7
    assert len(transport.calls) == 3
