from __future__ import annotations

import pytest

from app.api_client import ApiError, Item
from app.http import HttpResponse


def test_create_item_ok(client, transport) -> None:
    transport.responses[("POST", "/items")] = [
        HttpResponse(201, {"id": 7, "name": "X", "qty": 3})
    ]
    it = client.create_item("X", 3)
    assert it == Item(id=7, name="X", qty=3)
    # request body 正确透传
    assert transport.calls == [("POST", "/items", {"name": "X", "qty": 3})]


def test_create_item_4xx_raises_no_retry(client, transport) -> None:
    transport.responses[("POST", "/items")] = [HttpResponse(400, "empty name")]
    with pytest.raises(ApiError) as ei:
        client.create_item("", 3)
    assert ei.value.status == 400
    assert len(transport.calls) == 1


def test_create_item_5xx_retries_then_succeeds(client, transport) -> None:
    transport.responses[("POST", "/items")] = [
        HttpResponse(503, "down"),
        HttpResponse(500, "still down"),
        HttpResponse(201, {"id": 1, "name": "X", "qty": 1}),
    ]
    it = client.create_item("X", 1)
    assert it.id == 1
    assert len(transport.calls) == 3


def test_create_item_5xx_gives_up_after_max_retries(client, transport) -> None:
    # 原请求 1 + 重试 MAX_RETRIES(=3) = 4 次
    transport.responses[("POST", "/items")] = [HttpResponse(500, "down")] * 4
    with pytest.raises(ApiError) as ei:
        client.create_item("X", 1)
    assert ei.value.status == 500
    assert len(transport.calls) == 4
