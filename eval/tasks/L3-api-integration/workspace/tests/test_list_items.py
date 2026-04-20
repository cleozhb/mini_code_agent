from __future__ import annotations

import pytest

from app.api_client import ApiError, Item
from app.http import HttpResponse


def test_list_items_ok(client, transport) -> None:
    transport.responses[("GET", "/items")] = [
        HttpResponse(200, [
            {"id": 1, "name": "A", "qty": 2},
            {"id": 2, "name": "B", "qty": 5},
        ])
    ]
    assert client.list_items() == [Item(1, "A", 2), Item(2, "B", 5)]


def test_list_items_empty(client, transport) -> None:
    transport.responses[("GET", "/items")] = [HttpResponse(200, [])]
    assert client.list_items() == []


def test_list_items_5xx_retries(client, transport) -> None:
    transport.responses[("GET", "/items")] = [
        HttpResponse(500, "down"),
        HttpResponse(200, []),
    ]
    assert client.list_items() == []
    assert len(transport.calls) == 2


def test_list_items_4xx_raises(client, transport) -> None:
    transport.responses[("GET", "/items")] = [HttpResponse(403, "forbidden")]
    with pytest.raises(ApiError) as ei:
        client.list_items()
    assert ei.value.status == 403
    assert len(transport.calls) == 1
