"""L2-print-to-logger 的行为契约 —— Agent 替换 print→logger 后必须保持通过.

测试只断言**返回值和状态**，不断言 print 的 stdout 输出，这样把 print 换成
logger 调用时行为不变；但 Agent 如果顺手改了 return 值/状态就会被测出来。
"""

from __future__ import annotations

import pytest

from app.service import CartService
from app.utils import format_price, parse_sku


class TestCartService:
    def test_add_item_normal(self) -> None:
        cart = CartService()
        assert cart.add_item("A001", 2) is True
        assert cart.items == {"A001": 2}

    def test_add_item_accumulates_quantity(self) -> None:
        cart = CartService()
        cart.add_item("A001", 2)
        cart.add_item("A001", 3)
        assert cart.items["A001"] == 5

    def test_add_item_rejects_non_positive_quantity(self) -> None:
        cart = CartService()
        assert cart.add_item("A001", 0) is False
        assert cart.add_item("A001", -1) is False
        assert cart.items == {}

    def test_add_item_on_closed_cart_fails(self) -> None:
        cart = CartService()
        cart.add_item("A", 1)
        cart.checkout()
        assert cart.add_item("B", 1) is False
        assert "B" not in cart.items

    def test_remove_item_normal(self) -> None:
        cart = CartService()
        cart.add_item("A", 1)
        assert cart.remove_item("A") is True
        assert cart.items == {}

    def test_remove_unknown_item(self) -> None:
        cart = CartService()
        assert cart.remove_item("nope") is False

    def test_checkout_empty_fails(self) -> None:
        cart = CartService()
        assert cart.checkout() is None
        assert cart.closed is False

    def test_checkout_returns_snapshot_and_closes(self) -> None:
        cart = CartService()
        cart.add_item("A", 2)
        cart.add_item("B", 1)
        snap = cart.checkout()
        assert snap == {"A": 2, "B": 1}
        assert cart.closed is True

    def test_double_checkout_fails(self) -> None:
        cart = CartService()
        cart.add_item("A", 1)
        cart.checkout()
        assert cart.checkout() is None


class TestUtils:
    def test_format_price_basic(self) -> None:
        assert format_price(99.5) == "99.50 CNY"

    def test_format_price_custom_currency(self) -> None:
        assert format_price(12.345, "USD") == "12.35 USD"

    def test_format_price_negative(self) -> None:
        # 负数允许，只会发一条 warning（原来是 print，替换后是 logger.warning）
        assert format_price(-1.0) == "-1.00 CNY"

    def test_parse_sku_normalizes(self) -> None:
        assert parse_sku("  abc123 ") == "ABC123"

    def test_parse_sku_empty(self) -> None:
        assert parse_sku("   ") is None

    def test_parse_sku_non_alphanumeric(self) -> None:
        assert parse_sku("abc-123") is None
