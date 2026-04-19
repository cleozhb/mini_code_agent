"""L2-split-large-function 的行为契约。

Agent 拆分 process_order 时不能改这些测试。
测试断言的是**外部行为**：返回值、抛错、audit_log 顺序 —— 全部应保持不变。
"""

from __future__ import annotations

import pytest

from order_processor import (
    COUPON_DISCOUNTS,
    LineItem,
    Order,
    OrderError,
    SHIPPING_BASE,
    TAX_RATES,
    TIER_DISCOUNTS,
    process_order,
)


def _mk_order(**overrides) -> Order:
    defaults = dict(
        order_id="ORD-001",
        customer_tier="standard",
        items=[LineItem(sku="A", unit_price=100.0, quantity=1)],
        coupon_code=None,
        region="CN",
    )
    defaults.update(overrides)
    return Order(**defaults)


# ---------------------------------------------------------------------------
# 入参校验
# ---------------------------------------------------------------------------


class TestValidation:
    def test_rejects_empty_order_id(self) -> None:
        with pytest.raises(OrderError, match="order_id"):
            process_order(_mk_order(order_id=""))

    def test_rejects_unknown_tier(self) -> None:
        with pytest.raises(OrderError, match="customer_tier"):
            process_order(_mk_order(customer_tier="platinum"))

    def test_rejects_unknown_region(self) -> None:
        with pytest.raises(OrderError, match="region"):
            process_order(_mk_order(region="JP"))

    def test_rejects_empty_items(self) -> None:
        with pytest.raises(OrderError, match="至少"):
            process_order(_mk_order(items=[]))

    def test_rejects_negative_price(self) -> None:
        with pytest.raises(OrderError, match="unit_price"):
            process_order(
                _mk_order(items=[LineItem(sku="A", unit_price=-1.0, quantity=1)])
            )

    def test_rejects_zero_quantity(self) -> None:
        with pytest.raises(OrderError, match="quantity"):
            process_order(
                _mk_order(items=[LineItem(sku="A", unit_price=10.0, quantity=0)])
            )

    def test_rejects_invalid_coupon(self) -> None:
        with pytest.raises(OrderError, match="优惠券"):
            process_order(_mk_order(coupon_code="FAKE"))


# ---------------------------------------------------------------------------
# 数值计算
# ---------------------------------------------------------------------------


class TestPricing:
    def test_standard_cn_no_coupon(self) -> None:
        # 1 件 100 块，standard 无折扣，CN 13% 税，CN 运费 10
        o = process_order(_mk_order())
        assert o.subtotal == 100.0
        assert o.discount == 0.0
        assert o.tax == 13.0  # 100 * 0.13
        assert o.shipping == 10.0
        assert o.total == 123.0

    def test_gold_with_save10(self) -> None:
        # 2 件各 50 块 = 100，gold 5% + SAVE10 10% = 15% 折扣 = 15
        # 税基 85，CN 13% tax = 11.05；运费 10（未到 200）；total = 85 + 11.05 + 10 = 106.05
        o = process_order(
            _mk_order(
                customer_tier="gold",
                items=[LineItem(sku="A", unit_price=50.0, quantity=2)],
                coupon_code="SAVE10",
            )
        )
        assert o.subtotal == 100.0
        assert o.discount == 15.0
        assert o.tax == pytest.approx(11.05)
        assert o.shipping == 10.0
        assert o.total == pytest.approx(106.05)

    def test_vip_with_save20_caps_at_25pct(self) -> None:
        # vip 12% + SAVE20 20% = 32%，上限 25%；subtotal=100 → discount=25
        o = process_order(
            _mk_order(
                customer_tier="vip",
                coupon_code="SAVE20",
            )
        )
        assert o.discount == 25.0

    def test_large_order_halves_shipping(self) -> None:
        # subtotal=300 触发 taxable>=200 → 运费减半
        o = process_order(
            _mk_order(items=[LineItem(sku="A", unit_price=150.0, quantity=2)])
        )
        assert o.shipping == 5.0  # 10 * 0.5

    def test_freeship_coupon(self) -> None:
        o = process_order(_mk_order(coupon_code="FREESHIP"))
        assert o.shipping == 0.0

    def test_us_region_rates(self) -> None:
        o = process_order(_mk_order(region="US"))
        assert o.tax == pytest.approx(100 * TAX_RATES["US"])
        assert o.shipping == SHIPPING_BASE["US"]


# ---------------------------------------------------------------------------
# audit_log 顺序（外部可观测行为的一部分，拆分不能破坏）
# ---------------------------------------------------------------------------


class TestAuditLog:
    def test_log_sequence_is_stable(self) -> None:
        o = process_order(_mk_order(coupon_code="SAVE10"))
        stages = [entry.split(":", 1)[0] for entry in o.audit_log]
        assert stages == [
            "validated",
            "subtotal",
            "discount",
            "tax",
            "shipping",
            "total",
            "status",
        ]

    def test_status_becomes_processed(self) -> None:
        o = process_order(_mk_order())
        assert o.status == "processed"
        assert any(e.startswith("status:processed") for e in o.audit_log)

    def test_tax_and_discount_are_present_in_log(self) -> None:
        o = process_order(_mk_order(customer_tier="gold"))
        assert any(e.startswith("discount:") for e in o.audit_log)
        assert any(e.startswith("tax:") for e in o.audit_log)


# ---------------------------------------------------------------------------
# 常量没被改（防止 Agent 为了过测试篡改 TIER_DISCOUNTS 等值）
# ---------------------------------------------------------------------------


class TestConstantsUnchanged:
    def test_tier_discounts_values(self) -> None:
        assert TIER_DISCOUNTS == {"standard": 0.0, "gold": 0.05, "vip": 0.12}

    def test_tax_rates_values(self) -> None:
        assert TAX_RATES == {"CN": 0.13, "US": 0.07, "EU": 0.20}

    def test_coupon_discounts_values(self) -> None:
        assert COUPON_DISCOUNTS == {
            "SAVE10": 0.10,
            "SAVE20": 0.20,
            "FREESHIP": 0.0,
        }
