"""Order processor —— 把一个臃肿的 process_order 函数等着 Agent 来拆."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class LineItem:
    sku: str
    unit_price: float
    quantity: int


@dataclass
class Order:
    order_id: str
    customer_tier: str  # "standard" / "gold" / "vip"
    items: list[LineItem]
    coupon_code: str | None = None
    region: str = "CN"
    # 以下字段在 process_order 里被逐步填充
    subtotal: float = 0.0
    discount: float = 0.0
    tax: float = 0.0
    shipping: float = 0.0
    total: float = 0.0
    status: str = "pending"
    audit_log: list[str] = field(default_factory=list)


# 预置税率和折扣规则（Agent 拆分时不能改这些常量的数值）
TAX_RATES = {"CN": 0.13, "US": 0.07, "EU": 0.20}
TIER_DISCOUNTS = {"standard": 0.0, "gold": 0.05, "vip": 0.12}
COUPON_DISCOUNTS = {"SAVE10": 0.10, "SAVE20": 0.20, "FREESHIP": 0.0}
SHIPPING_BASE = {"CN": 10.0, "US": 20.0, "EU": 25.0}


class OrderError(Exception):
    pass


def process_order(order: Order) -> Order:
    """处理一笔订单：校验 → 小计 → 折扣 → 税金 → 运费 → 总价 → 结算.

    这是一个故意写得又长又糙的函数；任务目标是把它拆成职责清晰的小函数，
    保持外部行为（返回值、抛错、audit_log 内容顺序）完全一致。
    """
    # ---- 1. 入参校验 ------------------------------------------------------
    if not order.order_id or not isinstance(order.order_id, str):
        raise OrderError("order_id 必填")
    if order.customer_tier not in TIER_DISCOUNTS:
        raise OrderError(f"未知 customer_tier: {order.customer_tier!r}")
    if order.region not in TAX_RATES:
        raise OrderError(f"未知 region: {order.region!r}")
    if not order.items:
        raise OrderError("订单必须至少包含一件商品")
    for idx, item in enumerate(order.items):
        if not item.sku:
            raise OrderError(f"第 {idx} 件商品缺少 sku")
        if item.unit_price < 0:
            raise OrderError(f"sku={item.sku} 的 unit_price 不能为负")
        if item.quantity <= 0:
            raise OrderError(f"sku={item.sku} 的 quantity 必须为正数")
    order.audit_log.append(f"validated:{order.order_id}")

    # ---- 2. 小计 ----------------------------------------------------------
    subtotal = 0.0
    for item in order.items:
        line_total = item.unit_price * item.quantity
        subtotal += line_total
    subtotal = round(subtotal, 2)
    order.subtotal = subtotal
    order.audit_log.append(f"subtotal:{subtotal:.2f}")

    # ---- 3. 折扣（会员等级 + 优惠券）------------------------------------
    tier_rate = TIER_DISCOUNTS[order.customer_tier]
    coupon_rate = 0.0
    if order.coupon_code is not None:
        if order.coupon_code not in COUPON_DISCOUNTS:
            raise OrderError(f"无效优惠券: {order.coupon_code!r}")
        coupon_rate = COUPON_DISCOUNTS[order.coupon_code]
    # 两种折扣叠加，但总折扣上限 25%
    combined_rate = tier_rate + coupon_rate
    if combined_rate > 0.25:
        combined_rate = 0.25
    discount = round(subtotal * combined_rate, 2)
    order.discount = discount
    order.audit_log.append(
        f"discount:{discount:.2f}(tier={tier_rate},coupon={coupon_rate})"
    )

    # ---- 4. 税金 ----------------------------------------------------------
    taxable = subtotal - discount
    if taxable < 0:
        taxable = 0.0
    tax_rate = TAX_RATES[order.region]
    tax = round(taxable * tax_rate, 2)
    order.tax = tax
    order.audit_log.append(f"tax:{tax:.2f}(rate={tax_rate})")

    # ---- 5. 运费 ----------------------------------------------------------
    shipping = SHIPPING_BASE[order.region]
    # 大订单减免运费
    if taxable >= 200.0:
        shipping = shipping * 0.5
    # FREESHIP 券免运费
    if order.coupon_code == "FREESHIP":
        shipping = 0.0
    shipping = round(shipping, 2)
    order.shipping = shipping
    order.audit_log.append(f"shipping:{shipping:.2f}")

    # ---- 6. 总价与结算 ---------------------------------------------------
    total = round(taxable + tax + shipping, 2)
    order.total = total
    order.audit_log.append(f"total:{total:.2f}")
    order.status = "processed"
    order.audit_log.append(f"status:{order.status}")
    return order
