"""购物车结算。

对外暴露 `checkout(items, discount_pct, tax_rate)`：
  items: list[(name, unit_price, qty)]
  先算 subtotal，再按 discount_pct 打折，再按 tax_rate 加税，最后返回 total。
"""
from __future__ import annotations

from app.discount import apply_discount
from app.tax import compute_tax


def checkout(
    items: list[tuple[str, float, int]],
    discount_pct: float,
    tax_rate: float,
) -> float:
    """返回最终结算金额，保留 2 位小数。

    计算顺序：subtotal → 折扣 → 加税 → 返回 total。
    空购物车 → 0.0。
    """
    if not items:
        return 0.0
    subtotal = sum(unit * qty for _name, unit, qty in items)
    discounted = apply_discount(subtotal, discount_pct)
    tax = compute_tax(discounted, tax_rate)
    return round(discounted + tax, 2)
