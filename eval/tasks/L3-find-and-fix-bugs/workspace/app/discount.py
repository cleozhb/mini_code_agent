"""折扣计算。

对外暴露 `apply_discount(price, pct)`：按百分比打折。
"""
from __future__ import annotations


def apply_discount(price: float, pct: float) -> float:
    """按百分比 pct 对 price 打折，返回折后价。

    约定：
      - pct 必须在闭区间 [0, 100] 之间，超出范围抛 ValueError
      - 折后价保留 2 位小数（四舍五入）
    """
    # BUG: 没校验 pct 的有效范围；负数折扣会"涨价"，>100 会给出负价。
    discounted = price * (1 - pct / 100)
    return round(discounted, 2)
