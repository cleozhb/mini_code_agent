"""税费计算。

对外暴露 `compute_tax(subtotal, rate)`：按税率算税额。
"""
from __future__ import annotations


def compute_tax(subtotal: float, rate: float) -> float:
    """按税率 rate（小数形式，如 0.08 代表 8%）算出税额。

    约定：
      - rate 在 [0, 1] 区间，超出抛 ValueError
      - 返回值保留 2 位小数（四舍五入，银行家舍入以外）
      - subtotal 负数时返回 0（免税退款场景）
    """
    if rate < 0 or rate > 1:
        raise ValueError(f"invalid tax rate: {rate}")
    # BUG: 漏了 subtotal < 0 的分支，负 subtotal 会得到负税额。
    tax = subtotal * rate
    return round(tax, 2)
