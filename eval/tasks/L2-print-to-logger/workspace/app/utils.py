"""辅助函数 —— 也有几处 print() 要替换."""

from __future__ import annotations


def format_price(amount: float, currency: str = "CNY") -> str:
    if amount < 0:
        print(f"WARNING: negative amount={amount}, formatting anyway")
    return f"{amount:.2f} {currency}"


def parse_sku(raw: str) -> str | None:
    raw = raw.strip().upper()
    if not raw:
        print("ERROR: empty sku after strip")
        return None
    if not raw.isalnum():
        print(f"ERROR: sku has non-alphanumeric chars: {raw!r}")
        return None
    print(f"INFO: parsed sku={raw}")
    return raw
