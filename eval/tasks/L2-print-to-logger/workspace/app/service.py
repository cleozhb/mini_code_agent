"""购物车服务 —— 里面用 print() 打日志，等着 Agent 替换成 logger 调用."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CartService:
    items: dict[str, int] = field(default_factory=dict)  # sku -> quantity
    closed: bool = False

    def add_item(self, sku: str, quantity: int = 1) -> bool:
        if self.closed:
            print(f"ERROR: cart is closed, cannot add {sku}")
            return False
        if quantity <= 0:
            print(f"WARNING: ignoring non-positive quantity={quantity} for {sku}")
            return False
        self.items[sku] = self.items.get(sku, 0) + quantity
        print(f"INFO: added sku={sku} qty={quantity}, cart now has {len(self.items)} skus")
        return True

    def remove_item(self, sku: str) -> bool:
        if sku not in self.items:
            print(f"WARNING: cannot remove {sku}, not in cart")
            return False
        del self.items[sku]
        print(f"INFO: removed sku={sku}")
        return True

    def checkout(self) -> dict[str, int] | None:
        if self.closed:
            print("ERROR: cart already checked out")
            return None
        if not self.items:
            print("ERROR: cannot checkout empty cart")
            return None
        snapshot = dict(self.items)
        self.closed = True
        print(f"INFO: checkout successful, items={len(snapshot)}")
        return snapshot
