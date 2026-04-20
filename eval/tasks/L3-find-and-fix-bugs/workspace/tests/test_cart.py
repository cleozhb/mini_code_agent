from app.cart import checkout


def test_checkout_empty() -> None:
    assert checkout([], 10, 0.08) == 0.0


def test_checkout_basic() -> None:
    # 2 * 10 + 1 * 30 = 50; 10% off → 45; tax 8% → 3.6; total 48.6
    items = [("A", 10.0, 2), ("B", 30.0, 1)]
    assert checkout(items, 10, 0.08) == 48.6


def test_checkout_no_discount_no_tax() -> None:
    items = [("A", 5.0, 3)]
    assert checkout(items, 0, 0.0) == 15.0
