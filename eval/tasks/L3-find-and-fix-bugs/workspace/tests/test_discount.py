import pytest

from app.discount import apply_discount


def test_apply_discount_basic() -> None:
    assert apply_discount(100.0, 10) == 90.0


def test_apply_discount_zero() -> None:
    assert apply_discount(100.0, 0) == 100.0


def test_apply_discount_full() -> None:
    assert apply_discount(100.0, 100) == 0.0


def test_apply_discount_rounds_to_two_decimals() -> None:
    # 33.33% off of 10.00 = 6.667 → 6.67
    assert apply_discount(10.0, 33.33) == 6.67


@pytest.mark.parametrize("bad_pct", [-5, -0.1, 100.01, 120])
def test_apply_discount_rejects_out_of_range_pct(bad_pct: float) -> None:
    """折扣百分比必须在 [0, 100] 内：负值会"涨价"、>100 会给负价，都得拒绝。"""
    with pytest.raises(ValueError):
        apply_discount(100.0, bad_pct)
