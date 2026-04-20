import pytest

from app.tax import compute_tax


def test_compute_tax_basic() -> None:
    assert compute_tax(100.0, 0.08) == 8.0


def test_compute_tax_zero_rate() -> None:
    assert compute_tax(100.0, 0.0) == 0.0


def test_compute_tax_rounds_to_two_decimals() -> None:
    # 12.345 * 0.08 = 0.9876 → 0.99
    assert compute_tax(12.345, 0.08) == 0.99


def test_compute_tax_rejects_invalid_rate() -> None:
    with pytest.raises(ValueError):
        compute_tax(100.0, -0.01)
    with pytest.raises(ValueError):
        compute_tax(100.0, 1.5)


def test_compute_tax_refund_scenario_returns_zero() -> None:
    """负 subtotal（退款场景）税额应为 0，不能反向给负税（= 额外退税）。"""
    assert compute_tax(-50.0, 0.08) == 0.0
