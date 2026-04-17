from __future__ import annotations


def add(a: float, b: float) -> float:
    return a + b


def subtract(a: float, b: float) -> float:
    return a - b


def multiply(a: float, b: float) -> float:
    return a * b


def divide(a: float, b: float) -> float:
    if b == 0:
        raise ValueError("除数不能为 0")
    return a / b


def clamp(value: float, low: float, high: float) -> float:
    if low > high:
        raise ValueError("low 不能大于 high")
    if value < low:
        return low
    if value > high:
        return high
    return value
