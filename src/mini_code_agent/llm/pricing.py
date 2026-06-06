"""LLM token 用量成本估算."""

from __future__ import annotations

from dataclasses import dataclass, field

from .base import TokenUsage


@dataclass(frozen=True)
class ModelPricing:
    """每百万 token 美元价格."""

    input: float | None = None
    output: float | None = None
    cached_input: float | None = None
    reasoning: float | None = None
    cache_creation: float | None = None
    cache_read: float | None = None


@dataclass(frozen=True)
class CostEstimate:
    """一次 usage 的成本估算结果."""

    model: str
    total_cost: float
    input_cost: float = 0.0
    output_cost: float = 0.0
    cached_input_cost: float = 0.0
    reasoning_cost: float = 0.0
    cache_creation_cost: float = 0.0
    cache_read_cost: float = 0.0
    missing_price_items: list[str] = field(default_factory=list)

    @property
    def is_complete(self) -> bool:
        return not self.missing_price_items


_PRICE_TABLE: dict[str, ModelPricing] = {
    "gpt-4o": ModelPricing(input=2.5, output=10.0, cached_input=1.25),
    "gpt-4o-mini": ModelPricing(input=0.15, output=0.6, cached_input=0.075),
    "gpt-4-turbo": ModelPricing(input=10.0, output=30.0),
    "claude-sonnet-4-6": ModelPricing(
        input=3.0,
        output=15.0,
        cache_creation=3.75,
        cache_read=0.30,
    ),
    "claude-sonnet-4-20250514": ModelPricing(
        input=3.0,
        output=15.0,
        cache_creation=3.75,
        cache_read=0.30,
    ),
    "claude-opus-4-20250514": ModelPricing(
        input=15.0,
        output=75.0,
        cache_creation=18.75,
        cache_read=1.50,
    ),
    "deepseek-chat": ModelPricing(input=0.27, output=1.10),
    "deepseek-reasoner": ModelPricing(input=0.55, output=2.19),
}


def _pricing_for(model: str) -> ModelPricing | None:
    prices = _PRICE_TABLE.get(model)
    if prices:
        return prices
    for key, value in _PRICE_TABLE.items():
        if model.startswith(key):
            return value
    return None


def _line_cost(tokens: int, price_per_1m: float | None, missing_name: str) -> tuple[float, str | None]:
    if tokens <= 0:
        return 0.0, None
    if price_per_1m is None:
        return 0.0, missing_name
    return tokens / 1_000_000 * price_per_1m, None


def estimate_cost(model: str, usage: TokenUsage) -> CostEstimate | None:
    """估算费用（美元）.

    价格表不是实时官方报价，只用于把 usage 字段结构化计入估算。
    未知模型或缺少某个价格项时返回部分估算，并在 missing_price_items 里说明。
    """
    prices = _pricing_for(model)
    missing: list[str] = []

    if prices is None:
        for tokens, name in [
            (usage.input_tokens, "input"),
            (usage.output_tokens, "output"),
            (usage.cached_input_tokens, "cached_input"),
            (usage.reasoning_tokens, "reasoning"),
            (usage.cache_creation_input_tokens, "cache_creation"),
            (usage.cache_read_input_tokens, "cache_read"),
        ]:
            if tokens > 0:
                missing.append(name)
        return CostEstimate(model=model, total_cost=0.0, missing_price_items=missing)

    cached_input_tokens = min(usage.cached_input_tokens, usage.input_tokens)
    regular_input_tokens = usage.input_tokens - cached_input_tokens

    input_cost, miss = _line_cost(regular_input_tokens, prices.input, "input")
    if miss:
        missing.append(miss)

    cached_input_cost, miss = _line_cost(
        cached_input_tokens, prices.cached_input, "cached_input"
    )
    if miss:
        missing.append(miss)

    if usage.reasoning_tokens > 0 and prices.reasoning is not None:
        visible_output_tokens = max(usage.output_tokens - usage.reasoning_tokens, 0)
        output_cost, miss = _line_cost(visible_output_tokens, prices.output, "output")
        if miss:
            missing.append(miss)
        reasoning_cost, miss = _line_cost(
            usage.reasoning_tokens, prices.reasoning, "reasoning"
        )
        if miss:
            missing.append(miss)
    else:
        output_cost, miss = _line_cost(usage.output_tokens, prices.output, "output")
        reasoning_cost = 0.0
        if miss:
            missing.append(miss)

    cache_creation_cost, miss = _line_cost(
        usage.cache_creation_input_tokens,
        prices.cache_creation,
        "cache_creation",
    )
    if miss:
        missing.append(miss)

    cache_read_cost, miss = _line_cost(
        usage.cache_read_input_tokens,
        prices.cache_read,
        "cache_read",
    )
    if miss:
        missing.append(miss)

    total = (
        input_cost
        + output_cost
        + cached_input_cost
        + reasoning_cost
        + cache_creation_cost
        + cache_read_cost
    )
    return CostEstimate(
        model=model,
        total_cost=total,
        input_cost=input_cost,
        output_cost=output_cost,
        cached_input_cost=cached_input_cost,
        reasoning_cost=reasoning_cost,
        cache_creation_cost=cache_creation_cost,
        cache_read_cost=cache_read_cost,
        missing_price_items=missing,
    )
