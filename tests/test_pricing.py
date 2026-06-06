"""LLM 成本估算测试."""

from __future__ import annotations

from mini_code_agent.llm.base import TokenUsage
from mini_code_agent.llm.pricing import estimate_cost


def test_estimate_basic_input_output_cost():
    estimate = estimate_cost(
        "gpt-4o",
        TokenUsage(input_tokens=1_000_000, output_tokens=1_000_000),
    )

    assert estimate is not None
    assert estimate.total_cost == 12.5
    assert estimate.is_complete


def test_estimate_openai_cached_input_cost():
    estimate = estimate_cost(
        "gpt-4o",
        TokenUsage(input_tokens=1_000_000, cached_input_tokens=400_000),
    )

    assert estimate is not None
    assert estimate.input_cost == 1.5
    assert estimate.cached_input_cost == 0.5
    assert estimate.total_cost == 2.0


def test_estimate_openai_reasoning_tokens_are_covered_by_output_when_no_split_price():
    estimate = estimate_cost(
        "gpt-4o",
        TokenUsage(output_tokens=1_000_000, reasoning_tokens=250_000),
    )

    assert estimate is not None
    assert estimate.output_cost == 10.0
    assert estimate.reasoning_cost == 0.0
    assert estimate.is_complete


def test_estimate_anthropic_cache_creation_and_read():
    estimate = estimate_cost(
        "claude-sonnet-4-6",
        TokenUsage(
            input_tokens=1_000_000,
            output_tokens=1_000_000,
            cache_creation_input_tokens=1_000_000,
            cache_read_input_tokens=1_000_000,
        ),
    )

    assert estimate is not None
    assert estimate.input_cost == 3.0
    assert estimate.output_cost == 15.0
    assert estimate.cache_creation_cost == 3.75
    assert estimate.cache_read_cost == 0.30
    assert estimate.total_cost == 22.05


def test_estimate_unknown_model_returns_missing_items():
    estimate = estimate_cost(
        "unknown-model",
        TokenUsage(input_tokens=10, output_tokens=20),
    )

    assert estimate is not None
    assert estimate.total_cost == 0.0
    assert estimate.missing_price_items == ["input", "output"]


def test_estimate_partial_price_missing_for_cached_tokens():
    estimate = estimate_cost(
        "deepseek-chat",
        TokenUsage(input_tokens=100, cached_input_tokens=50),
    )

    assert estimate is not None
    assert estimate.input_cost > 0
    assert estimate.cached_input_cost == 0.0
    assert estimate.missing_price_items == ["cached_input"]
