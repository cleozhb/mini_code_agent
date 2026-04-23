"""资源消耗 — 记录 Worker 执行过程中消耗的资源."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ResourceUsage:
    """Worker 执行过程中消耗的资源."""

    tokens_input: int
    tokens_output: int
    tokens_total: int
    llm_calls: int  # LLM API 调用次数
    tool_calls: int  # 工具调用次数
    wall_time_seconds: float
    model_used: str  # "claude-sonnet-4-5" 等
