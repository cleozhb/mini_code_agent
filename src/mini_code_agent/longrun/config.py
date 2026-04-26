"""LongRunConfig — 长程任务的运行时配置."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class LongRunConfig:
    """长程任务的运行时配置.

    控制 checkpoint 策略、token 预算等。
    """

    # === Token 预算 ===
    token_budget: int = 500_000

    # === Checkpoint 策略 ===
    checkpoint_interval_tokens: int = 50_000  # 距上次 checkpoint 的 token 增量阈值
    checkpoint_on_subtask_complete: bool = True  # 子任务完成后自动 checkpoint
    max_checkpoints: int = 10  # 每个任务保留的最大 checkpoint 数

    # === 超时 ===
    max_wall_time_seconds: float = 3600.0  # 墙钟时间上限（秒）

    def to_dict(self) -> dict:
        """序列化为 JSON 兼容的 dict."""
        return {
            "token_budget": self.token_budget,
            "checkpoint_interval_tokens": self.checkpoint_interval_tokens,
            "checkpoint_on_subtask_complete": self.checkpoint_on_subtask_complete,
            "max_checkpoints": self.max_checkpoints,
            "max_wall_time_seconds": self.max_wall_time_seconds,
        }

    @classmethod
    def from_dict(cls, d: dict) -> LongRunConfig:
        """从 JSON dict 反序列化."""
        return cls(
            token_budget=d.get("token_budget", 500_000),
            checkpoint_interval_tokens=d.get("checkpoint_interval_tokens", 50_000),
            checkpoint_on_subtask_complete=d.get("checkpoint_on_subtask_complete", True),
            max_checkpoints=d.get("max_checkpoints", 10),
            max_wall_time_seconds=d.get("max_wall_time_seconds", 3600.0),
        )
