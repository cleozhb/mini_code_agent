"""决策记录 — Worker 在执行过程中做的关键选择."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ArtifactDecision:
    """Worker 在执行过程中做的关键选择.

    Decision 不是 Worker "想到就写"，而是由 Agent 通过以下启发式
    自动提取 + Worker 显式标注两种方式收集：
      - 启发式：创建新文件、引入新依赖、选择算法/数据结构时自动记录
      - 显式：Worker 可调用 record_decision 工具主动标注
    """

    description: str  # "选择使用 dataclass 而不是 Pydantic"
    reason: str  # Worker 给出的理由
    alternatives_considered: list[str] = field(default_factory=list)  # 考虑过的其他方案
    reversible: bool = True  # 是否容易回退
    step_number: int = 0  # 在 Worker 的第几步做的决定
