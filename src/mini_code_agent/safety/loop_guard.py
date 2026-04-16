"""循环保护器 — 防止 Agent 无限循环、重复操作、超出 token 预算."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field


@dataclass
class ToolCallRecord:
    """记录一次工具调用（用于重复检测）."""

    tool_name: str
    args_hash: str  # 参数的哈希，用于快速比较


class LoopGuard:
    """循环保护器.

    三重保护：
    1. 最大轮数限制
    2. 连续重复调用检测
    3. Token 预算控制
    """

    def __init__(
        self,
        max_rounds: int = 30,
        repeat_threshold: int = 3,
        max_tokens: int = 200_000,
        token_warning_ratio: float = 0.8,
    ) -> None:
        self.max_rounds = max_rounds
        self.repeat_threshold = repeat_threshold
        self.max_tokens = max_tokens
        self.token_warning_ratio = token_warning_ratio

        self._current_round: int = 0
        self._total_tokens: int = 0
        self._recent_calls: list[ToolCallRecord] = []

    # ----- 轮数控制 -----

    def next_round(self) -> str | None:
        """进入下一轮，返回限制消息或 None.

        Returns:
            None 表示正常继续，str 表示被限制（消息内容告知 LLM）
        """
        self._current_round += 1
        if self._current_round > self.max_rounds:
            return (
                f"[安全限制] 已达到最大工具调用轮数 ({self.max_rounds})。"
                "请根据已有信息直接回答，不要再调用工具。"
            )
        return None

    @property
    def current_round(self) -> int:
        return self._current_round

    # ----- 重复检测 -----

    def record_tool_call(self, tool_name: str, arguments: dict) -> str | None:
        """记录工具调用并检测重复.

        Returns:
            None 正常，str 警告消息
        """
        args_hash = self._hash_args(arguments)
        record = ToolCallRecord(tool_name=tool_name, args_hash=args_hash)
        self._recent_calls.append(record)

        # 检查最近 N 次是否都是相同调用
        if len(self._recent_calls) >= self.repeat_threshold:
            recent = self._recent_calls[-self.repeat_threshold:]
            if all(
                r.tool_name == recent[0].tool_name
                and r.args_hash == recent[0].args_hash
                for r in recent
            ):
                return (
                    f"[安全警告] 检测到连续 {self.repeat_threshold} 次相同的工具调用 "
                    f"({tool_name})。请检查是否陷入循环，尝试换一种方法。"
                )
        return None

    # ----- Token 预算 -----

    def add_tokens(self, tokens: int) -> str | None:
        """累加 token 消耗，检查预算.

        Returns:
            None 正常，str 警告/限制消息
        """
        self._total_tokens += tokens
        warning_threshold = int(self.max_tokens * self.token_warning_ratio)

        if self._total_tokens >= self.max_tokens:
            return (
                f"[安全限制] 已超出 token 预算 ({self._total_tokens}/{self.max_tokens})。"
                "请立即给出最终回答，不要再调用工具。"
            )
        if self._total_tokens >= warning_threshold:
            remaining = self.max_tokens - self._total_tokens
            return (
                f"[安全警告] Token 消耗接近预算上限 "
                f"({self._total_tokens}/{self.max_tokens})，"
                f"剩余 {remaining}。请尽快完成任务。"
            )
        return None

    @property
    def total_tokens(self) -> int:
        return self._total_tokens

    # ----- 重置 -----

    def reset(self) -> None:
        """重置所有计数器（对话重置时调用）."""
        self._current_round = 0
        self._total_tokens = 0
        self._recent_calls.clear()

    # ----- 内部方法 -----

    @staticmethod
    def _hash_args(arguments: dict) -> str:
        """对工具参数做哈希，用于快速比较."""
        serialized = json.dumps(arguments, sort_keys=True, ensure_ascii=False)
        return hashlib.md5(serialized.encode()).hexdigest()
