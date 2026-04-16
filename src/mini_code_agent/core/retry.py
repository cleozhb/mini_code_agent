"""重试控制器 — 追踪 Agent 自动修复的错误-修复循环."""

from __future__ import annotations

from dataclasses import dataclass, field


DEFAULT_MAX_RETRIES = 3


@dataclass
class AttemptRecord:
    """一次修复尝试的记录."""

    errors: list[str] = field(default_factory=list)
    fix_summary: str = ""  # 这次尝试中 LLM 给出的最终回答/修复说明


class RetryController:
    """控制 Agent 在验证失败后的自动重试次数.

    - 最多自动重试 max_retries 次
    - 每次重试时把历史错误和尝试过的修复告诉 LLM，避免重复踩坑
    - 超过重试次数后生成一份总结文本
    """

    def __init__(self, max_retries: int = DEFAULT_MAX_RETRIES) -> None:
        self.max_retries = max_retries
        self._attempts: list[AttemptRecord] = []

    # -------------------------- 状态 --------------------------

    @property
    def attempts(self) -> list[AttemptRecord]:
        return list(self._attempts)

    @property
    def attempts_count(self) -> int:
        return len(self._attempts)

    def can_retry(self) -> bool:
        """是否还能继续自动重试."""
        return len(self._attempts) < self.max_retries

    def reset(self) -> None:
        """清空历史尝试（每个顶层任务开始时调用）."""
        self._attempts.clear()

    def record_attempt(self, errors: list[str], fix_summary: str = "") -> None:
        """记录一次修复尝试."""
        self._attempts.append(
            AttemptRecord(errors=list(errors), fix_summary=fix_summary or "")
        )

    # -------------------------- Prompt 构建 --------------------------

    def build_retry_prompt(self, new_errors: list[str]) -> str:
        """构造回传给 LLM 的重试提示.

        会把本轮发现的错误 + 此前尝试过的修复拼在一起，提醒 LLM 不要重复。
        """
        remaining = max(0, self.max_retries - len(self._attempts))
        lines = [
            "验证发现问题，需要你修复后再继续。",
            "",
            "## 当前验证错误",
        ]
        for i, err in enumerate(new_errors, 1):
            lines.append(f"{i}. {err}")

        if self._attempts:
            lines.append("")
            lines.append("## 此前的修复尝试（未彻底解决）")
            for idx, att in enumerate(self._attempts, 1):
                summary = att.fix_summary.strip() or "（未给出明确修复说明）"
                # 截断太长的修复说明
                if len(summary) > 400:
                    summary = summary[:400] + "...（已截断）"
                err_preview = "; ".join(att.errors[:3]) if att.errors else "无"
                if len(err_preview) > 300:
                    err_preview = err_preview[:300] + "..."
                lines.append(f"### 尝试 {idx}")
                lines.append(f"- 当时遇到的错误：{err_preview}")
                lines.append(f"- 你尝试的修复：{summary}")
            lines.append("")
            lines.append(
                "请**不要重复**上面的修复思路，换个角度分析根因，"
                "必要时重新阅读相关代码和错误信息。"
            )

        lines.append("")
        lines.append(f"你还剩 {remaining} 次自动重试机会。")
        return "\n".join(lines)

    def build_giveup_summary(self) -> str:
        """超过重试次数后，给用户的说明."""
        lines = [
            f"我尝试了 {len(self._attempts)} 次自动修复但都没有成功：",
            "",
        ]
        for idx, att in enumerate(self._attempts, 1):
            summary = att.fix_summary.strip() or "（未给出明确修复说明）"
            if len(summary) > 400:
                summary = summary[:400] + "...（已截断）"
            err_preview = "; ".join(att.errors[:3]) if att.errors else "无"
            if len(err_preview) > 300:
                err_preview = err_preview[:300] + "..."
            lines.append(f"{idx}. 错误: {err_preview}")
            lines.append(f"   尝试的修复: {summary}")

        lines.append("")
        lines.append("建议你检查：")
        lines.append("- 相关依赖是否完整安装（运行环境、pytest 等）")
        lines.append("- 测试期望是否与当前实现匹配，必要时调整测试")
        lines.append("- 是否存在需要手动处理的环境问题（权限、外部服务等）")
        lines.append("- 我的理解是否偏离了你的真实意图，欢迎补充说明")
        return "\n".join(lines)
