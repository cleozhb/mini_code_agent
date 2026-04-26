"""Incremental Verifier 的数据结构."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

# 复用 Artifact 协议里的 CheckResult — 字段一致，不重复定义
from ..artifacts.verification import CheckResult as VerificationCheck

__all__ = ["IncrementalVerificationResult", "VerificationCheck"]


@dataclass
class IncrementalVerificationResult:
    """一次 incremental verification 的整体结果."""

    task_id: str
    level: int  # 1 or 2
    checks: list[VerificationCheck]
    overall_passed: bool
    files_verified: list[str]
    total_duration_seconds: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def any_failed(self) -> bool:
        """是否有任何非 skipped 的 check 失败."""
        return any(not c.passed and not c.skipped for c in self.checks)

    def get_failed_checks(self) -> list[VerificationCheck]:
        """所有失败的 check（不含 skipped）."""
        return [c for c in self.checks if not c.passed and not c.skipped]

    def get_check(self, name: str) -> VerificationCheck | None:
        """按名字取 check（可能多个，取第一个）."""
        for c in self.checks:
            if c.check_name == name:
                return c
        return None

    def summary(self) -> str:
        """给 LLM 看的一行/多行摘要."""
        if not self.checks:
            return f"L{self.level}: no checks"
        parts: list[str] = []
        for c in self.checks:
            if c.skipped:
                parts.append(f"⏭ {c.check_name}({c.skip_reason or 'skipped'})")
            elif c.passed:
                if c.items_checked:
                    parts.append(f"✅ {c.check_name}({c.items_checked})")
                else:
                    parts.append(f"✅ {c.check_name}")
            else:
                detail = (c.details or "").strip().splitlines()
                head = detail[0] if detail else ""
                if c.items_failed:
                    parts.append(
                        f"❌ {c.check_name}({c.items_failed} failed): {head}"
                    )
                else:
                    parts.append(f"❌ {c.check_name}: {head}")
        head_line = (
            f"L{self.level} {'PASS' if self.overall_passed else 'FAIL'} "
            f"({self.total_duration_seconds:.2f}s, {len(self.files_verified)} files)"
        )
        return head_line + "\n  " + "\n  ".join(parts)
