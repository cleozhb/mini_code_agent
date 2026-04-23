"""自验证结果 — Worker 在交付前自己跑过的检查."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CheckResult:
    """单项检查结果."""

    check_name: str  # "syntax" / "unit_test" / "lint" / ...
    passed: bool
    skipped: bool  # True 表示未执行（例如没有相关测试文件）
    skip_reason: str | None
    duration_seconds: float
    details: str  # 详细输出，通过时可以是 "ok"，失败时是错误信息
    items_checked: int  # 跑了多少个测试 / 检查了多少个文件
    items_failed: int


@dataclass
class SelfVerification:
    """Worker 在交付前自己跑过的检查集合."""

    syntax_check: CheckResult  # 语法检查（ast.parse 或 tsc --noEmit 等）
    lint_check: CheckResult | None  # lint 检查（可选）
    type_check: CheckResult | None  # 类型检查（可选）
    unit_test: CheckResult | None  # 相关单元测试（可选）
    import_check: CheckResult  # import 是否能解析

    overall_passed: bool  # 所有非 None 的检查是否全通过

    def summary(self) -> str:
        """一行摘要，如 '✅ syntax, ✅ 3 tests, ⚠️ 2 lint warnings'."""
        parts: list[str] = []
        checks: list[tuple[str, CheckResult | None]] = [
            ("syntax", self.syntax_check),
            ("lint", self.lint_check),
            ("type", self.type_check),
            ("test", self.unit_test),
            ("import", self.import_check),
        ]
        for name, check in checks:
            if check is None:
                continue
            if check.skipped:
                parts.append(f"⏭️ {name}")
            elif check.passed:
                if check.items_checked > 0:
                    parts.append(f"✅ {check.items_checked} {name}")
                else:
                    parts.append(f"✅ {name}")
            else:
                if check.items_failed > 0:
                    parts.append(f"❌ {check.items_failed} {name} failures")
                else:
                    parts.append(f"❌ {name}")
        return ", ".join(parts) if parts else "no checks"
