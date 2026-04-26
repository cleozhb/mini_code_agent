"""Incremental Verifier 整合层 —— 提供 verify_after_edit / verify_after_subtask."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from .level1 import QuickVerifier
from .level2 import UnitTestVerifier
from .types import IncrementalVerificationResult, VerificationCheck

if TYPE_CHECKING:
    from ..artifacts.artifact import SubtaskArtifact
    from ..artifacts.builder import ArtifactBuilder

logger = logging.getLogger(__name__)


class IncrementalVerifier:
    """统一入口，把 Level 1 / Level 2 串起来."""

    def __init__(
        self,
        level1: QuickVerifier | None = None,
        level2: UnitTestVerifier | None = None,
    ) -> None:
        self.level1 = level1 or QuickVerifier()
        self.level2 = level2 or UnitTestVerifier()

    async def verify_after_edit(
        self,
        files_changed: list[str],
        project_path: str,
        task_id: str = "",
    ) -> IncrementalVerificationResult:
        """每次 Edit 工具调用后触发 — 只跑层级 1."""
        if not files_changed:
            return IncrementalVerificationResult(
                task_id=task_id,
                level=1,
                checks=[],
                overall_passed=True,
                files_verified=[],
                total_duration_seconds=0.0,
            )
        return await self.level1.verify(files_changed, project_path, task_id=task_id)

    async def verify_after_subtask(
        self,
        artifact: "SubtaskArtifact",
        project_path: str,
    ) -> IncrementalVerificationResult:
        """子任务完成时触发 — 先跑 L1，通过后再跑 L2."""
        files = [e.path for e in artifact.patch.edits]
        task_id = artifact.task_id

        if not files:
            return IncrementalVerificationResult(
                task_id=task_id,
                level=1,
                checks=[],
                overall_passed=True,
                files_verified=[],
                total_duration_seconds=0.0,
            )

        l1 = await self.level1.verify(files, project_path, task_id=task_id)
        if not l1.overall_passed:
            logger.info("L1 failed, skipping L2: %s", l1.summary())
            return l1

        l2 = await self.level2.verify(files, project_path, task_id=task_id)
        return _merge_results(l1, l2)


def attach_verification_to_builder(
    builder: "ArtifactBuilder",
    result: IncrementalVerificationResult,
) -> None:
    """把 incremental verification 结果挂到 ArtifactBuilder 上.

    - 构造 SelfVerification（按 check_name 分桶 syntax/lint/type/unit_test/import）
    - 如果验证未通过且 builder 当前 confidence 是 DONE，降级为 UNCERTAIN
    """
    from ..artifacts.artifact import Confidence
    from ..artifacts.verification import CheckResult, SelfVerification

    def _empty(name: str) -> CheckResult:
        return CheckResult(
            check_name=name,
            passed=True,
            skipped=True,
            skip_reason="not run",
            duration_seconds=0.0,
            details="",
            items_checked=0,
            items_failed=0,
        )

    syntax = result.get_check("syntax") or _empty("syntax")
    import_c = result.get_check("import") or _empty("import")
    lint = result.get_check("lint")
    type_c = result.get_check("type_check") or result.get_check("type")
    unit = result.get_check("unit_test")

    self_verif = SelfVerification(
        syntax_check=syntax,
        lint_check=lint,
        type_check=type_c,
        unit_test=unit,
        import_check=import_c,
        overall_passed=result.overall_passed,
    )
    builder.attach_self_verification(self_verif)

    if not result.overall_passed and builder._confidence == Confidence.DONE:
        builder.set_confidence(
            Confidence.UNCERTAIN,
            f"验证未通过：{result.summary()}",
        )


def _merge_results(
    l1: IncrementalVerificationResult,
    l2: IncrementalVerificationResult,
) -> IncrementalVerificationResult:
    """合并 L1 + L2 的检查项，level 取 2."""
    merged_checks: list[VerificationCheck] = list(l1.checks) + list(l2.checks)
    overall = all(c.passed or c.skipped for c in merged_checks)
    files = list(dict.fromkeys(l1.files_verified + l2.files_verified))
    return IncrementalVerificationResult(
        task_id=l1.task_id or l2.task_id,
        level=2,
        checks=merged_checks,
        overall_passed=overall,
        files_verified=files,
        total_duration_seconds=l1.total_duration_seconds + l2.total_duration_seconds,
    )
