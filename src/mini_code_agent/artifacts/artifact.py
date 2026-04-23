"""核心 Artifact — SubtaskArtifact 及其相关枚举，整个协议的中心数据结构."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from .decision import ArtifactDecision
from .patch import EditOperation, FileEdit, Patch
from .resource import ResourceUsage
from .scope import ScopeCheck
from .verification import CheckResult, SelfVerification

logger = logging.getLogger(__name__)

_JSON_SIZE_WARNING_BYTES = 1 * 1024 * 1024  # 1MB


class Confidence(str, Enum):
    """Worker 对自己产出的信心."""

    DONE = "DONE"  # 完全完成，所有验证通过
    PARTIAL = "PARTIAL"  # 部分完成，知道哪里没完成
    UNCERTAIN = "UNCERTAIN"  # 完成了但不确定是否正确
    STUCK = "STUCK"  # 卡住了，无法继续（应触发 ESCALATE）


class ArtifactStatus(str, Enum):
    """Artifact 生命周期状态."""

    DRAFT = "DRAFT"  # Worker 还在编辑
    SUBMITTED = "SUBMITTED"  # Worker 提交给 Orchestrator / Reviewer
    UNDER_REVIEW = "UNDER_REVIEW"  # Reviewer 正在审查
    APPROVED = "APPROVED"  # Reviewer 通过
    REJECTED = "REJECTED"  # Reviewer 拒绝（不可修复，整体放弃）
    CHANGES_REQUESTED = "CHANGES_REQUESTED"  # Reviewer 要求修改
    APPLIED = "APPLIED"  # Orchestrator 已把 patch 合入主线
    REVERTED = "REVERTED"  # 合入后因为某些原因被回滚


# ──────────────────────────────────────────────────────────────────
# 序列化辅助
# ──────────────────────────────────────────────────────────────────

def _check_result_to_dict(cr: CheckResult) -> dict[str, Any]:
    return {
        "check_name": cr.check_name,
        "passed": cr.passed,
        "skipped": cr.skipped,
        "skip_reason": cr.skip_reason,
        "duration_seconds": cr.duration_seconds,
        "details": cr.details,
        "items_checked": cr.items_checked,
        "items_failed": cr.items_failed,
    }


def _check_result_from_dict(d: dict[str, Any]) -> CheckResult:
    return CheckResult(
        check_name=d["check_name"],
        passed=d["passed"],
        skipped=d["skipped"],
        skip_reason=d.get("skip_reason"),
        duration_seconds=d["duration_seconds"],
        details=d["details"],
        items_checked=d["items_checked"],
        items_failed=d["items_failed"],
    )


def _verification_to_dict(sv: SelfVerification) -> dict[str, Any]:
    return {
        "syntax_check": _check_result_to_dict(sv.syntax_check),
        "lint_check": _check_result_to_dict(sv.lint_check) if sv.lint_check else None,
        "type_check": _check_result_to_dict(sv.type_check) if sv.type_check else None,
        "unit_test": _check_result_to_dict(sv.unit_test) if sv.unit_test else None,
        "import_check": _check_result_to_dict(sv.import_check),
        "overall_passed": sv.overall_passed,
    }


def _verification_from_dict(d: dict[str, Any]) -> SelfVerification:
    return SelfVerification(
        syntax_check=_check_result_from_dict(d["syntax_check"]),
        lint_check=_check_result_from_dict(d["lint_check"]) if d.get("lint_check") else None,
        type_check=_check_result_from_dict(d["type_check"]) if d.get("type_check") else None,
        unit_test=_check_result_from_dict(d["unit_test"]) if d.get("unit_test") else None,
        import_check=_check_result_from_dict(d["import_check"]),
        overall_passed=d["overall_passed"],
    )


def _file_edit_to_dict(fe: FileEdit) -> dict[str, Any]:
    return {
        "path": fe.path,
        "operation": fe.operation.value,
        "old_content": fe.old_content,
        "new_content": fe.new_content,
        "old_path": fe.old_path,
        "unified_diff": fe.unified_diff,
        "lines_added": fe.lines_added,
        "lines_removed": fe.lines_removed,
    }


def _file_edit_from_dict(d: dict[str, Any]) -> FileEdit:
    return FileEdit(
        path=d["path"],
        operation=EditOperation(d["operation"]),
        old_content=d.get("old_content"),
        new_content=d.get("new_content"),
        old_path=d.get("old_path"),
        unified_diff=d["unified_diff"],
        lines_added=d["lines_added"],
        lines_removed=d["lines_removed"],
    )


def _patch_to_dict(p: Patch) -> dict[str, Any]:
    return {
        "edits": [_file_edit_to_dict(e) for e in p.edits],
        "total_files_changed": p.total_files_changed,
        "total_lines_added": p.total_lines_added,
        "total_lines_removed": p.total_lines_removed,
        "base_git_hash": p.base_git_hash,
    }


def _patch_from_dict(d: dict[str, Any]) -> Patch:
    return Patch(
        edits=[_file_edit_from_dict(e) for e in d["edits"]],
        total_files_changed=d["total_files_changed"],
        total_lines_added=d["total_lines_added"],
        total_lines_removed=d["total_lines_removed"],
        base_git_hash=d["base_git_hash"],
    )


def _scope_check_to_dict(sc: ScopeCheck) -> dict[str, Any]:
    return {
        "allowed_paths": sc.allowed_paths,
        "touched_paths": sc.touched_paths,
        "out_of_scope_paths": sc.out_of_scope_paths,
        "is_clean": sc.is_clean,
    }


def _scope_check_from_dict(d: dict[str, Any]) -> ScopeCheck:
    return ScopeCheck(
        allowed_paths=d["allowed_paths"],
        touched_paths=d["touched_paths"],
        out_of_scope_paths=d["out_of_scope_paths"],
        is_clean=d["is_clean"],
    )


def _decision_to_dict(ad: ArtifactDecision) -> dict[str, Any]:
    return {
        "description": ad.description,
        "reason": ad.reason,
        "alternatives_considered": ad.alternatives_considered,
        "reversible": ad.reversible,
        "step_number": ad.step_number,
    }


def _decision_from_dict(d: dict[str, Any]) -> ArtifactDecision:
    return ArtifactDecision(
        description=d["description"],
        reason=d["reason"],
        alternatives_considered=d.get("alternatives_considered", []),
        reversible=d.get("reversible", True),
        step_number=d.get("step_number", 0),
    )


def _resource_to_dict(ru: ResourceUsage) -> dict[str, Any]:
    return {
        "tokens_input": ru.tokens_input,
        "tokens_output": ru.tokens_output,
        "tokens_total": ru.tokens_total,
        "llm_calls": ru.llm_calls,
        "tool_calls": ru.tool_calls,
        "wall_time_seconds": ru.wall_time_seconds,
        "model_used": ru.model_used,
    }


def _resource_from_dict(d: dict[str, Any]) -> ResourceUsage:
    return ResourceUsage(
        tokens_input=d["tokens_input"],
        tokens_output=d["tokens_output"],
        tokens_total=d["tokens_total"],
        llm_calls=d["llm_calls"],
        tool_calls=d["tool_calls"],
        wall_time_seconds=d["wall_time_seconds"],
        model_used=d["model_used"],
    )


# ──────────────────────────────────────────────────────────────────
# SubtaskArtifact
# ──────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SubtaskArtifact:
    """整个 Artifact Protocol 的核心数据结构.

    一旦 finalize 后就是只读的（frozen dataclass）。
    """

    # === 身份 ===
    artifact_id: str  # UUID，全局唯一
    task_id: str  # 对应的 TaskNode.id
    created_at: datetime
    producer: str  # "worker-001" / "agent" / "orchestrator"

    # === 任务契约 ===
    task_description: str  # 任务描述（从 TaskNode 复制）
    allowed_paths: list[str]  # 任务允许操作的路径
    verification_spec: str  # 任务要求的验证方式（从 TaskNode 复制）

    # === 产出 ===
    patch: Patch  # 核心产出：代码变更
    self_verification: SelfVerification  # Worker 自己跑过的检查
    scope_check: ScopeCheck  # 越界检查
    decisions: list[ArtifactDecision]  # 关键决策
    resource_usage: ResourceUsage  # 资源消耗

    # === 自评 ===
    confidence: Confidence  # Worker 对自己产出的信心
    self_summary: str  # 2-3 句话，Worker 总结做了什么
    open_questions: list[str]  # Worker 遗留的疑问（如果有）

    # === 状态 ===
    status: ArtifactStatus  # 生命周期状态

    def to_json(self) -> dict[str, Any]:
        """序列化为 JSON 兼容的 dict."""
        return {
            "artifact_id": self.artifact_id,
            "task_id": self.task_id,
            "created_at": self.created_at.isoformat(),
            "producer": self.producer,
            "task_description": self.task_description,
            "allowed_paths": self.allowed_paths,
            "verification_spec": self.verification_spec,
            "patch": _patch_to_dict(self.patch),
            "self_verification": _verification_to_dict(self.self_verification),
            "scope_check": _scope_check_to_dict(self.scope_check),
            "decisions": [_decision_to_dict(d) for d in self.decisions],
            "resource_usage": _resource_to_dict(self.resource_usage),
            "confidence": self.confidence.value,
            "self_summary": self.self_summary,
            "open_questions": self.open_questions,
            "status": self.status.value,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> SubtaskArtifact:
        """从 JSON dict 反序列化."""
        return cls(
            artifact_id=data["artifact_id"],
            task_id=data["task_id"],
            created_at=datetime.fromisoformat(data["created_at"]),
            producer=data["producer"],
            task_description=data["task_description"],
            allowed_paths=data["allowed_paths"],
            verification_spec=data["verification_spec"],
            patch=_patch_from_dict(data["patch"]),
            self_verification=_verification_from_dict(data["self_verification"]),
            scope_check=_scope_check_from_dict(data["scope_check"]),
            decisions=[_decision_from_dict(d) for d in data.get("decisions", [])],
            resource_usage=_resource_from_dict(data["resource_usage"]),
            confidence=Confidence(data["confidence"]),
            self_summary=data["self_summary"],
            open_questions=data.get("open_questions", []),
            status=ArtifactStatus(data["status"]),
        )

    def save(self, path: str) -> None:
        """保存到磁盘."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        json_str = json.dumps(self.to_json(), ensure_ascii=False, indent=2)
        if len(json_str.encode("utf-8")) > _JSON_SIZE_WARNING_BYTES:
            logger.warning(
                "Artifact JSON 超过 1MB (%d bytes)，patch 太大通常意味着任务粒度不对",
                len(json_str.encode("utf-8")),
            )
        p.write_text(json_str, encoding="utf-8")

    @classmethod
    def load(cls, path: str) -> SubtaskArtifact:
        """从磁盘加载."""
        p = Path(path)
        data = json.loads(p.read_text(encoding="utf-8"))
        return cls.from_json(data)

    def summary_for_reviewer(self) -> str:
        """给 Reviewer 看的紧凑摘要（不含完整 patch 内容，只含元信息）."""
        lines = [
            f"Artifact: {self.artifact_id}",
            f"Task: {self.task_id} — {self.task_description}",
            f"Producer: {self.producer}",
            f"Status: {self.status.value}",
            f"Confidence: {self.confidence.value}",
            f"Summary: {self.self_summary}",
            f"Files changed: {self.patch.total_files_changed} "
            f"(+{self.patch.total_lines_added} / -{self.patch.total_lines_removed})",
            f"Scope: {'✅ clean' if self.scope_check.is_clean else '❌ out of scope: ' + ', '.join(self.scope_check.out_of_scope_paths)}",
            f"Verification: {self.self_verification.summary()}",
            f"Decisions: {len(self.decisions)}",
            f"Resources: {self.resource_usage.llm_calls} LLM calls, "
            f"{self.resource_usage.tokens_total} tokens, "
            f"{self.resource_usage.wall_time_seconds:.1f}s",
        ]
        if self.open_questions:
            lines.append(f"Open questions: {'; '.join(self.open_questions)}")
        return "\n".join(lines)

    def summary_for_ledger(self) -> dict[str, Any]:
        """给 Ledger 存档用的结构化摘要."""
        return {
            "artifact_id": self.artifact_id,
            "task_id": self.task_id,
            "created_at": self.created_at.isoformat(),
            "producer": self.producer,
            "status": self.status.value,
            "confidence": self.confidence.value,
            "self_summary": self.self_summary,
            "files_changed": self.patch.total_files_changed,
            "lines_added": self.patch.total_lines_added,
            "lines_removed": self.patch.total_lines_removed,
            "scope_clean": self.scope_check.is_clean,
            "verification_passed": self.self_verification.overall_passed,
            "decisions_count": len(self.decisions),
            "open_questions": self.open_questions,
            "tokens_total": self.resource_usage.tokens_total,
            "llm_calls": self.resource_usage.llm_calls,
            "wall_time_seconds": self.resource_usage.wall_time_seconds,
        }

    def diff_preview(self, max_lines: int = 40) -> str:
        """截断的 diff 预览，用于 CLI 展示."""
        full_diff = self.patch.to_unified_diff()
        if not full_diff:
            return "(no changes)"
        lines = full_diff.splitlines()
        if len(lines) <= max_lines:
            return full_diff
        truncated = lines[:max_lines]
        truncated.append(f"\n... ({len(lines) - max_lines} more lines)")
        return "\n".join(truncated)
