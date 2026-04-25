"""Ledger 数据结构 — TaskRunStatus、Milestone、CompletedTaskRecord 等."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum


class TaskRunStatus(str, Enum):
    """长程任务的运行状态."""

    NOT_STARTED = "NOT_STARTED"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    ABORTED = "ABORTED"


@dataclass
class Milestone:
    """长程任务中的里程碑."""

    id: str
    description: str  # "完成数据模型定义"
    associated_task_ids: list[str] = field(default_factory=list)
    expected_by_step: int = 0  # 预计在第 N 步前完成
    actual_step: int | None = None
    status: str = "PENDING"  # PENDING / REACHED / OVERDUE

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "description": self.description,
            "associated_task_ids": self.associated_task_ids,
            "expected_by_step": self.expected_by_step,
            "actual_step": self.actual_step,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Milestone:
        return cls(
            id=d["id"],
            description=d["description"],
            associated_task_ids=d.get("associated_task_ids", []),
            expected_by_step=d.get("expected_by_step", 0),
            actual_step=d.get("actual_step"),
            status=d.get("status", "PENDING"),
        )


@dataclass
class CompletedTaskRecord:
    """已完成子任务的摘要索引（Artifact 的轻量引用）."""

    task_id: str
    artifact_id: str  # 指向 ArtifactStore 中的完整 Artifact
    description: str
    self_summary: str  # 从 Artifact 复制过来
    files_changed: list[str] = field(default_factory=list)  # 从 Artifact.patch 抽取
    verification_passed: bool = False
    reviewer_verdict: str | None = None  # L2 以后由 Reviewer 填写
    confidence: str = "DONE"  # Artifact.confidence 的值
    step_number_start: int = 0
    step_number_end: int = 0
    token_count: int = 0
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "artifact_id": self.artifact_id,
            "description": self.description,
            "self_summary": self.self_summary,
            "files_changed": self.files_changed,
            "verification_passed": self.verification_passed,
            "reviewer_verdict": self.reviewer_verdict,
            "confidence": self.confidence,
            "step_number_start": self.step_number_start,
            "step_number_end": self.step_number_end,
            "token_count": self.token_count,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> CompletedTaskRecord:
        return cls(
            task_id=d["task_id"],
            artifact_id=d["artifact_id"],
            description=d["description"],
            self_summary=d["self_summary"],
            files_changed=d.get("files_changed", []),
            verification_passed=d.get("verification_passed", False),
            reviewer_verdict=d.get("reviewer_verdict"),
            confidence=d.get("confidence", "DONE"),
            step_number_start=d.get("step_number_start", 0),
            step_number_end=d.get("step_number_end", 0),
            token_count=d.get("token_count", 0),
            timestamp=datetime.fromisoformat(d["timestamp"]) if "timestamp" in d else datetime.now(UTC),
        )


@dataclass
class DecisionRecord:
    """跨任务的关键决策（从各个 Artifact 汇总）."""

    description: str
    reason: str
    source_task_id: str  # 在哪个子任务做出的
    reversible: bool = True
    step_number: int = 0

    def to_dict(self) -> dict:
        return {
            "description": self.description,
            "reason": self.reason,
            "source_task_id": self.source_task_id,
            "reversible": self.reversible,
            "step_number": self.step_number,
        }

    @classmethod
    def from_dict(cls, d: dict) -> DecisionRecord:
        return cls(
            description=d["description"],
            reason=d["reason"],
            source_task_id=d["source_task_id"],
            reversible=d.get("reversible", True),
            step_number=d.get("step_number", 0),
        )


@dataclass
class ActiveIssue:
    """活跃问题（在子任务执行中暴露的问题）."""

    id: str
    description: str  # "登录 API 的 token 过期时间配置不对"
    source_task_id: str  # 是哪个子任务暴露的问题
    severity: str = "warning"  # "blocker" / "warning" / "info"
    first_seen_step: int = 0
    resolution_attempts: int = 0
    resolved: bool = False

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "description": self.description,
            "source_task_id": self.source_task_id,
            "severity": self.severity,
            "first_seen_step": self.first_seen_step,
            "resolution_attempts": self.resolution_attempts,
            "resolved": self.resolved,
        }

    @classmethod
    def from_dict(cls, d: dict) -> ActiveIssue:
        return cls(
            id=d["id"],
            description=d["description"],
            source_task_id=d["source_task_id"],
            severity=d.get("severity", "warning"),
            first_seen_step=d.get("first_seen_step", 0),
            resolution_attempts=d.get("resolution_attempts", 0),
            resolved=d.get("resolved", False),
        )


@dataclass
class FailedAttemptRecord:
    """失败尝试的记录."""

    task_id: str
    artifact_id: str  # 那次失败产生的 Artifact（即使失败也存档）
    approach_description: str  # 尝试的做法
    failure_reason: str
    step_number: int = 0
    lesson_learned: str | None = None  # 如果有明显教训

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "artifact_id": self.artifact_id,
            "approach_description": self.approach_description,
            "failure_reason": self.failure_reason,
            "step_number": self.step_number,
            "lesson_learned": self.lesson_learned,
        }

    @classmethod
    def from_dict(cls, d: dict) -> FailedAttemptRecord:
        return cls(
            task_id=d["task_id"],
            artifact_id=d["artifact_id"],
            approach_description=d["approach_description"],
            failure_reason=d["failure_reason"],
            step_number=d.get("step_number", 0),
            lesson_learned=d.get("lesson_learned"),
        )
