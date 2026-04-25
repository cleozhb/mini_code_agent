"""TaskLedger 数据结构 — Agent 的"外部记忆"，记录长程任务的完整状态."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

from .ledger_types import (
    ActiveIssue,
    CompletedTaskRecord,
    DecisionRecord,
    FailedAttemptRecord,
    Milestone,
    TaskRunStatus,
)


@dataclass
class LedgerMeta:
    """Ledger 的轻量元信息（用于 list_all）."""

    task_id: str
    goal: str
    status: TaskRunStatus
    created_at: datetime
    updated_at: datetime
    completed_tasks: int
    total_tokens_used: int


@dataclass
class TaskLedger:
    """长程任务的完整 Ledger.

    在每轮开始时被读取、每个子任务结束后被更新、进程崩溃后从磁盘恢复。
    """

    # === 任务元信息 ===
    task_id: str  # 整个长程任务的 ID（不是子任务）
    goal: str  # 用户原始目标
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    status: TaskRunStatus = TaskRunStatus.NOT_STARTED

    # === 执行计划 ===
    plan_summary: str = ""
    task_graph_snapshot: dict = field(default_factory=dict)  # TaskGraph 的 JSON 序列化
    current_phase: str = ""
    current_task_id: str | None = None

    # === 进度 ===
    milestones: list[Milestone] = field(default_factory=list)
    completed_tasks: list[CompletedTaskRecord] = field(default_factory=list)
    decisions_made: list[DecisionRecord] = field(default_factory=list)

    # === 问题 ===
    active_issues: list[ActiveIssue] = field(default_factory=list)
    resolved_issues: list[ActiveIssue] = field(default_factory=list)
    failed_attempts: list[FailedAttemptRecord] = field(default_factory=list)

    # === 资源 ===
    total_tokens_used: int = 0
    total_steps: int = 0
    total_wall_time_seconds: float = 0.0
    token_budget: int = 0  # 从 config 读入
    token_budget_remaining: int = 0

    # === 版本 ===
    ledger_schema_version: str = "1.0"

    def to_dict(self) -> dict:
        """序列化为 JSON 兼容的 dict（字段顺序稳定）."""
        return {
            "ledger_schema_version": self.ledger_schema_version,
            "task_id": self.task_id,
            "goal": self.goal,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "status": self.status.value,
            "plan_summary": self.plan_summary,
            "task_graph_snapshot": self.task_graph_snapshot,
            "current_phase": self.current_phase,
            "current_task_id": self.current_task_id,
            "milestones": [m.to_dict() for m in self.milestones],
            "completed_tasks": [t.to_dict() for t in self.completed_tasks],
            "decisions_made": [d.to_dict() for d in self.decisions_made],
            "active_issues": [i.to_dict() for i in self.active_issues],
            "resolved_issues": [i.to_dict() for i in self.resolved_issues],
            "failed_attempts": [f.to_dict() for f in self.failed_attempts],
            "total_tokens_used": self.total_tokens_used,
            "total_steps": self.total_steps,
            "total_wall_time_seconds": self.total_wall_time_seconds,
            "token_budget": self.token_budget,
            "token_budget_remaining": self.token_budget_remaining,
        }

    @classmethod
    def from_dict(cls, d: dict) -> TaskLedger:
        """从 JSON dict 反序列化."""
        return cls(
            task_id=d["task_id"],
            goal=d["goal"],
            created_at=datetime.fromisoformat(d["created_at"]),
            updated_at=datetime.fromisoformat(d["updated_at"]),
            status=TaskRunStatus(d["status"]),
            plan_summary=d.get("plan_summary", ""),
            task_graph_snapshot=d.get("task_graph_snapshot", {}),
            current_phase=d.get("current_phase", ""),
            current_task_id=d.get("current_task_id"),
            milestones=[Milestone.from_dict(m) for m in d.get("milestones", [])],
            completed_tasks=[CompletedTaskRecord.from_dict(t) for t in d.get("completed_tasks", [])],
            decisions_made=[DecisionRecord.from_dict(dd) for dd in d.get("decisions_made", [])],
            active_issues=[ActiveIssue.from_dict(i) for i in d.get("active_issues", [])],
            resolved_issues=[ActiveIssue.from_dict(i) for i in d.get("resolved_issues", [])],
            failed_attempts=[FailedAttemptRecord.from_dict(f) for f in d.get("failed_attempts", [])],
            total_tokens_used=d.get("total_tokens_used", 0),
            total_steps=d.get("total_steps", 0),
            total_wall_time_seconds=d.get("total_wall_time_seconds", 0.0),
            token_budget=d.get("token_budget", 0),
            token_budget_remaining=d.get("token_budget_remaining", 0),
            ledger_schema_version=d.get("ledger_schema_version", "1.0"),
        )

    def to_meta(self) -> LedgerMeta:
        """提取轻量元信息."""
        return LedgerMeta(
            task_id=self.task_id,
            goal=self.goal,
            status=self.status,
            created_at=self.created_at,
            updated_at=self.updated_at,
            completed_tasks=len(self.completed_tasks),
            total_tokens_used=self.total_tokens_used,
        )
