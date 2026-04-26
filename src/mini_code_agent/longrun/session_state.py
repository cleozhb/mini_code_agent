"""SessionState — 会话状态快照，用于 Checkpoint 和 Resume."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum


class CheckpointTrigger(str, Enum):
    """Checkpoint 创建原因."""

    SUBTASK_COMPLETE = "SUBTASK_COMPLETE"       # 子任务完成后自动创建
    TOKEN_THRESHOLD = "TOKEN_THRESHOLD"         # 达到 token 阈值自动创建
    PHASE_TRANSITION = "PHASE_TRANSITION"       # TaskGraph 阶段切换时自动创建
    USER_PAUSE = "USER_PAUSE"                   # 用户触发 /pause
    BEFORE_RISKY_ACTION = "BEFORE_RISKY_ACTION" # 高风险操作前自动创建（L2.5）
    PROGRESS_STALL = "PROGRESS_STALL"           # ProgressMonitor 检测到卡死（L2.4）


@dataclass
class SessionState:
    """会话状态快照 — Checkpoint 的核心数据结构.

    设计原则：不存本体，只存引用（ID / hash / path），
    通过引用可以重建完整状态。
    """

    # === 身份 ===
    checkpoint_id: str              # UUID
    task_id: str                    # 对应的 TaskLedger.task_id
    created_at: datetime
    trigger: CheckpointTrigger      # 为什么创建这个 checkpoint

    # === 状态引用（不存本体，只存 ID）===
    ledger_path: str                # 指向 .agent/ledger/{task_id}.json
    ledger_hash: str                # Ledger 文件的 SHA256（用于验证一致性）

    # === 任务图状态 ===
    task_graph_json: str            # TaskGraph 的完整序列化
    current_task_id: str | None     # 正在执行的子任务

    # === 代码状态 ===
    git_checkpoint_hash: str        # 对应的 git commit hash
    git_branch: str                 # 当前分支
    uncommitted_changes: bool       # 是否有未提交的修改
                                    # （应该总是 False，为 True 说明 checkpoint 不完整）

    # === 进程元信息 ===
    config_snapshot: dict = field(default_factory=dict)  # LongRunConfig 序列化
    step_number: int = 0            # 全局步数

    # === 对话历史（非恢复必需，但保留以便调试）===
    recent_messages_summary: str = ""       # 最近对话的摘要
    recent_messages_full: list[dict] | None = None  # 可选：完整的最近 N 条消息

    def to_dict(self) -> dict:
        """序列化为 JSON 兼容的 dict."""
        return {
            "checkpoint_id": self.checkpoint_id,
            "task_id": self.task_id,
            "created_at": self.created_at.isoformat(),
            "trigger": self.trigger.value,
            "ledger_path": self.ledger_path,
            "ledger_hash": self.ledger_hash,
            "task_graph_json": self.task_graph_json,
            "current_task_id": self.current_task_id,
            "git_checkpoint_hash": self.git_checkpoint_hash,
            "git_branch": self.git_branch,
            "uncommitted_changes": self.uncommitted_changes,
            "config_snapshot": self.config_snapshot,
            "step_number": self.step_number,
            "recent_messages_summary": self.recent_messages_summary,
            "recent_messages_full": self.recent_messages_full,
        }

    def to_json(self) -> str:
        """序列化为 JSON 字符串."""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> SessionState:
        """从 JSON dict 反序列化."""
        return cls(
            checkpoint_id=d["checkpoint_id"],
            task_id=d["task_id"],
            created_at=datetime.fromisoformat(d["created_at"]),
            trigger=CheckpointTrigger(d["trigger"]),
            ledger_path=d["ledger_path"],
            ledger_hash=d["ledger_hash"],
            task_graph_json=d["task_graph_json"],
            current_task_id=d.get("current_task_id"),
            git_checkpoint_hash=d["git_checkpoint_hash"],
            git_branch=d["git_branch"],
            uncommitted_changes=d.get("uncommitted_changes", False),
            config_snapshot=d.get("config_snapshot", {}),
            step_number=d.get("step_number", 0),
            recent_messages_summary=d.get("recent_messages_summary", ""),
            recent_messages_full=d.get("recent_messages_full"),
        )

    @classmethod
    def from_json(cls, json_str: str) -> SessionState:
        """从 JSON 字符串反序列化."""
        return cls.from_dict(json.loads(json_str))


@dataclass
class CheckpointMeta:
    """Checkpoint 的轻量元信息（用于 index.json 和列表展示）."""

    id: str
    created_at: datetime
    trigger: CheckpointTrigger
    step_number: int
    token_count: int
    git_hash: str
    current_task_id: str | None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "created_at": self.created_at.isoformat(),
            "trigger": self.trigger.value,
            "step_number": self.step_number,
            "token_count": self.token_count,
            "git_hash": self.git_hash,
            "current_task_id": self.current_task_id,
        }

    @classmethod
    def from_dict(cls, d: dict) -> CheckpointMeta:
        return cls(
            id=d["id"],
            created_at=datetime.fromisoformat(d["created_at"]),
            trigger=CheckpointTrigger(d["trigger"]),
            step_number=d.get("step_number", 0),
            token_count=d.get("token_count", 0),
            git_hash=d.get("git_hash", ""),
            current_task_id=d.get("current_task_id"),
        )
