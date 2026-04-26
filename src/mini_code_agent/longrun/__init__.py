"""longrun 模块 — 长程任务管理：Task Ledger、进度追踪与断点恢复."""

from .checkpoint_manager import (
    CheckpointError,
    CheckpointManager,
    CorruptedCheckpointError,
)
from .config import LongRunConfig
from .ledger_manager import TaskLedgerManager
from .ledger_types import (
    ActiveIssue,
    CompletedTaskRecord,
    DecisionRecord,
    FailedAttemptRecord,
    Milestone,
    TaskRunStatus,
)
from .resume_manager import ResumeContext, ResumeError, ResumeManager, UncommittedChangesError
from .session_state import CheckpointMeta, CheckpointTrigger, SessionState
from .task_ledger import LedgerMeta, TaskLedger

__all__ = [
    "ActiveIssue",
    "CheckpointError",
    "CheckpointManager",
    "CheckpointMeta",
    "CheckpointTrigger",
    "CompletedTaskRecord",
    "CorruptedCheckpointError",
    "DecisionRecord",
    "FailedAttemptRecord",
    "LedgerMeta",
    "LongRunConfig",
    "Milestone",
    "ResumeContext",
    "ResumeError",
    "ResumeManager",
    "SessionState",
    "TaskLedger",
    "TaskLedgerManager",
    "TaskRunStatus",
    "UncommittedChangesError",
]
