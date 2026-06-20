"""longrun 模块 — 长程任务管理：Task Ledger、进度追踪与断点恢复."""

from .checkpoint_manager import (
    CheckpointError,
    CheckpointManager,
    CorruptedCheckpointError,
)
from .config import LongRunConfig
from .current_goal import CurrentGoalRef
from .ledger_manager import AmbiguousLedgerError, LedgerNotFoundError, TaskLedgerManager
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
    "AmbiguousLedgerError",
    "CheckpointError",
    "CheckpointManager",
    "CheckpointMeta",
    "CheckpointTrigger",
    "CompletedTaskRecord",
    "CorruptedCheckpointError",
    "CurrentGoalRef",
    "DecisionRecord",
    "FailedAttemptRecord",
    "LedgerMeta",
    "LedgerNotFoundError",
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
