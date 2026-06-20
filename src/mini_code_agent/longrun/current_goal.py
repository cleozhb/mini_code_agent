"""Current Goal pointer for deterministic /goal resume selection."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from .ledger_manager import LedgerNotFoundError, TaskLedgerManager
from .ledger_types import TaskRunStatus
from .task_ledger import TaskLedger


@dataclass
class CurrentGoalRef:
    """Pointer to the Goal that /goal resume should use by default."""

    task_id: str
    status: TaskRunStatus
    updated_at: datetime
    session_id: str | None = None
    trace_dir: str | None = None
    last_checkpoint_id: str | None = None

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "status": self.status.value,
            "updated_at": self.updated_at.isoformat(),
            "session_id": self.session_id,
            "trace_dir": self.trace_dir,
            "last_checkpoint_id": self.last_checkpoint_id,
        }

    @classmethod
    def from_dict(cls, data: dict) -> CurrentGoalRef:
        return cls(
            task_id=data["task_id"],
            status=TaskRunStatus(data["status"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            session_id=data.get("session_id"),
            trace_dir=data.get("trace_dir"),
            last_checkpoint_id=data.get("last_checkpoint_id"),
        )


def current_goal_path(project_path: str | Path) -> Path:
    return Path(project_path) / ".agent" / "current_goal.json"


def load_current_goal(project_path: str | Path) -> CurrentGoalRef | None:
    path = current_goal_path(project_path)
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return CurrentGoalRef.from_dict(json.load(f))


def save_current_goal(project_path: str | Path, ledger: TaskLedger) -> None:
    path = current_goal_path(project_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ref = CurrentGoalRef(
        task_id=ledger.task_id,
        status=ledger.status,
        updated_at=datetime.now(UTC),
        session_id=ledger.session_id,
        trace_dir=ledger.trace_dir,
        last_checkpoint_id=ledger.last_checkpoint_id,
    )
    tmp_path = str(path) + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(ref.to_dict(), f, ensure_ascii=False, indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def clear_current_goal(project_path: str | Path, task_id: str) -> None:
    path = current_goal_path(project_path)
    try:
        ref = load_current_goal(project_path)
    except (OSError, json.JSONDecodeError, KeyError, ValueError):
        ref = None
    if ref is None or ref.task_id == task_id:
        path.unlink(missing_ok=True)


def apply_trace_context(ledger: TaskLedger, trace_recorder: object | None) -> None:
    """Copy trace identity onto the ledger when tracing is enabled."""
    if trace_recorder is None:
        return
    context = getattr(trace_recorder, "context", None)
    if context is not None:
        ledger.session_id = getattr(context, "session_id", None)
    session_dir = getattr(trace_recorder, "session_dir", None)
    if session_dir is not None:
        ledger.trace_dir = str(session_dir)


def is_resumable_status(status: TaskRunStatus) -> bool:
    return status in {TaskRunStatus.PAUSED, TaskRunStatus.RUNNING}


def resolve_goal_to_resume(
    manager: TaskLedgerManager,
    project_path: str | Path,
    task_id_prefix: str | None = None,
) -> TaskLedger:
    """Resolve the exact Goal ledger for /goal resume.

    Explicit task IDs win. Without one, only the persisted current-goal pointer
    is used; callers should list candidates instead of guessing by updated_at.
    """
    if task_id_prefix and task_id_prefix.strip():
        ledger = manager.find_by_id_prefix(task_id_prefix)
        if not is_resumable_status(ledger.status):
            raise LedgerNotFoundError(
                f"Goal 不可恢复: {ledger.task_id[:8]} ({ledger.status.value})"
            )
        return ledger

    try:
        ref = load_current_goal(project_path)
    except (OSError, json.JSONDecodeError, KeyError, ValueError) as e:
        raise LedgerNotFoundError(f"当前 Goal 指针无效: {e}") from e
    if ref is None:
        raise LedgerNotFoundError("没有当前 Goal 指针")

    ledger = manager.load(ref.task_id)
    if not is_resumable_status(ledger.status):
        raise LedgerNotFoundError(
            f"当前 Goal 不可恢复: {ledger.task_id[:8]} ({ledger.status.value})"
        )
    return ledger


def list_resumable_goals(manager: TaskLedgerManager) -> list:
    return [m for m in manager.list_all() if is_resumable_status(m.status)]


def format_goal_candidates(candidates: list) -> str:
    lines = []
    for item in candidates:
        task_id = getattr(item, "task_id", "")
        status = getattr(item, "status", "")
        goal = getattr(item, "goal", "")
        status_text = status.value if hasattr(status, "value") else str(status)
        lines.append(f"{task_id[:8]}  {status_text:<8}  {goal}")
    return "\n".join(lines)
