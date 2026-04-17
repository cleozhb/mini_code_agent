"""eval 模块 — Evaluation 系统（benchmark 任务、runner、历史追踪）.

详细设计见 DESIGN.md。
"""

from .benchmark import (
    BenchmarkSuite,
    BenchmarkTask,
    compute_suite_hash,
    compute_task_hash,
)
from .snapshot import FileFingerprint, SnapshotDiff, capture, diff

__all__ = [
    "BenchmarkSuite",
    "BenchmarkTask",
    "FileFingerprint",
    "SnapshotDiff",
    "capture",
    "compute_suite_hash",
    "compute_task_hash",
    "diff",
]

