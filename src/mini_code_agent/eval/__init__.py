"""eval 模块 — Evaluation 系统（benchmark 任务、runner、历史追踪）.

详细设计见 DESIGN.md。
"""

from .benchmark import (
    BenchmarkSuite,
    BenchmarkTask,
    compute_suite_hash,
    compute_task_hash,
)
from .runner import (
    KNOWN_MODELS,
    EvalRunner,
    EvalSummary,
    FailureCategory,
    ModelPricing,
    SuiteResult,
    TaskResult,
    classify_failure,
    compute_edit_metrics,
    compute_summary,
    pricing_for,
    run_validate_script,
)
from .snapshot import FileFingerprint, SnapshotDiff, capture, diff

__all__ = [
    "BenchmarkSuite",
    "BenchmarkTask",
    "EvalRunner",
    "EvalSummary",
    "FailureCategory",
    "FileFingerprint",
    "KNOWN_MODELS",
    "ModelPricing",
    "SnapshotDiff",
    "SuiteResult",
    "TaskResult",
    "capture",
    "classify_failure",
    "compute_edit_metrics",
    "compute_suite_hash",
    "compute_summary",
    "compute_task_hash",
    "diff",
    "pricing_for",
    "run_validate_script",
]

