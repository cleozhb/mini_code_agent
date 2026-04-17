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
from .tracker import (
    ComparisonReport,
    EvalTracker,
    SummaryDiff,
    TaskMetricDiff,
    TrendPoint,
    TrendReport,
    suite_result_from_dict,
    suite_result_to_dict,
)

__all__ = [
    "BenchmarkSuite",
    "BenchmarkTask",
    "ComparisonReport",
    "EvalRunner",
    "EvalSummary",
    "EvalTracker",
    "FailureCategory",
    "FileFingerprint",
    "KNOWN_MODELS",
    "ModelPricing",
    "SnapshotDiff",
    "SuiteResult",
    "SummaryDiff",
    "TaskMetricDiff",
    "TaskResult",
    "TrendPoint",
    "TrendReport",
    "capture",
    "classify_failure",
    "compute_edit_metrics",
    "compute_suite_hash",
    "compute_summary",
    "compute_task_hash",
    "diff",
    "pricing_for",
    "run_validate_script",
    "suite_result_from_dict",
    "suite_result_to_dict",
]

