"""Eval 结果追踪：持久化 / 历史列举 / 两次对比 / 趋势.

详见 DESIGN.md §5。提供：
- `EvalTracker.save` — 把 SuiteResult 落盘为 JSON
- `EvalTracker.list_runs` — 加载历史 runs
- `EvalTracker.compare` — 按 (task_id, task_hash) 取交集做 diff；改/增/删的任务单列
- `EvalTracker.trend` — 5 个核心指标的 unicode sparkline + rich 渲染
"""

from __future__ import annotations

import dataclasses
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .runner import EvalSummary, SuiteResult, TaskResult


# ---------------------------------------------------------------------------
# JSON 序列化（SuiteResult 往返）
# ---------------------------------------------------------------------------


def suite_result_to_dict(sr: SuiteResult) -> dict:
    """dataclass → dict（给 json.dumps 吃）."""
    return asdict(sr)


def suite_result_from_dict(data: dict) -> SuiteResult:
    """dict → SuiteResult（从 json.load 结果还原）.

    做容忍：未知字段忽略、缺失字段按默认值补（向前演进时不炸老文件）。
    """
    summary = EvalSummary(**_only_known(data["summary"], EvalSummary))
    results = [
        TaskResult(**_only_known(r, TaskResult))
        for r in data.get("results", [])
    ]
    return SuiteResult(
        timestamp=str(data["timestamp"]),
        git_commit=data.get("git_commit"),
        suite_hash=str(data["suite_hash"]),
        model_name=str(data["model_name"]),
        results=results,
        summary=summary,
    )


def _only_known(data: dict, cls: type) -> dict:
    """只保留目标 dataclass 里定义过的字段；缺失字段交给 dataclass 自己的默认值."""
    fields = {f.name for f in dataclasses.fields(cls)}
    return {k: v for k, v in data.items() if k in fields}


# ---------------------------------------------------------------------------
# 文件名约定
# ---------------------------------------------------------------------------


# `{timestamp_safe}__{commit}__{seq}.json`，双下划线分隔避免被 commit/ts 里的 `_` 撞到
_FILENAME_RE = re.compile(
    r"^(?P<ts>.+?)__(?P<commit>[0-9a-f]+|nocommit)__(?P<seq>\d+)\.json$"
)


def _safe_timestamp(ts: str) -> str:
    """把 ISO 时间戳转成文件名安全的形式：冒号换成短横线."""
    return ts.replace(":", "-")


def _short_commit(commit: str | None) -> str:
    if not commit:
        return "nocommit"
    return commit[:12]


# ---------------------------------------------------------------------------
# 数据结构（diff 与 trend）
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TaskMetricDiff:
    """单个共有任务在两次 run 之间的对比（已按 task 聚合过 n 次 run）."""

    task_id: str
    task_hash: str
    pass_rate_a: float
    pass_rate_b: float
    avg_step_count_a: float
    avg_step_count_b: float
    avg_cost_usd_a: float
    avg_cost_usd_b: float
    avg_wall_time_a: float
    avg_wall_time_b: float

    @property
    def pass_rate_delta(self) -> float:
        return self.pass_rate_b - self.pass_rate_a


@dataclass(frozen=True)
class SummaryDiff:
    """EvalSummary 级别的对比；每个字段都是 (a, b) 二元组，delta 由属性即时算."""

    a: EvalSummary
    b: EvalSummary

    def delta(self, field_name: str) -> float:
        return getattr(self.b, field_name) - getattr(self.a, field_name)


@dataclass
class ComparisonReport:
    """两次 SuiteResult 的 diff（DESIGN.md §5）."""

    label_a: str
    label_b: str
    suite_hash_a: str
    suite_hash_b: str
    common_tasks: list[TaskMetricDiff] = field(default_factory=list)
    removed_tasks: list[str] = field(default_factory=list)     # id 在 a 里完全没有在 b
    added_tasks: list[str] = field(default_factory=list)       # id 在 b 里完全没有在 a
    changed_def_tasks: list[str] = field(default_factory=list) # 同 id 不同 task_hash
    summary: SummaryDiff | None = None

    @property
    def same_suite_hash(self) -> bool:
        return self.suite_hash_a == self.suite_hash_b

    def render(self) -> Group:
        return _render_comparison(self)


@dataclass(frozen=True)
class TrendPoint:
    """单次 run 的趋势点：只留 sparkline 要用的字段."""

    timestamp: str
    git_commit: str | None
    task_success_rate: float
    tool_error_rate: float
    avg_cost_usd: float
    avg_wall_time_seconds: float
    verifier_recovery_rate: float


# 核心 5 指标的名字 → TrendPoint 字段 & 显示名
_TREND_METRICS: list[tuple[str, str, str]] = [
    ("task_success_rate",      "task_success_rate",      "Success"),
    ("tool_error_rate",        "tool_error_rate",        "ToolErr"),
    ("avg_cost_usd",           "avg_cost_usd",           "Cost$"),
    ("avg_wall_time_seconds",  "avg_wall_time_seconds",  "WallT"),
    ("verifier_recovery_rate", "verifier_recovery_rate", "Recov"),
]


@dataclass
class TrendReport:
    """最近 N 次 run 的趋势（DESIGN.md §5）."""

    points: list[TrendPoint]                    # 按 timestamp 升序

    def series(self, metric: str) -> list[float]:
        return [getattr(p, metric) for p in self.points]

    def sparkline(self, metric: str) -> str:
        return _sparkline(self.series(metric))

    def render(self) -> Group:
        return _render_trend(self)


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------


class EvalTracker:
    """把 SuiteResult 存到 results_dir，支持列举、对比、趋势.

    文件命名：`{timestamp_safe}__{commit12_or_nocommit}__{seq:02d}.json`
    - `timestamp_safe` = iso 时间戳把 `:` 换成 `-`
    - 同 commit 多次保存，seq 从 00 递增，不覆盖
    - 文件名顺序与保存顺序大致一致（timestamp 字典序 = 时间序）
    """

    def __init__(self, results_dir: Path) -> None:
        self.results_dir = Path(results_dir)

    # -- 持久化 -----------------------------------------------------------

    def save(self, result: SuiteResult) -> Path:
        self.results_dir.mkdir(parents=True, exist_ok=True)
        ts = _safe_timestamp(result.timestamp)
        commit = _short_commit(result.git_commit)
        seq = self._next_seq(commit)
        path = self.results_dir / f"{ts}__{commit}__{seq:02d}.json"
        path.write_text(
            json.dumps(suite_result_to_dict(result), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return path

    def _next_seq(self, commit: str) -> int:
        if not self.results_dir.is_dir():
            return 0
        max_seq = -1
        for p in self.results_dir.glob("*.json"):
            m = _FILENAME_RE.match(p.name)
            if not m:
                continue
            if m.group("commit") != commit:
                continue
            try:
                max_seq = max(max_seq, int(m.group("seq")))
            except ValueError:
                continue
        return max_seq + 1

    # -- 列举 -------------------------------------------------------------

    def list_runs(self, last_n: int | None = None) -> list[SuiteResult]:
        """按 timestamp 升序返回历史 runs；last_n 取最后 N 个（即最新）."""
        if not self.results_dir.is_dir():
            return []
        files = sorted(
            (p for p in self.results_dir.glob("*.json") if _FILENAME_RE.match(p.name)),
            key=lambda p: p.name,  # 文件名前缀是 timestamp，字典序 = 时间序
        )
        runs: list[SuiteResult] = []
        for p in files:
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                runs.append(suite_result_from_dict(data))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                # 损坏文件不阻塞整个 list
                continue
        if last_n is not None and last_n > 0:
            runs = runs[-last_n:]
        return runs

    # -- 对比 -------------------------------------------------------------

    def compare(
        self,
        run_a: SuiteResult,
        run_b: SuiteResult,
        *,
        label_a: str | None = None,
        label_b: str | None = None,
    ) -> ComparisonReport:
        """两次 SuiteResult 的 diff.

        规则（DESIGN.md §5 / §9.1）：
        - 只有 `(task_id, task_hash)` 两边都一致的任务，算"共有"能直接比
        - 同 id 但 hash 不一致 → 放进 `changed_def_tasks`（警告：定义变了，结果不可比）
        - 只在 a 里有的 id → `removed_tasks`；只在 b 里有的 id → `added_tasks`
        """
        a_by_task = _group_by_task(run_a.results)
        b_by_task = _group_by_task(run_b.results)
        a_hashes = {tid: _any_hash(rs) for tid, rs in a_by_task.items()}
        b_hashes = {tid: _any_hash(rs) for tid, rs in b_by_task.items()}

        common: list[TaskMetricDiff] = []
        changed_def: list[str] = []
        removed: list[str] = []
        added: list[str] = []

        for tid, h_a in sorted(a_hashes.items()):
            if tid not in b_hashes:
                removed.append(tid)
                continue
            h_b = b_hashes[tid]
            if h_a != h_b:
                changed_def.append(tid)
                continue
            common.append(
                _build_task_diff(tid, h_a, a_by_task[tid], b_by_task[tid])
            )
        for tid in sorted(b_hashes.keys()):
            if tid not in a_hashes:
                added.append(tid)

        return ComparisonReport(
            label_a=label_a or run_a.timestamp,
            label_b=label_b or run_b.timestamp,
            suite_hash_a=run_a.suite_hash,
            suite_hash_b=run_b.suite_hash,
            common_tasks=common,
            removed_tasks=removed,
            added_tasks=added,
            changed_def_tasks=changed_def,
            summary=SummaryDiff(a=run_a.summary, b=run_b.summary),
        )

    # -- 趋势 -------------------------------------------------------------

    def trend(self, last_n: int = 10) -> TrendReport:
        runs = self.list_runs(last_n=last_n)
        points = [
            TrendPoint(
                timestamp=r.timestamp,
                git_commit=r.git_commit,
                task_success_rate=r.summary.task_success_rate,
                tool_error_rate=r.summary.tool_error_rate,
                avg_cost_usd=r.summary.total_cost_usd / max(1, len(r.results)),
                avg_wall_time_seconds=r.summary.avg_wall_time_seconds,
                verifier_recovery_rate=r.summary.verifier_recovery_rate,
            )
            for r in runs
        ]
        return TrendReport(points=points)


# ---------------------------------------------------------------------------
# 内部工具
# ---------------------------------------------------------------------------


def _group_by_task(results: Iterable[TaskResult]) -> dict[str, list[TaskResult]]:
    by: dict[str, list[TaskResult]] = {}
    for r in results:
        by.setdefault(r.task_id, []).append(r)
    return by


def _any_hash(rs: list[TaskResult]) -> str:
    """取某任务一批 run 的 task_hash；同一个 run 内同 id 必然 hash 一致."""
    return rs[0].task_hash


def _build_task_diff(
    task_id: str,
    task_hash: str,
    a_runs: list[TaskResult],
    b_runs: list[TaskResult],
) -> TaskMetricDiff:
    def _pass_rate(rs: list[TaskResult]) -> float:
        return sum(1 for r in rs if r.passed) / len(rs)

    def _avg(rs: list[TaskResult], attr: str) -> float:
        return sum(getattr(r, attr) for r in rs) / len(rs)

    return TaskMetricDiff(
        task_id=task_id,
        task_hash=task_hash,
        pass_rate_a=_pass_rate(a_runs),
        pass_rate_b=_pass_rate(b_runs),
        avg_step_count_a=_avg(a_runs, "step_count"),
        avg_step_count_b=_avg(b_runs, "step_count"),
        avg_cost_usd_a=_avg(a_runs, "cost_usd"),
        avg_cost_usd_b=_avg(b_runs, "cost_usd"),
        avg_wall_time_a=_avg(a_runs, "wall_time_seconds"),
        avg_wall_time_b=_avg(b_runs, "wall_time_seconds"),
    )


# ---------------------------------------------------------------------------
# Sparkline 渲染
# ---------------------------------------------------------------------------


_SPARK_CHARS = "▁▂▃▄▅▆▇█"


def _sparkline(values: list[float]) -> str:
    """unicode 块字符 sparkline；空序列返回空串，常数序列返回"中间高"字符."""
    if not values:
        return ""
    lo, hi = min(values), max(values)
    if hi == lo:
        return _SPARK_CHARS[len(_SPARK_CHARS) // 2] * len(values)
    n = len(_SPARK_CHARS) - 1
    return "".join(
        _SPARK_CHARS[int(round((v - lo) / (hi - lo) * n))] for v in values
    )


# ---------------------------------------------------------------------------
# Rich 渲染
# ---------------------------------------------------------------------------


def _arrow(delta: float, *, higher_is_better: bool = True) -> Text:
    """返回带颜色的 ↑/↓/-；higher_is_better=False 时颜色反着来."""
    if delta > 0:
        good = higher_is_better
        return Text("↑", style="green" if good else "red")
    if delta < 0:
        good = not higher_is_better
        return Text("↓", style="green" if good else "red")
    return Text("·", style="dim")


def _fmt_delta(delta: float, *, higher_is_better: bool = True, fmt: str = "+.3f") -> Text:
    """带箭头 + 颜色的 delta 字符串."""
    arrow = _arrow(delta, higher_is_better=higher_is_better)
    color = arrow.style or ""
    return Text(f"{arrow.plain} {delta:{fmt}}", style=color)


def _render_comparison(rep: ComparisonReport) -> Group:
    parts: list = []

    # 顶部 meta
    header = Text.assemble(
        ("Compare: ", "bold"),
        (rep.label_a, "cyan"),
        " → ",
        (rep.label_b, "magenta"),
    )
    parts.append(header)
    if not rep.same_suite_hash:
        parts.append(Text(
            f"⚠ suite_hash 不同（a={rep.suite_hash_a[:12]} / b={rep.suite_hash_b[:12]}）"
            "：仅对 task_hash 相同的任务做 diff",
            style="yellow",
        ))

    # Summary diff 表
    if rep.summary is not None:
        t = Table(title="Summary", show_lines=False)
        t.add_column("metric")
        t.add_column("A", justify="right")
        t.add_column("B", justify="right")
        t.add_column("Δ", justify="right")
        for name, higher_better, fmt in (
            ("task_success_rate",        True,  "+.3f"),
            ("tool_error_rate",          False, "+.3f"),
            ("verifier_first_pass_rate", True,  "+.3f"),
            ("verifier_recovery_rate",   True,  "+.3f"),
            ("avg_step_count",           False, "+.2f"),
            ("total_cost_usd",           False, "+.4f"),
            ("avg_wall_time_seconds",    False, "+.2f"),
            ("avg_edit_precision",       True,  "+.3f"),
            ("avg_edit_recall",          True,  "+.3f"),
        ):
            a = getattr(rep.summary.a, name)
            b = getattr(rep.summary.b, name)
            t.add_row(
                name,
                f"{a:.4f}" if isinstance(a, float) else str(a),
                f"{b:.4f}" if isinstance(b, float) else str(b),
                _fmt_delta(b - a, higher_is_better=higher_better, fmt=fmt),
            )
        parts.append(t)

    # 共有任务
    if rep.common_tasks:
        t = Table(title=f"Common tasks ({len(rep.common_tasks)})")
        t.add_column("task")
        t.add_column("pass A→B", justify="right")
        t.add_column("Δ pass", justify="right")
        t.add_column("steps A→B", justify="right")
        t.add_column("cost A→B", justify="right")
        for d in rep.common_tasks:
            t.add_row(
                d.task_id,
                f"{d.pass_rate_a:.2f}→{d.pass_rate_b:.2f}",
                _fmt_delta(d.pass_rate_delta, higher_is_better=True, fmt="+.2f"),
                f"{d.avg_step_count_a:.1f}→{d.avg_step_count_b:.1f}",
                f"${d.avg_cost_usd_a:.4f}→${d.avg_cost_usd_b:.4f}",
            )
        parts.append(t)

    # 各类差异任务
    for title, items, style in (
        ("Removed (只在 A)", rep.removed_tasks, "red"),
        ("Added (只在 B)", rep.added_tasks, "green"),
        ("Def changed (同 id 不同 hash)", rep.changed_def_tasks, "yellow"),
    ):
        if items:
            parts.append(
                Panel(
                    Text(", ".join(items), style=style),
                    title=f"{title} ({len(items)})",
                    border_style=style,
                )
            )

    return Group(*parts)


def _render_trend(rep: TrendReport) -> Group:
    if not rep.points:
        return Group(Text("（还没有 eval 记录）", style="dim"))

    t = Table(title=f"Trend (最近 {len(rep.points)} 次)")
    t.add_column("metric")
    t.add_column("sparkline")
    t.add_column("first", justify="right")
    t.add_column("last", justify="right")
    t.add_column("Δ", justify="right")

    for attr, _field, label in _TREND_METRICS:
        series = rep.series(attr)
        first, last = series[0], series[-1]
        delta = last - first
        # 哪些指标越高越好
        higher_better = attr in {"task_success_rate", "verifier_recovery_rate"}
        t.add_row(
            label,
            rep.sparkline(attr),
            f"{first:.4f}",
            f"{last:.4f}",
            _fmt_delta(delta, higher_is_better=higher_better, fmt="+.4f"),
        )
    return Group(t)


__all__ = [
    "ComparisonReport",
    "EvalTracker",
    "SummaryDiff",
    "TaskMetricDiff",
    "TrendPoint",
    "TrendReport",
    "suite_result_from_dict",
    "suite_result_to_dict",
]
