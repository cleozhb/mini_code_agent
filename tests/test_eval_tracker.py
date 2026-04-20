"""EvalTracker 测试：save/list_runs/compare/trend + JSON 往返 + sparkline."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
from rich.console import Console

from mini_code_agent.eval import (
    ComparisonReport,
    EvalSummary,
    EvalTracker,
    SuiteResult,
    TaskResult,
    TrendReport,
    suite_result_from_dict,
    suite_result_to_dict,
)
from mini_code_agent.eval.tracker import _sparkline


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


def _make_task_result(
    *,
    task_id: str = "A",
    task_hash: str = "h-A",
    run_index: int = 0,
    passed: bool = True,
    stop_reason: str = "ok",
    step_count: int = 5,
    tool_error_count: int = 0,
    prompt_tokens: int = 100,
    completion_tokens: int = 50,
    cost_usd: float = 0.01,
    wall_time_seconds: float = 1.0,
    verifier_first_passed: bool | None = None,
    verifier_final_passed: bool | None = None,
    files_changed_actual: list[str] | None = None,
    edit_precision: float = 1.0,
    edit_recall: float = 1.0,
    failure_category: str | None = None,
    validation_details: str = "",
) -> TaskResult:
    return TaskResult(
        task_id=task_id,
        task_hash=task_hash,
        run_index=run_index,
        passed=passed,
        stop_reason=stop_reason,
        step_count=step_count,
        tool_error_count=tool_error_count,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cost_usd=cost_usd,
        wall_time_seconds=wall_time_seconds,
        verifier_first_passed=verifier_first_passed,
        verifier_final_passed=verifier_final_passed,
        files_changed_actual=files_changed_actual or [],
        edit_precision=edit_precision,
        edit_recall=edit_recall,
        failure_category=failure_category,
        validation_details=validation_details,
    )


def _make_summary(**overrides) -> EvalSummary:
    base = dict(
        task_success_rate=1.0,
        by_level={1: 1.0},
        by_failure_category={},
        avg_step_count=5.0,
        tool_error_rate=0.0,
        verifier_first_pass_rate=None,
        verifier_recovery_rate=None,
        avg_prompt_tokens=100.0,
        avg_completion_tokens=50.0,
        total_cost_usd=0.02,
        avg_wall_time_seconds=1.0,
        avg_edit_precision=1.0,
        avg_edit_recall=1.0,
    )
    base.update(overrides)
    return EvalSummary(**base)


def _make_suite_result(
    *,
    timestamp: str = "2026-04-17T10:00:00+00:00",
    git_commit: str | None = "abcdef1234567890",
    suite_hash: str = "s-hash",
    model_name: str = "mock",
    results: list[TaskResult] | None = None,
    summary: EvalSummary | None = None,
) -> SuiteResult:
    if results is None:
        results = [_make_task_result()]
    return SuiteResult(
        timestamp=timestamp,
        git_commit=git_commit,
        suite_hash=suite_hash,
        model_name=model_name,
        results=results,
        summary=summary or _make_summary(),
    )


# ---------------------------------------------------------------------------
# JSON 序列化往返
# ---------------------------------------------------------------------------


class TestJSONRoundtrip:
    def test_roundtrip_preserves_all_fields(self) -> None:
        sr = _make_suite_result(
            results=[
                _make_task_result(
                    task_id="A",
                    verifier_first_passed=False,
                    verifier_final_passed=True,
                    files_changed_actual=["x.py", "y.py"],
                    failure_category=None,
                ),
                _make_task_result(
                    task_id="B",
                    task_hash="h-B",
                    passed=False,
                    failure_category="validation_fail",
                ),
            ],
            summary=_make_summary(
                by_failure_category={"validation_fail": 1},
                task_success_rate=0.5,
            ),
        )
        data = suite_result_to_dict(sr)
        j = json.loads(json.dumps(data))  # 过一遍 JSON 才真测得出有没有不可序列化字段
        back = suite_result_from_dict(j)

        assert back.timestamp == sr.timestamp
        assert back.git_commit == sr.git_commit
        assert back.suite_hash == sr.suite_hash
        assert back.model_name == sr.model_name
        assert len(back.results) == 2
        assert back.results[0].task_id == "A"
        assert back.results[0].verifier_first_passed is False
        assert back.results[0].files_changed_actual == ["x.py", "y.py"]
        assert back.results[1].failure_category == "validation_fail"
        assert back.summary.task_success_rate == 0.5
        assert back.summary.by_failure_category == {"validation_fail": 1}

    def test_from_dict_tolerates_unknown_fields(self) -> None:
        sr = _make_suite_result()
        data = suite_result_to_dict(sr)
        data["results"][0]["future_field_claude_added"] = "ignored"
        data["future_top_level"] = 42
        # 不应抛异常
        back = suite_result_from_dict(data)
        assert back.results[0].task_id == "A"

    def test_from_dict_tolerates_missing_optional_fields(self) -> None:
        sr = _make_suite_result()
        data = suite_result_to_dict(sr)
        # 模拟老文件没有 workspace_path
        data["results"][0].pop("workspace_path", None)
        back = suite_result_from_dict(data)
        assert back.results[0].workspace_path is None


# ---------------------------------------------------------------------------
# EvalTracker.save + list_runs
# ---------------------------------------------------------------------------


class TestSaveAndList:
    def test_save_creates_file_and_parses_back(self, tmp_path: Path) -> None:
        tracker = EvalTracker(tmp_path)
        sr = _make_suite_result()
        path = tracker.save(sr)
        assert path.exists()
        assert path.suffix == ".json"
        # 文件名含 commit 短 hash
        assert "abcdef123456" in path.name

        loaded = json.loads(path.read_text())
        assert loaded["suite_hash"] == "s-hash"

    def test_same_commit_seq_increments_no_overwrite(self, tmp_path: Path) -> None:
        tracker = EvalTracker(tmp_path)
        sr1 = _make_suite_result(timestamp="2026-04-17T10:00:00+00:00")
        sr2 = _make_suite_result(timestamp="2026-04-17T10:05:00+00:00")
        sr3 = _make_suite_result(timestamp="2026-04-17T10:10:00+00:00")

        p1 = tracker.save(sr1)
        p2 = tracker.save(sr2)
        p3 = tracker.save(sr3)

        # 三个文件都存在，不互相覆盖
        assert p1 != p2 != p3
        assert p1.exists() and p2.exists() and p3.exists()
        # seq 应该是 00 / 01 / 02
        seqs = sorted(int(p.stem.rsplit("__", 1)[-1]) for p in (p1, p2, p3))
        assert seqs == [0, 1, 2]

    def test_different_commit_seq_starts_from_zero(self, tmp_path: Path) -> None:
        tracker = EvalTracker(tmp_path)
        tracker.save(_make_suite_result(git_commit="aaaaaaaaaaaa"))
        tracker.save(_make_suite_result(git_commit="aaaaaaaaaaaa"))
        p3 = tracker.save(_make_suite_result(git_commit="bbbbbbbbbbbb"))
        # 新 commit 从 00 开始
        assert p3.stem.endswith("__00")

    def test_nocommit_when_git_commit_none(self, tmp_path: Path) -> None:
        tracker = EvalTracker(tmp_path)
        p = tracker.save(_make_suite_result(git_commit=None))
        assert "nocommit" in p.name

    def test_timestamp_colons_sanitized(self, tmp_path: Path) -> None:
        tracker = EvalTracker(tmp_path)
        p = tracker.save(_make_suite_result(timestamp="2026-04-17T10:00:00+00:00"))
        # 文件名里不能有冒号（Windows 兼容性 + 简化解析）
        assert ":" not in p.name

    def test_list_runs_empty_dir(self, tmp_path: Path) -> None:
        tracker = EvalTracker(tmp_path / "does-not-exist")
        assert tracker.list_runs() == []

    def test_list_runs_sorted_by_timestamp(self, tmp_path: Path) -> None:
        tracker = EvalTracker(tmp_path)
        tracker.save(_make_suite_result(timestamp="2026-04-17T12:00:00+00:00", suite_hash="S3"))
        tracker.save(_make_suite_result(timestamp="2026-04-17T10:00:00+00:00", suite_hash="S1"))
        tracker.save(_make_suite_result(timestamp="2026-04-17T11:00:00+00:00", suite_hash="S2"))

        runs = tracker.list_runs()
        assert [r.suite_hash for r in runs] == ["S1", "S2", "S3"]

    def test_list_runs_last_n(self, tmp_path: Path) -> None:
        tracker = EvalTracker(tmp_path)
        for i, ts in enumerate([
            "2026-04-17T10:00:00+00:00",
            "2026-04-17T11:00:00+00:00",
            "2026-04-17T12:00:00+00:00",
            "2026-04-17T13:00:00+00:00",
        ]):
            tracker.save(_make_suite_result(timestamp=ts, suite_hash=f"S{i}"))

        latest_two = tracker.list_runs(last_n=2)
        assert [r.suite_hash for r in latest_two] == ["S2", "S3"]

    def test_list_runs_ignores_non_matching_files(self, tmp_path: Path) -> None:
        tracker = EvalTracker(tmp_path)
        tracker.save(_make_suite_result(suite_hash="good"))
        # 乱扔一个不符合命名的文件，不应影响 list
        (tmp_path / "random.json").write_text("{}")
        (tmp_path / "notes.md").write_text("# hi")

        runs = tracker.list_runs()
        assert len(runs) == 1
        assert runs[0].suite_hash == "good"

    def test_list_runs_skips_corrupt_json(self, tmp_path: Path) -> None:
        tracker = EvalTracker(tmp_path)
        good_path = tracker.save(_make_suite_result(suite_hash="ok"))
        # 手造一个符合命名但内容损坏的文件
        (tmp_path / "2026-04-17T09-00-00+00-00__deadbeef0000__00.json").write_text("not json")

        runs = tracker.list_runs()
        # 损坏的被跳过，好的仍在
        assert len(runs) == 1
        assert runs[0].suite_hash == "ok"


# ---------------------------------------------------------------------------
# EvalTracker.compare
# ---------------------------------------------------------------------------


class TestCompare:
    def _sr(self, results: list[TaskResult], *, suite_hash: str = "s") -> SuiteResult:
        return _make_suite_result(results=results, suite_hash=suite_hash)

    def test_common_tasks_simple_diff(self, tmp_path: Path) -> None:
        tracker = EvalTracker(tmp_path)
        a = self._sr([
            _make_task_result(task_id="A", task_hash="h-A", passed=False,
                              step_count=10, cost_usd=0.05, wall_time_seconds=4.0),
        ])
        b = self._sr([
            _make_task_result(task_id="A", task_hash="h-A", passed=True,
                              step_count=6, cost_usd=0.02, wall_time_seconds=2.0),
        ])
        rep = tracker.compare(a, b)
        assert len(rep.common_tasks) == 1
        d = rep.common_tasks[0]
        assert d.task_id == "A"
        assert d.pass_rate_a == 0.0
        assert d.pass_rate_b == 1.0
        assert d.pass_rate_delta == 1.0
        assert d.avg_step_count_a == 10.0
        assert d.avg_step_count_b == 6.0
        assert d.avg_cost_usd_a == pytest.approx(0.05)
        assert d.avg_cost_usd_b == pytest.approx(0.02)

    def test_intersect_when_suite_hash_differs(self, tmp_path: Path) -> None:
        """DESIGN.md §9.1: suite_hash 不同时只对比 task_hash 相同的任务."""
        tracker = EvalTracker(tmp_path)
        a = self._sr([
            _make_task_result(task_id="A", task_hash="h-A", passed=True),
            _make_task_result(task_id="B", task_hash="h-B", passed=True),
        ], suite_hash="suite-v1")
        b = self._sr([
            _make_task_result(task_id="A", task_hash="h-A", passed=True),
            _make_task_result(task_id="C", task_hash="h-C", passed=True),
        ], suite_hash="suite-v2")

        rep = tracker.compare(a, b)
        assert not rep.same_suite_hash
        assert [d.task_id for d in rep.common_tasks] == ["A"]
        assert rep.removed_tasks == ["B"]
        assert rep.added_tasks == ["C"]
        assert rep.changed_def_tasks == []

    def test_same_task_id_different_hash_is_changed_def(self, tmp_path: Path) -> None:
        tracker = EvalTracker(tmp_path)
        a = self._sr([_make_task_result(task_id="A", task_hash="h-old", passed=True)])
        b = self._sr([_make_task_result(task_id="A", task_hash="h-new", passed=True)])
        rep = tracker.compare(a, b)
        assert rep.common_tasks == []
        assert rep.changed_def_tasks == ["A"]
        assert rep.removed_tasks == []
        assert rep.added_tasks == []

    def test_multi_run_task_pass_rate_aggregated(self, tmp_path: Path) -> None:
        """同一任务多 run 时 pass_rate 按 #pass / #runs 算."""
        tracker = EvalTracker(tmp_path)
        a = self._sr([
            _make_task_result(task_id="A", task_hash="h", run_index=0, passed=True),
            _make_task_result(task_id="A", task_hash="h", run_index=1, passed=True),
            _make_task_result(task_id="A", task_hash="h", run_index=2, passed=False),
        ])
        b = self._sr([
            _make_task_result(task_id="A", task_hash="h", run_index=0, passed=True),
            _make_task_result(task_id="A", task_hash="h", run_index=1, passed=True),
            _make_task_result(task_id="A", task_hash="h", run_index=2, passed=True),
        ])
        rep = tracker.compare(a, b)
        [d] = rep.common_tasks
        assert d.pass_rate_a == pytest.approx(2 / 3)
        assert d.pass_rate_b == 1.0

    def test_summary_diff_exposes_deltas(self, tmp_path: Path) -> None:
        tracker = EvalTracker(tmp_path)
        a = self._sr([_make_task_result()], )
        a = replace(a, summary=_make_summary(task_success_rate=0.6, tool_error_rate=0.1))
        b = replace(a, summary=_make_summary(task_success_rate=0.9, tool_error_rate=0.05))
        rep = tracker.compare(a, b)
        assert rep.summary.delta("task_success_rate") == pytest.approx(0.3)
        assert rep.summary.delta("tool_error_rate") == pytest.approx(-0.05)

    def test_render_smoke(self, tmp_path: Path) -> None:
        """render 不崩、输出含关键字段."""
        tracker = EvalTracker(tmp_path)
        a = self._sr([
            _make_task_result(task_id="A", task_hash="h", passed=False),
            _make_task_result(task_id="GONE", task_hash="h-gone", passed=True),
        ], suite_hash="v1")
        b = self._sr([
            _make_task_result(task_id="A", task_hash="h", passed=True),
            _make_task_result(task_id="NEW", task_hash="h-new", passed=True),
        ], suite_hash="v2")
        rep = tracker.compare(a, b, label_a="run-1", label_b="run-2")

        console = Console(file=None, record=True, width=120, color_system=None)
        console.print(rep.render())
        text = console.export_text()

        assert "run-1" in text and "run-2" in text
        assert "Compare" in text
        assert "suite_hash 不同" in text  # 被警示
        assert "GONE" in text  # removed
        assert "NEW" in text   # added
        # common 任务的 task id
        assert "A" in text


# ---------------------------------------------------------------------------
# EvalTracker.trend
# ---------------------------------------------------------------------------


class TestTrend:
    def test_empty(self, tmp_path: Path) -> None:
        tracker = EvalTracker(tmp_path)
        rep = tracker.trend()
        assert rep.points == []
        # 渲染空趋势不崩
        console = Console(record=True, width=80, color_system=None)
        console.print(rep.render())
        assert "还没有" in console.export_text()

    def test_collects_points_in_order(self, tmp_path: Path) -> None:
        tracker = EvalTracker(tmp_path)
        for i, ts in enumerate([
            "2026-04-10T10:00:00+00:00",
            "2026-04-11T10:00:00+00:00",
            "2026-04-12T10:00:00+00:00",
        ]):
            tracker.save(_make_suite_result(
                timestamp=ts,
                summary=_make_summary(
                    task_success_rate=0.3 + 0.2 * i,
                    tool_error_rate=0.1 - 0.03 * i,
                ),
            ))
        rep = tracker.trend(last_n=10)
        assert [p.timestamp for p in rep.points] == [
            "2026-04-10T10:00:00+00:00",
            "2026-04-11T10:00:00+00:00",
            "2026-04-12T10:00:00+00:00",
        ]
        assert rep.series("task_success_rate") == pytest.approx([0.3, 0.5, 0.7])

    def test_last_n_respected(self, tmp_path: Path) -> None:
        tracker = EvalTracker(tmp_path)
        for i in range(5):
            tracker.save(_make_suite_result(
                timestamp=f"2026-04-1{i}T10:00:00+00:00",
                summary=_make_summary(task_success_rate=i / 10.0),
            ))
        rep = tracker.trend(last_n=2)
        assert len(rep.points) == 2
        # 最后两次：i=3, 4 → 0.3, 0.4
        assert rep.series("task_success_rate") == pytest.approx([0.3, 0.4])

    def test_sparkline_renders_non_empty(self, tmp_path: Path) -> None:
        tracker = EvalTracker(tmp_path)
        for i in range(3):
            tracker.save(_make_suite_result(
                timestamp=f"2026-04-1{i}T10:00:00+00:00",
                summary=_make_summary(task_success_rate=i / 10.0),
            ))
        rep = tracker.trend()
        s = rep.sparkline("task_success_rate")
        assert len(s) == 3
        # 递增序列 → 最后一位是最高字符
        assert s[-1] == "█"
        assert s[0] == "▁"

    def test_render_smoke(self, tmp_path: Path) -> None:
        tracker = EvalTracker(tmp_path)
        for i in range(4):
            tracker.save(_make_suite_result(
                timestamp=f"2026-04-1{i}T10:00:00+00:00",
                summary=_make_summary(
                    task_success_rate=i / 10.0,
                    tool_error_rate=0.1,
                    total_cost_usd=0.01 * (i + 1),
                ),
            ))
        rep = tracker.trend()
        console = Console(record=True, width=120, color_system=None)
        console.print(rep.render())
        text = console.export_text()
        assert "Trend" in text
        assert "Success" in text
        assert "ToolErr" in text
        assert "Cost$" in text


# ---------------------------------------------------------------------------
# _sparkline
# ---------------------------------------------------------------------------


class TestSparkline:
    def test_empty(self) -> None:
        assert _sparkline([]) == ""

    def test_constant(self) -> None:
        # 全等 → 返回中间高字符 × n（避免除零；"▁▂▃▄▅▆▇█"[4] == "▅"）
        assert _sparkline([0.5, 0.5, 0.5]) == "▅▅▅"

    def test_monotonic(self) -> None:
        s = _sparkline([0.0, 0.25, 0.5, 0.75, 1.0])
        assert len(s) == 5
        assert s[0] == "▁"
        assert s[-1] == "█"
        # 单调非递减
        for a, b in zip(s, s[1:]):
            assert a <= b

    def test_ranges_normalized(self) -> None:
        # 相对高低而非绝对；10 和 20 的 sparkline 与 0.1 和 0.2 的相同
        assert _sparkline([10.0, 20.0]) == _sparkline([0.1, 0.2])

    def test_all_none(self) -> None:
        # eval 模式下 verifier_recovery_rate 一列会全是 None：返回空串而不是报错
        assert _sparkline([None, None, None]) == ""

    def test_partial_none(self) -> None:
        # None 位置用空格占位，非 None 位置按有效值的 min/max 归一化
        s = _sparkline([1.0, None, 2.0, 3.0])
        assert len(s) == 4
        assert s[1] == " "
        assert s[0] == "▁"
        assert s[-1] == "█"
