"""`main.py eval` 子命令的测试：参数解析、分派、compare/trend、端到端跑单任务."""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import replace
from pathlib import Path
from typing import AsyncIterator

import pytest
from rich.console import Console

from mini_code_agent.cli.eval_cmd import (
    _find_run,
    add_eval_subparser,
    build_agent_factory,
    run_eval_command,
)
from mini_code_agent.core import Agent
from mini_code_agent.eval import (
    BenchmarkSuite,
    BenchmarkTask,
    EvalSummary,
    EvalTracker,
    SuiteResult,
    TaskResult,
)
from mini_code_agent.llm import (
    LLMClient,
    LLMResponse,
    Message,
    StreamDelta,
    TokenUsage,
    ToolCall,
    ToolParam,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
TASKS_DIR = REPO_ROOT / "eval" / "tasks"


# ---------------------------------------------------------------------------
# 辅助：parser + 结果 fixture
# ---------------------------------------------------------------------------


def _make_parser() -> argparse.ArgumentParser:
    """只装 eval 子命令的一个裸 parser（不走 main.py）."""
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command")
    add_eval_subparser(sub)
    return p


def _task_result(task_id: str = "A", task_hash: str = "h",
                 run_index: int = 0, passed: bool = True) -> TaskResult:
    return TaskResult(
        task_id=task_id, task_hash=task_hash, run_index=run_index,
        passed=passed, stop_reason="ok", step_count=3, tool_error_count=0,
        prompt_tokens=100, completion_tokens=50, cost_usd=0.01,
        wall_time_seconds=1.0, verifier_first_passed=None,
        verifier_final_passed=None, files_changed_actual=[],
        edit_precision=1.0, edit_recall=1.0, failure_category=None,
        validation_details="",
    )


def _summary(**kwargs) -> EvalSummary:
    base = dict(
        task_success_rate=1.0, by_level={1: 1.0}, by_failure_category={},
        avg_step_count=3.0, tool_error_rate=0.0,
        verifier_first_pass_rate=0.0, verifier_recovery_rate=0.0,
        avg_prompt_tokens=100.0, avg_completion_tokens=50.0,
        total_cost_usd=0.01, avg_wall_time_seconds=1.0,
        avg_edit_precision=1.0, avg_edit_recall=1.0,
    )
    base.update(kwargs)
    return EvalSummary(**base)


def _suite_result(timestamp: str, *, suite_hash: str = "s",
                  git_commit: str = "abcdef1234567890",
                  success: float = 1.0) -> SuiteResult:
    return SuiteResult(
        timestamp=timestamp, git_commit=git_commit,
        suite_hash=suite_hash, model_name="mock",
        results=[_task_result(passed=success >= 1.0)],
        summary=_summary(task_success_rate=success),
    )


# ---------------------------------------------------------------------------
# argparse 配置
# ---------------------------------------------------------------------------


class TestEvalSubparser:
    def test_defaults(self) -> None:
        args = _make_parser().parse_args(["eval"])
        assert args.command == "eval"
        assert args.level is None
        assert args.task is None
        assert args.tag is None
        assert args.runs == 3
        assert args.parallel == 1
        assert args.no_save is False
        assert args.compare is None
        assert args.trend is None
        assert args.tasks_dir == "eval/tasks"
        assert args.results_dir == "eval/results"

    def test_level_and_task_filters(self) -> None:
        args = _make_parser().parse_args(
            ["eval", "--level", "1", "--task", "X", "--tag", "python"]
        )
        assert args.level == 1 and args.task == "X" and args.tag == "python"

    def test_trend_bare_defaults_to_10(self) -> None:
        args = _make_parser().parse_args(["eval", "--trend"])
        assert args.trend == 10

    def test_trend_custom_n(self) -> None:
        args = _make_parser().parse_args(["eval", "--trend", "5"])
        assert args.trend == 5

    def test_compare_empty_list(self) -> None:
        args = _make_parser().parse_args(["eval", "--compare"])
        assert args.compare == []

    def test_compare_two_runs(self) -> None:
        args = _make_parser().parse_args(["eval", "--compare", "a", "b"])
        assert args.compare == ["a", "b"]

    def test_no_command(self) -> None:
        args = _make_parser().parse_args([])
        assert args.command is None


# ---------------------------------------------------------------------------
# --trend 分支
# ---------------------------------------------------------------------------


class TestTrendCommand:
    async def test_empty_results_still_exits_zero(
        self, tmp_path: Path, capsys,
    ) -> None:
        args = _make_parser().parse_args([
            "eval", "--trend",
            "--results-dir", str(tmp_path),
        ])
        rc = await run_eval_command(args)
        assert rc == 0
        out = capsys.readouterr().out
        assert "还没有" in out

    async def test_renders_recent_runs(self, tmp_path: Path, capsys) -> None:
        tracker = EvalTracker(tmp_path)
        for i, ts in enumerate([
            "2026-04-15T10:00:00+00:00",
            "2026-04-16T10:00:00+00:00",
            "2026-04-17T10:00:00+00:00",
        ]):
            tracker.save(_suite_result(ts, success=0.4 + 0.2 * i))

        args = _make_parser().parse_args([
            "eval", "--trend", "2",
            "--results-dir", str(tmp_path),
        ])
        rc = await run_eval_command(args)
        assert rc == 0
        out = capsys.readouterr().out
        assert "Trend" in out
        assert "Success" in out


# ---------------------------------------------------------------------------
# --compare 分支
# ---------------------------------------------------------------------------


class TestCompareCommand:
    async def test_fewer_than_two_runs_returns_2(
        self, tmp_path: Path, capsys,
    ) -> None:
        # 0 runs
        args = _make_parser().parse_args([
            "eval", "--compare", "--results-dir", str(tmp_path),
        ])
        rc = await run_eval_command(args)
        assert rc == 2
        assert "至少需要 2 次" in capsys.readouterr().out

    async def test_compare_latest_two(self, tmp_path: Path, capsys) -> None:
        tracker = EvalTracker(tmp_path)
        tracker.save(_suite_result("2026-04-15T10:00:00+00:00", success=0.3))
        tracker.save(_suite_result("2026-04-16T10:00:00+00:00", success=0.5))
        tracker.save(_suite_result("2026-04-17T10:00:00+00:00", success=0.9))

        args = _make_parser().parse_args([
            "eval", "--compare", "--results-dir", str(tmp_path),
        ])
        rc = await run_eval_command(args)
        assert rc == 0
        out = capsys.readouterr().out
        # 默认对比最近两次 = 2026-04-16 → 2026-04-17
        assert "2026-04-16" in out
        assert "2026-04-17" in out
        # 2026-04-15 不出现在对比里
        assert "2026-04-15" not in out

    async def test_compare_by_timestamp_substring(
        self, tmp_path: Path, capsys,
    ) -> None:
        tracker = EvalTracker(tmp_path)
        tracker.save(_suite_result("2026-04-15T10:00:00+00:00", success=0.3))
        tracker.save(_suite_result("2026-04-16T10:00:00+00:00", success=0.5))
        tracker.save(_suite_result("2026-04-17T10:00:00+00:00", success=0.9))

        args = _make_parser().parse_args([
            "eval", "--compare", "04-15", "04-17",
            "--results-dir", str(tmp_path),
        ])
        rc = await run_eval_command(args)
        assert rc == 0
        out = capsys.readouterr().out
        assert "04-15" in out and "04-17" in out

    async def test_compare_wrong_arity_returns_2(
        self, tmp_path: Path, capsys,
    ) -> None:
        tracker = EvalTracker(tmp_path)
        tracker.save(_suite_result("2026-04-16T10:00:00+00:00"))
        tracker.save(_suite_result("2026-04-17T10:00:00+00:00"))
        args = _make_parser().parse_args([
            "eval", "--compare", "only-one",
            "--results-dir", str(tmp_path),
        ])
        rc = await run_eval_command(args)
        assert rc == 2
        assert "要么" in capsys.readouterr().out

    async def test_compare_unknown_run_returns_2(
        self, tmp_path: Path, capsys,
    ) -> None:
        tracker = EvalTracker(tmp_path)
        tracker.save(_suite_result("2026-04-16T10:00:00+00:00"))
        tracker.save(_suite_result("2026-04-17T10:00:00+00:00"))
        args = _make_parser().parse_args([
            "eval", "--compare", "nope", "04-17",
            "--results-dir", str(tmp_path),
        ])
        rc = await run_eval_command(args)
        assert rc == 2
        assert "找不到" in capsys.readouterr().out


class TestFindRun:
    """_find_run 单测（compare 的底层模糊匹配）."""

    def test_by_timestamp(self) -> None:
        runs = [
            _suite_result("2026-04-15T10:00:00+00:00"),
            _suite_result("2026-04-17T10:00:00+00:00"),
        ]
        console = Console()
        r = _find_run(runs, "04-15", console)
        assert r is not None and r.timestamp.startswith("2026-04-15")

    def test_by_commit_prefix(self) -> None:
        runs = [
            _suite_result("2026-04-15T10:00:00+00:00",
                          git_commit="abc123xxxxxxxxx"),
            _suite_result("2026-04-17T10:00:00+00:00",
                          git_commit="def456xxxxxxxxx"),
        ]
        r = _find_run(runs, "def456", Console())
        assert r is not None and r.timestamp.startswith("2026-04-17")

    def test_ambiguous_returns_none(self) -> None:
        runs = [
            _suite_result("2026-04-15T10:00:00+00:00"),
            _suite_result("2026-04-17T10:00:00+00:00"),
        ]
        r = _find_run(runs, "2026", Console())
        assert r is None


# ---------------------------------------------------------------------------
# eval 主流程（过滤 + 端到端 mock factory）
# ---------------------------------------------------------------------------


class _ScriptedLLM(LLMClient):
    def __init__(self, responses: list[LLMResponse]) -> None:
        super().__init__(model="mock-llm")
        self._responses = list(responses)
        self._i = 0

    async def chat(
        self, messages: list[Message], tools: list[ToolParam] | None = None,
    ) -> LLMResponse:
        r = self._responses[min(self._i, len(self._responses) - 1)]
        self._i += 1
        self._accumulate_usage(r.usage)
        return r

    def chat_stream(
        self, messages: list[Message], tools: list[ToolParam] | None = None,
    ) -> AsyncIterator[StreamDelta]:
        raise NotImplementedError


_L1_TASK_DIR = TASKS_DIR / "L1-add-function"


class TestBuildAgentFactory:
    def test_factory_binds_task_limits(self, tmp_path: Path) -> None:
        """factory 应把 task.max_steps/max_tokens/max_wall_time 绑到 Agent 的安全层."""
        task = BenchmarkTask.load(_L1_TASK_DIR)
        # 改一个明显的值确认真的被绑进去
        task = replace(task, max_steps=7, max_tokens=333, max_wall_time_seconds=9)

        llm = _ScriptedLLM([LLMResponse(content="done", usage=TokenUsage(1, 1))])
        factory = build_agent_factory(llm)

        workspace = tmp_path / "ws"
        workspace.mkdir()
        agent = factory(workspace, task)

        assert agent.loop_guard is not None
        assert agent.loop_guard.max_rounds == 7
        assert agent.loop_guard.max_tokens == 333
        assert agent.max_wall_time_seconds == 9
        assert agent.file_guard is not None

        # BashTool 注册了，且 cwd 指向 workspace
        bash = agent.tool_registry.get("Bash")
        assert bash is not None
        assert bash.cwd == str(workspace)


class TestRunEvalEndToEnd:
    """用 monkeypatch 把 create_client 换成 mock，端到端跑单任务."""

    async def test_task_not_found_returns_2(
        self, tmp_path: Path, monkeypatch, capsys,
    ) -> None:
        # 复制真实 tasks 目录（不污染仓库）
        tasks = tmp_path / "tasks"
        shutil.copytree(TASKS_DIR, tasks)

        monkeypatch.setattr(
            "mini_code_agent.cli.eval_cmd.create_client",
            lambda **_kw: _ScriptedLLM([LLMResponse("ok", usage=TokenUsage(1, 1))]),
        )

        args = _make_parser().parse_args([
            "eval",
            "--task", "does-not-exist",
            "--tasks-dir", str(tasks),
            "--results-dir", str(tmp_path / "results"),
            "--no-save",
        ])
        rc = await run_eval_command(args)
        assert rc == 2
        assert "任务不存在" in capsys.readouterr().out

    async def test_level_filter_eliminates_all_returns_2(
        self, tmp_path: Path, monkeypatch, capsys,
    ) -> None:
        tasks = tmp_path / "tasks"
        shutil.copytree(TASKS_DIR, tasks)

        monkeypatch.setattr(
            "mini_code_agent.cli.eval_cmd.create_client",
            lambda **_kw: _ScriptedLLM([LLMResponse("ok", usage=TokenUsage(1, 1))]),
        )

        args = _make_parser().parse_args([
            "eval", "--level", "3",
            "--tasks-dir", str(tasks),
            "--no-save",
        ])
        rc = await run_eval_command(args)
        assert rc == 2
        assert "没有任务" in capsys.readouterr().out

    async def test_tasks_dir_missing_returns_2(
        self, tmp_path: Path, monkeypatch, capsys,
    ) -> None:
        monkeypatch.setattr(
            "mini_code_agent.cli.eval_cmd.create_client",
            lambda **_kw: _ScriptedLLM([LLMResponse("ok", usage=TokenUsage(1, 1))]),
        )

        args = _make_parser().parse_args([
            "eval",
            "--tasks-dir", str(tmp_path / "no-such-dir"),
            "--no-save",
        ])
        rc = await run_eval_command(args)
        assert rc == 2
        assert "tasks 目录不存在" in capsys.readouterr().out

    async def test_full_run_no_save(
        self, tmp_path: Path, monkeypatch, capsys,
    ) -> None:
        """跑 L1-add-function 一次，用 noop LLM → validation_fail → rc=1；结果不落盘."""
        tasks = tmp_path / "tasks"
        shutil.copytree(TASKS_DIR, tasks)
        results = tmp_path / "results"

        monkeypatch.setattr(
            "mini_code_agent.cli.eval_cmd.create_client",
            lambda **_kw: _ScriptedLLM([
                LLMResponse(content="我不改", usage=TokenUsage(50, 10)),
            ]),
        )

        args = _make_parser().parse_args([
            "eval",
            "--task", "L1-add-function",
            "--runs", "1",
            "--tasks-dir", str(tasks),
            "--results-dir", str(results),
            "--no-save",
            "--model", "deepseek-chat",  # 用已知定价避免 unknown 警告
        ])
        rc = await run_eval_command(args)
        assert rc == 1  # 没通过
        out = capsys.readouterr().out
        assert "Eval Summary" in out
        assert "Per-task" in out
        # Rich 表格可能按列宽截断，断言看得见的前缀/关键子串
        assert "L1-add-funct" in out
        assert "validation_" in out
        # --no-save：results dir 保持空/不存在
        assert not results.exists() or not any(results.iterdir())

    async def test_run_saves_result_when_save_enabled(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        """即便任务失败，没带 --no-save 时也该落盘结果（tracker 的 save 被调用）."""
        tasks = tmp_path / "tasks"
        shutil.copytree(TASKS_DIR, tasks)
        results = tmp_path / "results"

        monkeypatch.setattr(
            "mini_code_agent.cli.eval_cmd.create_client",
            lambda **_kw: _ScriptedLLM([
                LLMResponse(content="我不改", usage=TokenUsage(50, 10)),
            ]),
        )

        args = _make_parser().parse_args([
            "eval",
            "--task", "L1-add-function",
            "--runs", "1",
            "--tasks-dir", str(tasks),
            "--results-dir", str(results),
            "--model", "deepseek-chat",
        ])
        rc = await run_eval_command(args)
        assert rc == 1  # noop LLM → validation_fail，但仍要落盘
        files = list(results.glob("*.json"))
        assert len(files) == 1
        data = json.loads(files[0].read_text())
        assert data["model_name"] == "deepseek-chat"
        assert data["summary"]["task_success_rate"] == 0.0
        assert data["results"][0]["failure_category"] == "validation_fail"

    async def test_unknown_model_warns_and_still_runs(
        self, tmp_path: Path, monkeypatch, capsys,
    ) -> None:
        tasks = tmp_path / "tasks"
        shutil.copytree(TASKS_DIR, tasks)

        monkeypatch.setattr(
            "mini_code_agent.cli.eval_cmd.create_client",
            lambda **_kw: _ScriptedLLM([
                LLMResponse(content="noop", usage=TokenUsage(50, 10)),
            ]),
        )

        args = _make_parser().parse_args([
            "eval",
            "--task", "L1-add-function",
            "--runs", "1",
            "--tasks-dir", str(tasks),
            "--results-dir", str(tmp_path / "r"),
            "--no-save",
            "--model", "some-new-unknown-model",
        ])
        rc = await run_eval_command(args)
        # 跑完了（rc=1 因为 validation_fail），只要没在定价那一步崩就算对
        assert rc in (0, 1)
        out = capsys.readouterr().out
        assert "未知模型" in out
        assert "cost_usd 将按 0 计" in out
