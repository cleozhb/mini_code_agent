"""Eval runner 测试：纯函数单测 + MockLLM 端到端."""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import replace
from pathlib import Path
from typing import AsyncIterator

import pytest

from mini_code_agent.core import Agent
from mini_code_agent.eval import (
    BenchmarkSuite,
    BenchmarkTask,
    EvalRunner,
    KNOWN_MODELS,
    ModelPricing,
    TaskResult,
    classify_failure,
    compute_edit_metrics,
    compute_summary,
    pricing_for,
    run_validate_script,
)
from mini_code_agent.eval.runner import _chdir
from mini_code_agent.llm import (
    LLMClient,
    LLMResponse,
    Message,
    StreamDelta,
    TokenUsage,
    ToolCall,
    ToolParam,
)
from mini_code_agent.safety import FileGuard
from mini_code_agent.tools import ToolRegistry, WriteFileTool


REPO_ROOT = Path(__file__).resolve().parents[1]
TASKS_DIR = REPO_ROOT / "eval" / "tasks"


# ---------------------------------------------------------------------------
# Mock LLM 客户端们
# ---------------------------------------------------------------------------


class _ScriptedLLM(LLMClient):
    """按脚本顺序吐响应的 Mock."""

    def __init__(self, responses: list[LLMResponse]) -> None:
        super().__init__(model="mock")
        self._responses = list(responses)
        self._i = 0

    async def chat(
        self,
        messages: list[Message],
        tools: list[ToolParam] | None = None,
    ) -> LLMResponse:
        if self._i >= len(self._responses):
            raise RuntimeError("Mock 响应已耗尽")
        r = self._responses[self._i]
        self._i += 1
        self._accumulate_usage(r.usage)
        return r

    def chat_stream(
        self,
        messages: list[Message],
        tools: list[ToolParam] | None = None,
    ) -> AsyncIterator[StreamDelta]:
        raise NotImplementedError


class _SlowLLM(LLMClient):
    """await 任意长时间再返回，用来触发 max_wall_time 超时."""

    def __init__(self, delay: float) -> None:
        super().__init__(model="slow-mock")
        self.delay = delay

    async def chat(
        self, messages: list[Message], tools: list[ToolParam] | None = None,
    ) -> LLMResponse:
        await asyncio.sleep(self.delay)
        return LLMResponse(content="done", usage=TokenUsage(10, 10))

    def chat_stream(self, messages, tools=None):
        raise NotImplementedError


class _RaisingLLM(LLMClient):
    """chat() 直接抛异常，用来触发 agent_error 归类."""

    def __init__(self) -> None:
        super().__init__(model="raise-mock")

    async def chat(self, messages, tools=None):
        raise RuntimeError("boom")

    def chat_stream(self, messages, tools=None):
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Agent factories 供 EvalRunner 使用
# ---------------------------------------------------------------------------


_SOLUTION_UTILS = """\
from datetime import datetime, timezone

def format_ts_as_date(ts):
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")
"""

_SOLUTION_TESTS = """\
from utils import format_ts_as_date


def test_epoch():
    assert format_ts_as_date(0) == "1970-01-01"


def test_later():
    assert format_ts_as_date(1_700_000_000).startswith("2023-")
"""


def _success_factory(workspace: Path, task: BenchmarkTask) -> Agent:  # noqa: ARG001
    _ = task  # factory 按签名接收 task，但本 mock 路径不需要用
    utils_path = str(workspace / "src" / "utils.py")
    tests_path = str(workspace / "tests" / "test_utils.py")
    responses = [
        LLMResponse(
            content="",
            tool_calls=[
                ToolCall(
                    id="c1",
                    name="WriteFile",
                    arguments={"path": utils_path, "content": _SOLUTION_UTILS},
                ),
            ],
            usage=TokenUsage(100, 50),
        ),
        LLMResponse(
            content="",
            tool_calls=[
                ToolCall(
                    id="c2",
                    name="WriteFile",
                    arguments={"path": tests_path, "content": _SOLUTION_TESTS},
                ),
            ],
            usage=TokenUsage(150, 40),
        ),
        LLMResponse(content="完成。", usage=TokenUsage(50, 10)),
    ]
    registry = ToolRegistry()
    registry.register(WriteFileTool())
    return Agent(_ScriptedLLM(responses), registry, "you are a test agent")


def _noop_factory(workspace: Path, task: BenchmarkTask) -> Agent:  # noqa: ARG001
    """Agent 直接给一句纯文本，什么都不改 → 应该走 validation_fail."""
    _ = task
    responses = [
        LLMResponse(content="我不打算改任何文件。", usage=TokenUsage(80, 20)),
    ]
    registry = ToolRegistry()
    return Agent(_ScriptedLLM(responses), registry, "noop")


def _timeout_factory(workspace: Path, task: BenchmarkTask) -> Agent:  # noqa: ARG001
    _ = task
    registry = ToolRegistry()
    return Agent(
        _SlowLLM(delay=2.0),
        registry,
        "slow",
        max_wall_time_seconds=0.2,
    )


def _raising_factory(workspace: Path, task: BenchmarkTask) -> Agent:  # noqa: ARG001
    _ = task
    registry = ToolRegistry()
    return Agent(_RaisingLLM(), registry, "raise")


def _relative_path_factory(workspace: Path, task: BenchmarkTask) -> Agent:  # noqa: ARG001
    """模拟真实 LLM：用**相对路径**调 WriteFile，配真实 FileGuard.

    这是 PR4c-fix2 引入的回归场景：
    没有 runner 的 _chdir 包裹时，FileGuard.is_path_allowed 会把 "hello.txt"
    resolve 成 <进程cwd>/hello.txt，判定不在 workspace 内、直接拦截。
    现在 runner 在 agent.run() 外面 chdir 到 workspace，这种相对路径应能正常落地。
    """
    _ = task
    responses = [
        LLMResponse(
            content="",
            tool_calls=[
                ToolCall(
                    id="c1",
                    name="WriteFile",
                    arguments={"path": "hello.txt", "content": "hi"},
                ),
            ],
            usage=TokenUsage(10, 5),
        ),
        LLMResponse(content="完成", usage=TokenUsage(5, 2)),
    ]
    registry = ToolRegistry()
    registry.register(WriteFileTool())
    return Agent(
        _ScriptedLLM(responses),
        registry,
        "you are a test agent; use relative paths.",
        file_guard=FileGuard(work_dir=workspace),
    )


# ---------------------------------------------------------------------------
# ModelPricing / KNOWN_MODELS
# ---------------------------------------------------------------------------


class TestModelPricing:
    def test_cost_math(self) -> None:
        p = ModelPricing(input_per_1k=0.001, output_per_1k=0.002)
        # 1000 tokens × 0.001 / 1000 + 2000 × 0.002 / 1000 = 0.001 + 0.004 = 0.005
        assert p.cost_usd(1000, 2000) == pytest.approx(0.005)

    def test_zero_tokens(self) -> None:
        p = ModelPricing(0.001, 0.002)
        assert p.cost_usd(0, 0) == 0.0

    def test_known_models_covers_deepseek(self) -> None:
        assert "deepseek-chat" in KNOWN_MODELS
        assert KNOWN_MODELS["deepseek-chat"].input_per_1k == pytest.approx(0.00014)

    def test_pricing_for_known(self) -> None:
        assert pricing_for("gpt-4o").output_per_1k == pytest.approx(0.010)

    def test_pricing_for_unknown_raises(self) -> None:
        with pytest.raises(KeyError, match="未知模型"):
            pricing_for("no-such-model")


# ---------------------------------------------------------------------------
# compute_edit_metrics
# ---------------------------------------------------------------------------


class TestEditMetrics:
    def test_all_match(self) -> None:
        p, r = compute_edit_metrics(["a", "b"], ["a", "b"])
        assert p == 1.0 and r == 1.0

    def test_extra_file_hurts_precision(self) -> None:
        # actual 多改了一个 —— precision 降，recall 不变
        p, r = compute_edit_metrics(["a", "b", "c"], ["a", "b"])
        assert p == pytest.approx(2 / 3)
        assert r == 1.0

    def test_missing_file_hurts_recall(self) -> None:
        p, r = compute_edit_metrics(["a"], ["a", "b"])
        assert p == 1.0
        assert r == 0.5

    def test_empty_actual(self) -> None:
        p, r = compute_edit_metrics([], ["a"])
        assert p == 0.0
        assert r == 0.0

    def test_empty_expected_recall_is_one(self) -> None:
        # "任务没指定预期改动" → recall 记 1 而不是除 0
        p, r = compute_edit_metrics(["x"], [])
        assert p == 0.0
        assert r == 1.0

    def test_dedup_via_set(self) -> None:
        p, r = compute_edit_metrics(["a", "a", "b"], ["a", "b"])
        assert p == 1.0 and r == 1.0


# ---------------------------------------------------------------------------
# classify_failure
# ---------------------------------------------------------------------------


class TestClassifyFailure:
    def test_agent_exception_wins(self) -> None:
        # agent_error 优先级最高：即使 validate 意外过也要标 agent_error
        assert classify_failure(
            stop_reason="ok", validation_passed=True, had_agent_exception=True
        ) == "agent_error"

    def test_timeout(self) -> None:
        assert classify_failure(
            stop_reason="timeout", validation_passed=False, had_agent_exception=False
        ) == "timeout"

    def test_max_tokens(self) -> None:
        assert classify_failure(
            stop_reason="max_tokens", validation_passed=False, had_agent_exception=False
        ) == "max_tokens"

    def test_max_rounds(self) -> None:
        assert classify_failure(
            stop_reason="max_rounds", validation_passed=False, had_agent_exception=False
        ) == "max_rounds"

    def test_validation_fail(self) -> None:
        assert classify_failure(
            stop_reason="ok", validation_passed=False, had_agent_exception=False
        ) == "validation_fail"

    def test_success_returns_none(self) -> None:
        assert classify_failure(
            stop_reason="ok", validation_passed=True, had_agent_exception=False
        ) is None


# ---------------------------------------------------------------------------
# run_validate_script
# ---------------------------------------------------------------------------


def _write_validate_script(path: Path, body: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)
    return path


class TestRunValidateScript:
    def test_passes(self, tmp_path: Path) -> None:
        script = _write_validate_script(
            tmp_path / "v.py",
            "import json; print('ok'); print(json.dumps({'passed': True}))\n",
        )
        ws = tmp_path / "ws"
        ws.mkdir()
        passed, details = run_validate_script(script, ws)
        assert passed is True
        assert "exit=0" in details

    def test_fails_json(self, tmp_path: Path) -> None:
        script = _write_validate_script(
            tmp_path / "v.py",
            "import json; print(json.dumps({'passed': False, 'details': 'nope'}))\n",
        )
        passed, details = run_validate_script(script, tmp_path)
        assert passed is False
        assert "nope" in details

    def test_last_line_not_json(self, tmp_path: Path) -> None:
        script = _write_validate_script(
            tmp_path / "v.py",
            "print('just a message')\n",
        )
        passed, details = run_validate_script(script, tmp_path)
        assert passed is False
        assert "非合法 JSON" in details

    def test_missing_passed_key(self, tmp_path: Path) -> None:
        script = _write_validate_script(
            tmp_path / "v.py",
            "import json; print(json.dumps({'details': 'oops'}))\n",
        )
        passed, details = run_validate_script(script, tmp_path)
        assert passed is False
        assert "缺 passed" in details

    def test_empty_stdout(self, tmp_path: Path) -> None:
        script = _write_validate_script(
            tmp_path / "v.py",
            "import sys; sys.stderr.write('err\\n')\n",
        )
        passed, details = run_validate_script(script, tmp_path)
        assert passed is False
        assert "stdout 为空" in details

    def test_timeout(self, tmp_path: Path) -> None:
        script = _write_validate_script(
            tmp_path / "v.py",
            "import time; time.sleep(5)\n",
        )
        passed, details = run_validate_script(script, tmp_path, timeout=1)
        assert passed is False
        assert "超时" in details

    def test_passes_ignoring_noise_before_last_line(self, tmp_path: Path) -> None:
        script = _write_validate_script(
            tmp_path / "v.py",
            "import json\n"
            "print('noise line 1')\n"
            "print('--- debug ---')\n"
            "print(json.dumps({'passed': True, 'detail': 'ok'}))\n",
        )
        passed, details = run_validate_script(script, tmp_path)
        assert passed is True


# ---------------------------------------------------------------------------
# compute_summary
# ---------------------------------------------------------------------------


def _make_result(
    task_id: str,
    run_index: int,
    passed: bool,
    *,
    task_hash: str = "h",
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


def _fake_suite(tasks: list[tuple[str, int]]) -> BenchmarkSuite:
    """构造只包含 id 和 level 字段有用的假 suite（summary 只读这两个字段）."""
    fake_tasks = [
        BenchmarkTask(
            id=tid,
            level=lvl,
            description="",
            workspace_dir=Path("/tmp"),
            validate_script=Path("/tmp/v"),
            expected_files=(),
            max_steps=1,
            max_tokens=1,
            max_wall_time_seconds=1,
            tags=(),
            task_hash="h",
        )
        for tid, lvl in tasks
    ]
    return BenchmarkSuite(tasks=fake_tasks, suite_hash="s")


class TestComputeSummary:
    def test_empty(self) -> None:
        suite = _fake_suite([])
        s = compute_summary([], suite)
        assert s.task_success_rate == 0.0
        assert s.by_level == {}
        assert s.by_failure_category == {}

    def test_all_pass_single_run(self) -> None:
        suite = _fake_suite([("A", 1), ("B", 2)])
        results = [
            _make_result("A", 0, True),
            _make_result("B", 0, True),
        ]
        s = compute_summary(results, suite)
        assert s.task_success_rate == 1.0
        assert s.by_level == {1: 1.0, 2: 1.0}
        assert s.by_failure_category == {}
        assert s.total_cost_usd == pytest.approx(0.02)

    def test_per_task_average_not_per_run(self) -> None:
        """2 任务 × 2 run，任务 A 全过、任务 B 全挂 → 任务成功率 = (1 + 0)/2 = 0.5."""
        suite = _fake_suite([("A", 1), ("B", 1)])
        results = [
            _make_result("A", 0, True),
            _make_result("A", 1, True),
            _make_result("B", 0, False, failure_category="validation_fail"),
            _make_result("B", 1, False, failure_category="validation_fail"),
        ]
        s = compute_summary(results, suite)
        assert s.task_success_rate == 0.5
        assert s.by_level == {1: 0.5}
        assert s.by_failure_category == {"validation_fail": 2}

    def test_partial_task_pass_rate(self) -> None:
        """任务 A 跑 3 次：2 过 1 挂 → 任务 A 通过率 2/3 → 任务成功率 2/3."""
        suite = _fake_suite([("A", 1)])
        results = [
            _make_result("A", 0, True),
            _make_result("A", 1, True),
            _make_result("A", 2, False, failure_category="max_rounds"),
        ]
        s = compute_summary(results, suite)
        assert s.task_success_rate == pytest.approx(2 / 3)
        assert s.by_level == {1: pytest.approx(2 / 3)}
        assert s.by_failure_category == {"max_rounds": 1}

    def test_tool_error_rate_pooled(self) -> None:
        suite = _fake_suite([("A", 1)])
        results = [
            _make_result("A", 0, True, step_count=10, tool_error_count=1),
            _make_result("A", 1, True, step_count=10, tool_error_count=3),
        ]
        s = compute_summary(results, suite)
        # 4 / 20 = 0.2
        assert s.tool_error_rate == pytest.approx(0.2)

    def test_tool_error_rate_zero_calls(self) -> None:
        suite = _fake_suite([("A", 1)])
        results = [_make_result("A", 0, True, step_count=0, tool_error_count=0)]
        s = compute_summary(results, suite)
        assert s.tool_error_rate == 0.0

    def test_verifier_recovery_rate(self) -> None:
        """4 次触发 verifier：2 首次过，2 首次挂；2 挂的里 1 挽回 → recovery=1/2=0.5."""
        suite = _fake_suite([("A", 1)])
        results = [
            _make_result("A", 0, True, verifier_first_passed=True, verifier_final_passed=True),
            _make_result("A", 1, True, verifier_first_passed=True, verifier_final_passed=True),
            _make_result("A", 2, True, verifier_first_passed=False, verifier_final_passed=True),  # recovered
            _make_result("A", 3, False, verifier_first_passed=False, verifier_final_passed=False,
                         failure_category="validation_fail"),
        ]
        s = compute_summary(results, suite)
        # first pass 2/4 = 0.5
        assert s.verifier_first_pass_rate == 0.5
        # recovery 1/2 = 0.5
        assert s.verifier_recovery_rate == 0.5

    def test_verifier_rates_when_never_triggered(self) -> None:
        suite = _fake_suite([("A", 1)])
        results = [_make_result("A", 0, True)]  # verifier_first_passed=None
        s = compute_summary(results, suite)
        assert s.verifier_first_pass_rate == 0.0
        assert s.verifier_recovery_rate == 0.0

    def test_averages(self) -> None:
        suite = _fake_suite([("A", 1), ("B", 2)])
        results = [
            _make_result("A", 0, True,
                         step_count=4, prompt_tokens=100, completion_tokens=30,
                         cost_usd=0.01, wall_time_seconds=2.0,
                         edit_precision=1.0, edit_recall=1.0),
            _make_result("B", 0, True,
                         step_count=6, prompt_tokens=200, completion_tokens=70,
                         cost_usd=0.03, wall_time_seconds=4.0,
                         edit_precision=0.5, edit_recall=0.8),
        ]
        s = compute_summary(results, suite)
        assert s.avg_step_count == 5.0
        assert s.avg_prompt_tokens == 150.0
        assert s.avg_completion_tokens == 50.0
        assert s.total_cost_usd == pytest.approx(0.04)
        assert s.avg_wall_time_seconds == 3.0
        assert s.avg_edit_precision == pytest.approx(0.75)
        assert s.avg_edit_recall == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# EvalRunner —— 端到端（跑真 L1-add-function 任务）
# ---------------------------------------------------------------------------


_L1_TASK_DIR = TASKS_DIR / "L1-add-function"


@pytest.fixture
def l1_task() -> BenchmarkTask:
    return BenchmarkTask.load(_L1_TASK_DIR)


class TestChdirContext:
    def test_chdir_restores_on_success(self, tmp_path: Path) -> None:
        prev = os.getcwd()
        target = tmp_path / "target"
        target.mkdir()
        with _chdir(target):
            assert Path(os.getcwd()).resolve() == target.resolve()
        assert os.getcwd() == prev

    def test_chdir_restores_on_exception(self, tmp_path: Path) -> None:
        prev = os.getcwd()
        target = tmp_path / "target"
        target.mkdir()
        with pytest.raises(RuntimeError, match="boom"):
            with _chdir(target):
                assert Path(os.getcwd()).resolve() == target.resolve()
                raise RuntimeError("boom")
        assert os.getcwd() == prev


class TestEvalRunnerE2E:
    async def test_success_run(self, tmp_path: Path, l1_task: BenchmarkTask) -> None:
        runner = EvalRunner(
            agent_factory=_success_factory,
            model_name="mock",
            pricing=ModelPricing(input_per_1k=0.001, output_per_1k=0.002),
            runs_per_task=1,
            workspace_root=tmp_path,
        )
        results = await runner.run_task(l1_task)
        assert len(results) == 1
        r = results[0]

        assert r.task_id == "L1-add-function"
        assert r.task_hash == l1_task.task_hash
        assert r.run_index == 0
        assert r.passed is True, f"validation should pass; details={r.validation_details}"
        assert r.failure_category is None
        assert r.stop_reason == "ok"
        assert r.step_count == 2  # 两次 WriteFile
        assert r.tool_error_count == 0
        assert r.prompt_tokens == 100 + 150 + 50
        assert r.completion_tokens == 50 + 40 + 10
        assert r.cost_usd == pytest.approx(
            (r.prompt_tokens * 0.001 + r.completion_tokens * 0.002) / 1000
        )

        # edit metrics：正好改了 expected_files 里的两个
        assert set(r.files_changed_actual) == {"src/utils.py", "tests/test_utils.py"}
        assert r.edit_precision == 1.0
        assert r.edit_recall == 1.0

        # verifier 没触发
        assert r.verifier_first_passed is None
        assert r.verifier_final_passed is None

        # 工作区保留在 tmp_path 下，路径记在 result 里方便调试
        assert r.workspace_path is not None
        assert Path(r.workspace_path).is_dir()
        assert str(tmp_path) in r.workspace_path

    async def test_relative_paths_resolved_against_workspace(
        self, tmp_path: Path, l1_task: BenchmarkTask, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """回归：Agent 用相对路径 WriteFile("hello.txt") 时，文件必须落在 workspace.

        修复前：FileGuard 用进程 cwd 解析相对路径，判定路径"不在 workspace"、
        直接拦截所有写入。我们在 runner._run_task_once 里 chdir 到 workspace
        绕开这个问题。
        """
        # 模拟用户在仓库根目录启动 main.py 的场景：进程 cwd = 别处
        outside = tmp_path / "outside"
        outside.mkdir()
        monkeypatch.chdir(outside)

        runner = EvalRunner(
            agent_factory=_relative_path_factory,
            model_name="mock",
            pricing=ModelPricing(0.0, 0.0),
            runs_per_task=1,
            workspace_root=tmp_path / "workspaces",
        )
        [r] = await runner.run_task(l1_task)

        # 不应被 FileGuard 拦截
        assert r.tool_error_count == 0, (
            f"WriteFile 相对路径被 FileGuard 误拦：{r.validation_details}"
        )
        # 文件落在 workspace，不在进程 cwd
        ws = Path(r.workspace_path)
        assert (ws / "hello.txt").is_file(), "相对路径写的文件应在 workspace 里"
        assert not (outside / "hello.txt").exists(), "文件不应跑到进程 cwd 去"

        # chdir 离开后应被复原
        assert Path.cwd() == outside

    async def test_validation_fail(self, tmp_path: Path, l1_task: BenchmarkTask) -> None:
        runner = EvalRunner(
            agent_factory=_noop_factory,
            model_name="mock",
            pricing=ModelPricing(0.001, 0.001),
            runs_per_task=1,
            workspace_root=tmp_path,
        )
        [r] = await runner.run_task(l1_task)
        assert r.passed is False
        assert r.failure_category == "validation_fail"
        assert r.stop_reason == "ok"
        # 什么都没改
        assert r.files_changed_actual == []
        assert r.edit_precision == 0.0
        assert r.edit_recall == 0.0

    async def test_agent_error(self, tmp_path: Path, l1_task: BenchmarkTask) -> None:
        runner = EvalRunner(
            agent_factory=_raising_factory,
            model_name="mock",
            pricing=ModelPricing(0.0, 0.0),
            runs_per_task=1,
            workspace_root=tmp_path,
        )
        [r] = await runner.run_task(l1_task)
        assert r.passed is False
        assert r.failure_category == "agent_error"
        assert "RuntimeError" in r.validation_details
        assert "boom" in r.validation_details
        # stop_reason 兜底为 "error"
        assert r.stop_reason == "error"

    async def test_timeout(self, tmp_path: Path, l1_task: BenchmarkTask) -> None:
        runner = EvalRunner(
            agent_factory=_timeout_factory,
            model_name="mock",
            pricing=ModelPricing(0.0, 0.0),
            runs_per_task=1,
            workspace_root=tmp_path,
        )
        [r] = await runner.run_task(l1_task)
        assert r.passed is False
        assert r.failure_category == "timeout"
        assert r.stop_reason == "timeout"
        assert r.wall_time_seconds < 2.0  # 远低于 _SlowLLM 的 2s delay

    async def test_runs_per_task_3(
        self, tmp_path: Path, l1_task: BenchmarkTask
    ) -> None:
        runner = EvalRunner(
            agent_factory=_noop_factory,  # 不依赖有状态的 factory
            model_name="mock",
            pricing=ModelPricing(0.0, 0.0),
            runs_per_task=3,
            workspace_root=tmp_path,
        )
        results = await runner.run_task(l1_task)
        assert len(results) == 3
        assert [r.run_index for r in results] == [0, 1, 2]
        # 三个 workspace 各自独立
        paths = {r.workspace_path for r in results}
        assert len(paths) == 3

    async def test_factory_receives_task(
        self, tmp_path: Path, l1_task: BenchmarkTask,
    ) -> None:
        """factory 必须拿到 (workspace, task)，不然没法按 task 限制 LoopGuard/wall_time."""
        seen: list[tuple[Path, BenchmarkTask]] = []

        def factory(workspace: Path, task: BenchmarkTask) -> Agent:
            seen.append((workspace, task))
            registry = ToolRegistry()
            return Agent(
                _ScriptedLLM([LLMResponse(content="hi", usage=TokenUsage(10, 5))]),
                registry,
                "spy",
            )

        runner = EvalRunner(
            agent_factory=factory,
            model_name="mock",
            pricing=ModelPricing(0.0, 0.0),
            runs_per_task=1,
            workspace_root=tmp_path,
        )
        await runner.run_task(l1_task)

        assert len(seen) == 1
        ws, t = seen[0]
        assert ws.is_dir()
        assert t.id == "L1-add-function"
        assert t.task_hash == l1_task.task_hash
        assert t.max_steps == l1_task.max_steps

    async def test_run_suite_aggregates(
        self, tmp_path: Path, l1_task: BenchmarkTask
    ) -> None:
        suite = BenchmarkSuite(tasks=[l1_task], suite_hash="abc")
        runner = EvalRunner(
            agent_factory=_success_factory,
            model_name="mock",
            pricing=ModelPricing(0.001, 0.002),
            runs_per_task=2,
            workspace_root=tmp_path,
        )
        suite_result = await runner.run_suite(suite)

        assert suite_result.suite_hash == "abc"
        assert suite_result.model_name == "mock"
        assert len(suite_result.results) == 2
        assert all(r.passed for r in suite_result.results)
        assert suite_result.summary.task_success_rate == 1.0
        assert suite_result.summary.by_level == {1: 1.0}
        assert suite_result.summary.by_failure_category == {}
        # timestamp 是 ISO 格式
        assert "T" in suite_result.timestamp
