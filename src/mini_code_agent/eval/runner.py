"""Eval runner — 在临时工作区里跑单个 BenchmarkTask 并量化结果.

详见 DESIGN.md §2 / §4 / §9.2。本模块只管：
- 复制 workspace 到临时目录
- 交给 agent_factory 构建 Agent 跑 task.description
- 跑前 snapshot / 跑完 diff 得 files_changed_actual
- 跑 validate.py，解析最后一行 JSON
- 把所有结果组装成 TaskResult / SuiteResult / EvalSummary
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Literal

from ..core.agent import Agent, AgentResult
from . import snapshot
from .benchmark import BenchmarkSuite, BenchmarkTask

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 定价表
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelPricing:
    """模型定价：USD / 1K tokens."""

    input_per_1k: float
    output_per_1k: float

    def cost_usd(self, prompt_tokens: int, completion_tokens: int) -> float:
        return (
            prompt_tokens * self.input_per_1k
            + completion_tokens * self.output_per_1k
        ) / 1000.0


# 预置定价（DESIGN.md §4.3）。实际跑 eval 时用户可传自定义覆盖。
KNOWN_MODELS: dict[str, ModelPricing] = {
    "deepseek-chat":     ModelPricing(0.00014, 0.00028),
    "gpt-4o":            ModelPricing(0.0025, 0.010),
    "gpt-4o-mini":       ModelPricing(0.00015, 0.00060),
    "claude-sonnet-4-5": ModelPricing(0.003, 0.015),
}


def pricing_for(model_name: str) -> ModelPricing:
    """按模型名查预置定价；找不到时抛 KeyError."""
    try:
        return KNOWN_MODELS[model_name]
    except KeyError as e:
        raise KeyError(
            f"未知模型 {model_name!r}；在 KNOWN_MODELS 里添加或显式传 pricing=..."
        ) from e


# ---------------------------------------------------------------------------
# failure_category（DESIGN.md §2.3）
# ---------------------------------------------------------------------------


FailureCategory = Literal[
    "timeout", "max_tokens", "max_rounds",
    "validation_fail", "agent_error",
]


def classify_failure(
    *,
    stop_reason: str,
    validation_passed: bool,
    had_agent_exception: bool,
) -> str | None:
    """返回 failure_category。None 代表任务通过，无失败归类."""
    if had_agent_exception:
        return "agent_error"
    if stop_reason == "timeout":
        return "timeout"
    if stop_reason == "max_tokens":
        return "max_tokens"
    if stop_reason == "max_rounds":
        return "max_rounds"
    if not validation_passed:
        return "validation_fail"
    return None


# ---------------------------------------------------------------------------
# edit_precision / edit_recall（DESIGN.md §2）
# ---------------------------------------------------------------------------


def compute_edit_metrics(
    actual: list[str],
    expected: list[str],
) -> tuple[float, float]:
    """返回 (precision, recall).

    约定：
    - precision = |actual ∩ expected| / |actual|，actual 为空时记为 0
    - recall    = |actual ∩ expected| / |expected|，expected 为空时记为 1
      （"任务没指定预期改动"天然算召回满分，避免除 0）
    """
    actual_set = set(actual)
    expected_set = set(expected)
    inter = actual_set & expected_set

    precision = len(inter) / len(actual_set) if actual_set else 0.0
    recall = len(inter) / len(expected_set) if expected_set else 1.0
    return precision, recall


# ---------------------------------------------------------------------------
# validate.py 子进程
# ---------------------------------------------------------------------------


VALIDATE_TIMEOUT_S = 60


def run_validate_script(
    validate_script: Path,
    workspace: Path,
    *,
    timeout: int = VALIDATE_TIMEOUT_S,
) -> tuple[bool, str]:
    """以子进程方式跑 validate.py，返回 (passed, details).

    解析 DESIGN.md §3 协议：stdout 最后一行必须是合法 JSON 且含 "passed": bool。
    解析失败 → passed=False，由 classify_failure 归类 validation_fail。
    """
    try:
        proc = subprocess.run(
            [sys.executable, str(validate_script)],
            cwd=str(workspace),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as e:
        stdout = e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
        stderr = e.stderr.decode() if isinstance(e.stderr, bytes) else (e.stderr or "")
        return False, (
            f"[validate.py 超时 {timeout}s]\n"
            f"--- stdout ---\n{stdout}\n"
            f"--- stderr ---\n{stderr}"
        )

    details = (
        f"--- validate.py stdout ---\n{proc.stdout}\n"
        f"--- validate.py stderr ---\n{proc.stderr}\n"
        f"--- exit={proc.returncode} ---"
    )

    lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    if not lines:
        return False, f"[validate.py stdout 为空]\n{details}"

    try:
        payload = json.loads(lines[-1])
    except json.JSONDecodeError as e:
        return False, f"[validate.py 末行非合法 JSON: {e}]\n{details}"

    if not isinstance(payload, dict) or "passed" not in payload:
        return False, f"[validate.py JSON 缺 passed 字段]\n{details}"

    return bool(payload["passed"]), details


# ---------------------------------------------------------------------------
# git HEAD commit
# ---------------------------------------------------------------------------


def _current_git_commit(cwd: Path | None = None) -> str | None:
    """尝试读当前仓库 HEAD；不在 git 仓库里时返回 None."""
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    commit = proc.stdout.strip()
    return commit or None


# ---------------------------------------------------------------------------
# Result dataclasses（DESIGN.md §2）
# ---------------------------------------------------------------------------


@dataclass
class TaskResult:
    """单次 run 的结果。同一任务 n=3 次 run 会有 3 个 TaskResult."""

    task_id: str
    task_hash: str
    run_index: int
    passed: bool
    stop_reason: str
    step_count: int
    tool_error_count: int
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    wall_time_seconds: float
    verifier_first_passed: bool | None
    verifier_final_passed: bool | None
    files_changed_actual: list[str]
    edit_precision: float
    edit_recall: float
    failure_category: str | None
    validation_details: str
    workspace_path: str | None = None  # 保留给调试：对应的临时目录


@dataclass
class EvalSummary:
    """对一次 suite run 的所有 TaskResult 做聚合指标."""

    task_success_rate: float
    by_level: dict[int, float]
    by_failure_category: dict[str, int]
    avg_step_count: float
    tool_error_rate: float
    verifier_first_pass_rate: float
    verifier_recovery_rate: float
    avg_prompt_tokens: float
    avg_completion_tokens: float
    total_cost_usd: float
    avg_wall_time_seconds: float
    avg_edit_precision: float
    avg_edit_recall: float


@dataclass
class SuiteResult:
    """一次完整 eval 运行的顶层结果。可 JSON 序列化后落盘."""

    timestamp: str
    git_commit: str | None
    suite_hash: str
    model_name: str
    results: list[TaskResult]
    summary: EvalSummary


# ---------------------------------------------------------------------------
# 聚合逻辑
# ---------------------------------------------------------------------------


def compute_summary(
    results: list[TaskResult],
    suite: BenchmarkSuite,
) -> EvalSummary:
    """把 n*m 条 TaskResult 聚合成 EvalSummary.

    任务成功率的算法（DESIGN.md §2 EvalSummary 备注）：
        1) 先按 task_id 分组，算"这个任务 n 次里通过几次 / n" 得到任务通过率
        2) 再对所有任务的通过率做简单平均

    这样单个任务的 n=3 噪音不会因为任务数不同被稀释。
    """
    if not results:
        return EvalSummary(
            task_success_rate=0.0,
            by_level={},
            by_failure_category={},
            avg_step_count=0.0,
            tool_error_rate=0.0,
            verifier_first_pass_rate=0.0,
            verifier_recovery_rate=0.0,
            avg_prompt_tokens=0.0,
            avg_completion_tokens=0.0,
            total_cost_usd=0.0,
            avg_wall_time_seconds=0.0,
            avg_edit_precision=0.0,
            avg_edit_recall=0.0,
        )

    # task_id → list[TaskResult]
    by_task: dict[str, list[TaskResult]] = {}
    for r in results:
        by_task.setdefault(r.task_id, []).append(r)
    task_pass_rates: dict[str, float] = {
        tid: sum(1 for r in rs if r.passed) / len(rs)
        for tid, rs in by_task.items()
    }
    task_success_rate = sum(task_pass_rates.values()) / len(task_pass_rates)

    # by_level
    task_level = {t.id: t.level for t in suite.tasks}
    level_to_rates: dict[int, list[float]] = {}
    for tid, rate in task_pass_rates.items():
        lvl = task_level.get(tid)
        if lvl is None:
            continue
        level_to_rates.setdefault(lvl, []).append(rate)
    by_level = {
        lvl: sum(rs) / len(rs) for lvl, rs in level_to_rates.items()
    }

    # by_failure_category —— 每次 run 的分类都计数（通过的为 None 不计）
    by_failure_category: dict[str, int] = {}
    for r in results:
        if r.failure_category is None:
            continue
        by_failure_category[r.failure_category] = (
            by_failure_category.get(r.failure_category, 0) + 1
        )

    # tool 相关
    total_tool_calls = sum(r.step_count for r in results)
    total_tool_errors = sum(r.tool_error_count for r in results)
    tool_error_rate = (
        total_tool_errors / total_tool_calls if total_tool_calls else 0.0
    )

    # verifier 指标（只在触发过 verifier 的 run 里算）
    verified = [r for r in results if r.verifier_first_passed is not None]
    if verified:
        verifier_first_pass_rate = (
            sum(1 for r in verified if r.verifier_first_passed) / len(verified)
        )
        failed_first = [r for r in verified if r.verifier_first_passed is False]
        recovered = [r for r in failed_first if r.verifier_final_passed is True]
        verifier_recovery_rate = (
            len(recovered) / len(failed_first) if failed_first else 0.0
        )
    else:
        verifier_first_pass_rate = 0.0
        verifier_recovery_rate = 0.0

    n = len(results)
    return EvalSummary(
        task_success_rate=task_success_rate,
        by_level=by_level,
        by_failure_category=by_failure_category,
        avg_step_count=sum(r.step_count for r in results) / n,
        tool_error_rate=tool_error_rate,
        verifier_first_pass_rate=verifier_first_pass_rate,
        verifier_recovery_rate=verifier_recovery_rate,
        avg_prompt_tokens=sum(r.prompt_tokens for r in results) / n,
        avg_completion_tokens=sum(r.completion_tokens for r in results) / n,
        total_cost_usd=sum(r.cost_usd for r in results),
        avg_wall_time_seconds=sum(r.wall_time_seconds for r in results) / n,
        avg_edit_precision=sum(r.edit_precision for r in results) / n,
        avg_edit_recall=sum(r.edit_recall for r in results) / n,
    )


# ---------------------------------------------------------------------------
# EvalRunner
# ---------------------------------------------------------------------------


AgentFactory = Callable[[Path], Agent]


class EvalRunner:
    """把 BenchmarkTask / BenchmarkSuite 跑起来，吐出 TaskResult / SuiteResult.

    由外部传入 agent_factory(workspace) → Agent：factory 负责每次 run 构造一个
    干净的 Agent（BashTool(cwd=workspace)、FileGuard(work_dir=workspace)、
    LoopGuard(max_rounds=task.max_steps, max_tokens=task.max_tokens)、
    Agent(max_wall_time_seconds=task.max_wall_time_seconds)）。
    Runner 不替用户做这件事，因为每个项目的安全控制策略不一样。
    """

    def __init__(
        self,
        *,
        agent_factory: AgentFactory,
        model_name: str,
        pricing: ModelPricing,
        runs_per_task: int = 3,
        parallel_tasks: int = 1,
        workspace_root: Path | None = None,
    ) -> None:
        self.agent_factory = agent_factory
        self.model_name = model_name
        self.pricing = pricing
        self.runs_per_task = max(1, runs_per_task)
        self.parallel_tasks = max(1, parallel_tasks)
        self.workspace_root = (
            Path(workspace_root) if workspace_root is not None
            else Path(tempfile.gettempdir())
        )

    async def run_task(self, task: BenchmarkTask) -> list[TaskResult]:
        """串行跑 n 次 task（同一任务的多次 run 必须串行，避免 tmp workspace 重名争抢）."""
        results: list[TaskResult] = []
        for i in range(self.runs_per_task):
            results.append(await self._run_task_once(task, i))
        return results

    async def run_suite(self, suite: BenchmarkSuite) -> SuiteResult:
        """跑整个 suite，按 parallel_tasks 控制任务间并发."""
        sem = asyncio.Semaphore(self.parallel_tasks)

        async def _wrap(t: BenchmarkTask) -> list[TaskResult]:
            async with sem:
                return await self.run_task(t)

        task_results_nested = await asyncio.gather(
            *[_wrap(t) for t in suite.tasks]
        )
        all_results: list[TaskResult] = []
        for rs in task_results_nested:
            all_results.extend(rs)

        summary = compute_summary(all_results, suite)
        return SuiteResult(
            timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            git_commit=_current_git_commit(),
            suite_hash=suite.suite_hash,
            model_name=self.model_name,
            results=all_results,
            summary=summary,
        )

    async def _run_task_once(
        self, task: BenchmarkTask, run_index: int,
    ) -> TaskResult:
        # 1. 临时工作区
        ws_name = f"eval-{task.id}-{run_index}-{uuid.uuid4().hex[:8]}"
        workspace = self.workspace_root / ws_name
        if workspace.exists():
            shutil.rmtree(workspace)
        shutil.copytree(task.workspace_dir, workspace)

        # 2. 跑前快照
        before_snap = snapshot.capture(workspace)

        # 3. 构造 Agent 并跑
        agent = self.agent_factory(workspace)

        t0 = time.monotonic()
        agent_result: AgentResult | None = None
        agent_error: BaseException | None = None
        try:
            agent_result = await agent.run(task.description)
        except Exception as e:  # noqa: BLE001
            logger.exception("agent.run() 抛异常")
            agent_error = e
        wall_time = time.monotonic() - t0

        # 4. 跑后 diff（即便 agent 挂了，也要看看文件有没有动）
        diff = snapshot.diff(workspace, before_snap)
        files_changed = diff.changed
        precision, recall = compute_edit_metrics(
            files_changed, list(task.expected_files)
        )

        # 5. 跑 validate.py（agent 异常时跳过，直接失败）
        if agent_error is not None:
            validation_passed = False
            validation_details = (
                f"[agent exception] {type(agent_error).__name__}: {agent_error}"
            )
        else:
            validation_passed, validation_details = run_validate_script(
                task.validate_script, workspace
            )

        # 6. 归类 + 打分
        stop_reason = agent_result.stop_reason if agent_result else "error"
        category = classify_failure(
            stop_reason=stop_reason,
            validation_passed=validation_passed,
            had_agent_exception=agent_error is not None,
        )
        prompt_tokens = agent_result.usage.input_tokens if agent_result else 0
        completion_tokens = (
            agent_result.usage.output_tokens if agent_result else 0
        )
        cost = self.pricing.cost_usd(prompt_tokens, completion_tokens)

        return TaskResult(
            task_id=task.id,
            task_hash=task.task_hash,
            run_index=run_index,
            passed=validation_passed,
            stop_reason=stop_reason,
            step_count=agent_result.tool_calls_count if agent_result else 0,
            tool_error_count=agent_result.tool_calls_errors if agent_result else 0,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=cost,
            wall_time_seconds=wall_time,
            verifier_first_passed=(
                agent_result.verifier_first_passed if agent_result else None
            ),
            verifier_final_passed=(
                agent_result.verifier_final_passed if agent_result else None
            ),
            files_changed_actual=files_changed,
            edit_precision=precision,
            edit_recall=recall,
            failure_category=category,
            validation_details=validation_details,
            workspace_path=str(workspace),
        )
