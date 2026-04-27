"""GraphExecutor L1 集成测试 — SubtaskRunner + Ledger + Checkpoint + Verifier."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest

from mini_code_agent.artifacts import (
    ArtifactStore,
    Confidence,
    SubtaskArtifact,
)
from mini_code_agent.core import (
    AgentStuckError,
    GraphContext,
    GraphExecutor,
    SubtaskRunner,
    TaskGraph,
    TaskNode,
    TaskStatus,
)
from mini_code_agent.core.agent import AgentResult
from mini_code_agent.llm.base import TokenUsage
from mini_code_agent.longrun import (
    CheckpointManager,
    LongRunConfig,
    TaskLedgerManager,
)
from mini_code_agent.longrun.ledger_types import TaskRunStatus
from mini_code_agent.safety.git_checkpoint import GitCheckpoint


# ──────────────────────────────────────────────────────────────────
# 假 Agent / 假 Verifier — 隔离 LLM 与外部依赖
# ──────────────────────────────────────────────────────────────────


class FakeAgent:
    """最小的 Agent 替身：可控制 run 的行为；提供 observer / reset."""

    def __init__(
        self,
        run_results: list[Any] | None = None,
    ) -> None:
        # run_results: 每次 run 返回一个项；项可以是 AgentResult 或 Exception
        self._run_results = list(run_results or [])
        self._call_count = 0
        self.observers: list[Any] = []
        self._files_changed: list[str] = []
        self.messages: list[Any] = []

    def add_observer(self, obs: Any) -> None:
        self.observers.append(obs)

    def remove_observer(self, obs: Any | None = None) -> None:
        if obs is None:
            self.observers.clear()
        else:
            try:
                self.observers.remove(obs)
            except ValueError:
                pass

    def reset(self) -> None:
        self._files_changed = []

    async def run(self, prompt: str) -> AgentResult:
        if not self._run_results:
            return AgentResult(content="default ok", usage=TokenUsage())
        item = self._run_results.pop(0) if self._call_count < len(self._run_results) else self._run_results[-1]
        self._call_count += 1
        if isinstance(item, BaseException):
            raise item
        return item


class FakeVerifier:
    """假 IncrementalVerifier — 按需返回 pass / fail."""

    def __init__(self, passed: bool = True, level2_passed: bool = True) -> None:
        self.passed = passed
        self.level2_passed = level2_passed
        self.level2 = self  # 让 SubtaskRunner.verifier.level2.verify 拿到自身

    async def verify_after_edit(self, files_changed, project_path, task_id=""):
        from mini_code_agent.verify.types import IncrementalVerificationResult
        from mini_code_agent.artifacts.verification import CheckResult
        return IncrementalVerificationResult(
            task_id=task_id,
            level=1,
            checks=[
                CheckResult(
                    check_name="syntax",
                    passed=self.passed,
                    skipped=False,
                    skip_reason=None,
                    duration_seconds=0.01,
                    details="ok" if self.passed else "syntax error",
                    items_checked=len(files_changed),
                    items_failed=0 if self.passed else len(files_changed),
                ),
                CheckResult(
                    check_name="import",
                    passed=True, skipped=False, skip_reason=None,
                    duration_seconds=0.0, details="ok",
                    items_checked=0, items_failed=0,
                ),
            ],
            overall_passed=self.passed,
            files_verified=list(files_changed),
            total_duration_seconds=0.01,
        )

    async def verify(self, files_changed, project_path, task_id=""):
        # Level2 接口
        from mini_code_agent.verify.types import IncrementalVerificationResult
        from mini_code_agent.artifacts.verification import CheckResult
        return IncrementalVerificationResult(
            task_id=task_id,
            level=2,
            checks=[
                CheckResult(
                    check_name="unit_test",
                    passed=self.level2_passed,
                    skipped=False,
                    skip_reason=None,
                    duration_seconds=0.05,
                    details="ok" if self.level2_passed else "test failed",
                    items_checked=1,
                    items_failed=0 if self.level2_passed else 1,
                ),
            ],
            overall_passed=self.level2_passed,
            files_verified=list(files_changed),
            total_duration_seconds=0.05,
        )


# ──────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True,
    )


@pytest.fixture
def tmp_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "t@t")
    _git(repo, "config", "user.name", "t")
    (repo / "README.md").write_text("seed\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "seed")
    return repo


@pytest.fixture
def stores(tmp_repo: Path):
    artifact_store = ArtifactStore(storage_dir=str(tmp_repo / ".agent" / "artifacts"))
    ledger_manager = TaskLedgerManager(storage_dir=str(tmp_repo / ".agent" / "ledger"))
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=str(tmp_repo / ".agent" / "checkpoints"),
        ledger_manager=ledger_manager,
        git_checkpoint=GitCheckpoint(cwd=str(tmp_repo)),
        cwd=str(tmp_repo),
    )
    return artifact_store, ledger_manager, checkpoint_manager


def _make_graph_3() -> TaskGraph:
    g = TaskGraph()
    g.original_goal = "build a tiny module in 3 steps"
    g.add_task(TaskNode(id="t1", description="step1", files_involved=["a.py"]))
    g.add_task(TaskNode(id="t2", description="step2", files_involved=["b.py"], dependencies=["t1"]))
    g.add_task(TaskNode(id="t3", description="step3", files_involved=["c.py"], dependencies=["t2"]))
    return g


def _agent_writes(repo: Path, files: list[tuple[str, str]]):
    """构造一个 FakeAgent.run 的回调式行为：把 files 写到 repo，返回 ok 结果."""
    def _do() -> AgentResult:
        for path, content in files:
            full = repo / path
            full.parent.mkdir(parents=True, exist_ok=True)
            full.write_text(content, encoding="utf-8")
        return AgentResult(content=f"wrote {len(files)} files", usage=TokenUsage(input_tokens=10, output_tokens=5))
    return _do


class _ScriptedAgent(FakeAgent):
    """每次 run() 调用 callable 来动态生成结果（可写文件）."""

    def __init__(self, scripts: list[Any]) -> None:
        super().__init__()
        self._scripts = list(scripts)

    async def run(self, prompt: str) -> AgentResult:
        if not self._scripts:
            return AgentResult(content="ok", usage=TokenUsage())
        item = self._scripts.pop(0)
        if isinstance(item, BaseException):
            raise item
        result = item() if callable(item) else item
        # 把 token 用量通过 observer 通知出去，让 ArtifactBuilder 记录
        for obs in list(self.observers):
            try:
                obs.on_llm_call(
                    result.usage.input_tokens,
                    result.usage.output_tokens,
                    "fake-model",
                )
            except Exception:
                pass
        return result


# ──────────────────────────────────────────────────────────────────
# 测试
# ──────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_three_subtasks_each_produce_artifact(tmp_repo, stores):
    artifact_store, ledger_manager, checkpoint_manager = stores
    graph = _make_graph_3()

    agent = _ScriptedAgent([
        _agent_writes(tmp_repo, [("a.py", "print('a')\n")]),
        _agent_writes(tmp_repo, [("b.py", "print('b')\n")]),
        _agent_writes(tmp_repo, [("c.py", "print('c')\n")]),
    ])
    runner = SubtaskRunner(
        agent=agent,
        artifact_store=artifact_store,
        verifier=FakeVerifier(passed=True),
        git_checkpoint=GitCheckpoint(cwd=str(tmp_repo)),
    )
    config = LongRunConfig(token_budget=100_000)
    executor = GraphExecutor(
        project_path=str(tmp_repo),
        max_retries=2,
        subtask_runner=runner,
        ledger_manager=ledger_manager,
        checkpoint_manager=checkpoint_manager,
        longrun_config=config,
    )
    ledger = ledger_manager.create("goal", graph, budget=100_000)

    # 关键：每个子任务跑完后，必须 commit 工作区，
    # 否则下一个子任务的 git diff 会包含上一个子任务的所有文件。
    async def commit_and_run():
        result = await executor.execute_with_ledger(graph, ledger, str(tmp_repo))
        return result

    result = await commit_and_run()

    # 三个任务都完成
    assert result.tasks_completed == 3
    assert result.tasks_failed == 0
    # Ledger 有 3 条 CompletedTaskRecord（每个子任务即时写入）
    assert len(ledger.completed_tasks) == 3
    # 每个任务都对应一个 artifact
    for ct in ledger.completed_tasks:
        metas = artifact_store.list_for_task(ct.task_id)
        assert len(metas) >= 1


@pytest.mark.asyncio
async def test_stuck_then_retry_succeeds(tmp_repo, stores):
    artifact_store, ledger_manager, checkpoint_manager = stores

    graph = TaskGraph()
    graph.add_task(TaskNode(id="solo", description="single task", files_involved=["x.py"]))

    # 第一次抛 STUCK，第二次成功
    agent = _ScriptedAgent([
        AgentStuckError(reason="ran out of ideas", question="should i use stdlib?"),
        _agent_writes(tmp_repo, [("x.py", "x = 1\n")]),
    ])
    runner = SubtaskRunner(
        agent=agent,
        artifact_store=artifact_store,
        verifier=FakeVerifier(passed=True),
        git_checkpoint=GitCheckpoint(cwd=str(tmp_repo)),
    )
    executor = GraphExecutor(
        project_path=str(tmp_repo),
        max_retries=2,
        subtask_runner=runner,
        ledger_manager=ledger_manager,
        checkpoint_manager=checkpoint_manager,
        longrun_config=LongRunConfig(token_budget=10_000),
    )
    ledger = ledger_manager.create("g", graph, budget=10_000)
    result = await executor.execute_with_ledger(graph, ledger, str(tmp_repo))

    # 重试后成功
    assert result.tasks_completed == 1
    assert result.tasks_failed == 0
    # 应有一条失败尝试 + 一条完成记录
    assert len(ledger.failed_attempts) == 1
    assert len(ledger.completed_tasks) == 1


@pytest.mark.asyncio
async def test_stuck_exhausts_retries_marks_failed(tmp_repo, stores):
    artifact_store, ledger_manager, checkpoint_manager = stores
    graph = TaskGraph()
    graph.add_task(TaskNode(id="solo", description="t", files_involved=["x.py"]))

    # 永远 STUCK
    agent = _ScriptedAgent([AgentStuckError(reason="nope")] * 10)
    runner = SubtaskRunner(
        agent=agent,
        artifact_store=artifact_store,
        verifier=FakeVerifier(passed=True),
        git_checkpoint=GitCheckpoint(cwd=str(tmp_repo)),
    )
    executor = GraphExecutor(
        project_path=str(tmp_repo),
        max_retries=2,
        subtask_runner=runner,
        ledger_manager=ledger_manager,
        checkpoint_manager=checkpoint_manager,
        longrun_config=LongRunConfig(token_budget=10_000),
    )
    ledger = ledger_manager.create("g", graph, budget=10_000)
    result = await executor.execute_with_ledger(graph, ledger, str(tmp_repo))

    assert result.tasks_failed == 1
    assert graph.nodes["solo"].status == TaskStatus.FAILED
    # 失败尝试 = 初次 + 2 次重试 = 3 条
    assert len(ledger.failed_attempts) == 3


@pytest.mark.asyncio
async def test_verification_failed_records_uncertain(tmp_repo, stores):
    artifact_store, ledger_manager, checkpoint_manager = stores
    graph = TaskGraph()
    graph.add_task(TaskNode(id="solo", description="t", files_involved=["x.py"]))

    agent = _ScriptedAgent([
        _agent_writes(tmp_repo, [("x.py", "syntax error here\n")]),
    ])
    runner = SubtaskRunner(
        agent=agent,
        artifact_store=artifact_store,
        verifier=FakeVerifier(passed=False),  # 验证失败
        git_checkpoint=GitCheckpoint(cwd=str(tmp_repo)),
    )
    executor = GraphExecutor(
        project_path=str(tmp_repo),
        max_retries=2,
        subtask_runner=runner,
        ledger_manager=ledger_manager,
        checkpoint_manager=checkpoint_manager,
        longrun_config=LongRunConfig(token_budget=10_000),
    )
    ledger = ledger_manager.create("g", graph, budget=10_000)
    result = await executor.execute_with_ledger(graph, ledger, str(tmp_repo))

    assert result.tasks_completed == 1  # 当作完成（UNCERTAIN）
    # Ledger 应该有 active_issue
    assert len(ledger.active_issues) >= 1
    # CompletedTaskRecord.confidence 应该是 UNCERTAIN
    assert ledger.completed_tasks[0].confidence == Confidence.UNCERTAIN.value


@pytest.mark.asyncio
async def test_checkpoint_triggered_after_subtask(tmp_repo, stores):
    """子任务跑完且超过 token 阈值 → 触发自动 checkpoint."""
    artifact_store, ledger_manager, checkpoint_manager = stores

    graph = TaskGraph()
    graph.add_task(TaskNode(id="solo", description="t", files_involved=["x.py"]))

    def _write_with_huge_tokens():
        (tmp_repo / "x.py").write_text("x=1\n", encoding="utf-8")
        # 模拟一次大量 token 消耗
        return AgentResult(content="ok", usage=TokenUsage(input_tokens=60_000, output_tokens=0))

    agent = _ScriptedAgent([_write_with_huge_tokens])
    runner = SubtaskRunner(
        agent=agent,
        artifact_store=artifact_store,
        verifier=FakeVerifier(passed=True),
        git_checkpoint=GitCheckpoint(cwd=str(tmp_repo)),
    )
    config = LongRunConfig(
        token_budget=200_000,
        checkpoint_interval_tokens=10_000,  # 阈值小，方便触发
    )
    executor = GraphExecutor(
        project_path=str(tmp_repo),
        max_retries=2,
        subtask_runner=runner,
        ledger_manager=ledger_manager,
        checkpoint_manager=checkpoint_manager,
        longrun_config=config,
    )
    ledger = ledger_manager.create("g", graph, budget=200_000)
    await executor.execute_with_ledger(graph, ledger, str(tmp_repo))

    # 检查 checkpoint 是否被创建
    metas = checkpoint_manager.list_checkpoints(ledger.task_id)
    assert len(metas) >= 1


@pytest.mark.asyncio
async def test_scope_check_records_out_of_scope_paths(tmp_repo, stores):
    """Agent 越界写文件 — Artifact.scope_check 标记为不 clean，但 L1 不拦截."""
    artifact_store, ledger_manager, checkpoint_manager = stores

    graph = TaskGraph()
    graph.add_task(TaskNode(
        id="solo",
        description="modify a.py",
        files_involved=["a.py"],   # 只允许 a.py 及配套
    ))

    agent = _ScriptedAgent([
        _agent_writes(tmp_repo, [
            ("a.py", "x=1\n"),
            ("forbidden/elsewhere.py", "y=2\n"),  # 越界
        ]),
    ])
    runner = SubtaskRunner(
        agent=agent,
        artifact_store=artifact_store,
        verifier=FakeVerifier(passed=True),
        git_checkpoint=GitCheckpoint(cwd=str(tmp_repo)),
    )
    executor = GraphExecutor(
        project_path=str(tmp_repo),
        max_retries=2,
        subtask_runner=runner,
        ledger_manager=ledger_manager,
        checkpoint_manager=checkpoint_manager,
        longrun_config=LongRunConfig(token_budget=10_000),
    )
    ledger = ledger_manager.create("g", graph, budget=10_000)
    result = await executor.execute_with_ledger(graph, ledger, str(tmp_repo))

    # L1 不拦截越界 — 任务仍然完成
    assert result.tasks_completed == 1
    # 但 Artifact 应该记录越界
    metas = artifact_store.list_for_task("solo")
    assert len(metas) >= 1
    artifact = SubtaskArtifact.load(metas[0].path)
    assert artifact.scope_check.is_clean is False
    assert any(
        p.startswith("forbidden/") for p in artifact.scope_check.out_of_scope_paths
    )
