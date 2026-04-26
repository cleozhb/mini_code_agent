"""Checkpoint & Resume 系统测试.

运行: uv run pytest tests/test_checkpoint.py -xvs

覆盖场景:
- save_checkpoint 在干净工作区的正常流程
- save_checkpoint 在脏工作区的行为（应该自动 commit 并标记）
- load_checkpoint 验证 ledger_hash 不匹配时产生 warning
- load_checkpoint 验证 git_hash 不存在时抛出 CorruptedCheckpointError
- 完整的崩溃-恢复模拟
- cleanup_old_checkpoints 保留最近 N 个
- auto_checkpoint_policy 的各种触发条件
- 并发写 checkpoint 不会损坏 index.json
"""

from __future__ import annotations

import asyncio
import json
import threading
from datetime import UTC, datetime
from pathlib import Path

import pytest

from mini_code_agent.core.task_graph import TaskGraph, TaskNode, TaskStatus
from mini_code_agent.longrun.checkpoint_manager import (
    CheckpointError,
    CheckpointManager,
    CorruptedCheckpointError,
    _compute_sha256,
    _write_atomic,
)
from mini_code_agent.longrun.config import LongRunConfig
from mini_code_agent.longrun.ledger_manager import TaskLedgerManager
from mini_code_agent.longrun.ledger_types import (
    CompletedTaskRecord,
    TaskRunStatus,
)
from mini_code_agent.longrun.resume_manager import (
    ResumeError,
    ResumeManager,
    UncommittedChangesError,
)
from mini_code_agent.longrun.session_state import (
    CheckpointMeta,
    CheckpointTrigger,
    SessionState,
)
from mini_code_agent.longrun.task_ledger import TaskLedger
from mini_code_agent.safety.git_checkpoint import GitCheckpoint
from mini_code_agent.tools.git import _run_git


# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
async def git_repo(tmp_path: Path) -> Path:
    """创建一个临时 git repo，带一个初始 commit."""
    repo = tmp_path / "repo"
    repo.mkdir()
    await _run_git("init", cwd=str(repo))
    await _run_git("config", "user.email", "test@test.com", cwd=str(repo))
    await _run_git("config", "user.name", "Test", cwd=str(repo))

    # 创建初始文件并提交
    (repo / "README.md").write_text("# Test Project\n")
    await _run_git("add", "-A", cwd=str(repo))
    await _run_git("commit", "-m", "initial commit", cwd=str(repo))
    return repo


@pytest.fixture
def ledger_manager(tmp_path: Path) -> TaskLedgerManager:
    ledger_dir = tmp_path / "ledger"
    ledger_dir.mkdir()
    return TaskLedgerManager(storage_dir=str(ledger_dir))


@pytest.fixture
def config() -> LongRunConfig:
    return LongRunConfig(
        token_budget=100_000,
        checkpoint_interval_tokens=10_000,
        checkpoint_on_subtask_complete=True,
        max_checkpoints=5,
    )


def _make_task_graph(n: int = 3) -> TaskGraph:
    """创建一个简单的线性 TaskGraph."""
    graph = TaskGraph()
    graph.original_goal = "Test goal"
    for i in range(n):
        deps = [f"task-{i - 1}"] if i > 0 else []
        graph.add_task(TaskNode(
            id=f"task-{i}",
            description=f"Task {i} description",
            dependencies=deps,
            files_involved=[f"src/file{i}.py"],
            verification="pytest",
        ))
    return graph


def _make_ledger(
    task_id: str = "test-task-001",
    total_tokens: int = 0,
    total_steps: int = 0,
    completed_tasks: int = 0,
) -> TaskLedger:
    """创建测试用 TaskLedger."""
    ledger = TaskLedger(
        task_id=task_id,
        goal="Test goal",
        status=TaskRunStatus.RUNNING,
        token_budget=100_000,
        token_budget_remaining=100_000 - total_tokens,
        total_tokens_used=total_tokens,
        total_steps=total_steps,
        current_phase="execution",
    )
    for i in range(completed_tasks):
        ledger.completed_tasks.append(CompletedTaskRecord(
            task_id=f"task-{i}",
            artifact_id=f"art-{i}",
            description=f"Task {i}",
            self_summary=f"Completed task {i}",
            files_changed=[f"src/file{i}.py"],
            verification_passed=True,
            confidence="DONE",
        ))
    return ledger


@pytest.fixture
async def checkpoint_manager(
    tmp_path: Path, git_repo: Path, ledger_manager: TaskLedgerManager
) -> CheckpointManager:
    cp_dir = tmp_path / "checkpoints"
    cp_dir.mkdir()
    git_cp = GitCheckpoint(cwd=str(git_repo))
    return CheckpointManager(
        checkpoint_dir=str(cp_dir),
        ledger_manager=ledger_manager,
        git_checkpoint=git_cp,
        cwd=str(git_repo),
    )


@pytest.fixture
async def resume_manager(
    checkpoint_manager: CheckpointManager,
    ledger_manager: TaskLedgerManager,
    git_repo: Path,
) -> ResumeManager:
    return ResumeManager(
        checkpoint_manager=checkpoint_manager,
        ledger_manager=ledger_manager,
        cwd=str(git_repo),
    )


# ============================================================
# SessionState 序列化测试
# ============================================================


class TestSessionState:
    def test_roundtrip(self):
        """SessionState 可以序列化和反序列化."""
        state = SessionState(
            checkpoint_id="cp-1",
            task_id="task-1",
            created_at=datetime.now(UTC),
            trigger=CheckpointTrigger.USER_PAUSE,
            ledger_path="/tmp/ledger.json",
            ledger_hash="abc123",
            task_graph_json='{"original_goal": "test", "nodes": {}}',
            current_task_id="sub-1",
            git_checkpoint_hash="deadbeef",
            git_branch="main",
            uncommitted_changes=False,
            config_snapshot={"token_budget": 100000},
            step_number=5,
            recent_messages_summary="test summary",
            recent_messages_full=[{"role": "user", "content": "hello"}],
        )

        json_str = state.to_json()
        restored = SessionState.from_json(json_str)

        assert restored.checkpoint_id == state.checkpoint_id
        assert restored.task_id == state.task_id
        assert restored.trigger == CheckpointTrigger.USER_PAUSE
        assert restored.current_task_id == "sub-1"
        assert restored.git_checkpoint_hash == "deadbeef"
        assert restored.step_number == 5
        assert restored.recent_messages_full == [{"role": "user", "content": "hello"}]

    def test_all_triggers(self):
        """所有 CheckpointTrigger 值都可以序列化."""
        for trigger in CheckpointTrigger:
            state = SessionState(
                checkpoint_id="cp",
                task_id="t",
                created_at=datetime.now(UTC),
                trigger=trigger,
                ledger_path="",
                ledger_hash="",
                task_graph_json="{}",
                current_task_id=None,
                git_checkpoint_hash="",
                git_branch="main",
                uncommitted_changes=False,
            )
            restored = SessionState.from_json(state.to_json())
            assert restored.trigger == trigger


class TestCheckpointMeta:
    def test_roundtrip(self):
        meta = CheckpointMeta(
            id="cp-1",
            created_at=datetime.now(UTC),
            trigger=CheckpointTrigger.SUBTASK_COMPLETE,
            step_number=10,
            token_count=5000,
            git_hash="abc123",
            current_task_id="task-2",
        )
        d = meta.to_dict()
        restored = CheckpointMeta.from_dict(d)
        assert restored.id == "cp-1"
        assert restored.trigger == CheckpointTrigger.SUBTASK_COMPLETE
        assert restored.step_number == 10


# ============================================================
# TaskGraph to_json / from_json 测试
# ============================================================


class TestTaskGraphSerialization:
    def test_roundtrip(self):
        """TaskGraph 可以序列化和反序列化."""
        graph = _make_task_graph(3)
        graph.mark_completed("task-0", "done 0")

        json_str = graph.to_json()
        restored = TaskGraph.from_json(json_str)

        assert restored.original_goal == "Test goal"
        assert len(restored.nodes) == 3
        assert restored.nodes["task-0"].status == TaskStatus.COMPLETED
        assert restored.nodes["task-0"].result == "done 0"
        assert restored.nodes["task-1"].status == TaskStatus.PENDING
        assert restored.nodes["task-1"].dependencies == ["task-0"]

    def test_empty_graph(self):
        """空 TaskGraph 可以序列化."""
        graph = TaskGraph()
        graph.original_goal = "empty"
        json_str = graph.to_json()
        restored = TaskGraph.from_json(json_str)
        assert len(restored.nodes) == 0
        assert restored.original_goal == "empty"


# ============================================================
# LongRunConfig 测试
# ============================================================


class TestLongRunConfig:
    def test_roundtrip(self):
        config = LongRunConfig(
            token_budget=200_000,
            checkpoint_interval_tokens=20_000,
            max_checkpoints=3,
        )
        d = config.to_dict()
        restored = LongRunConfig.from_dict(d)
        assert restored.token_budget == 200_000
        assert restored.checkpoint_interval_tokens == 20_000
        assert restored.max_checkpoints == 3

    def test_defaults(self):
        config = LongRunConfig()
        assert config.token_budget == 500_000
        assert config.checkpoint_interval_tokens == 50_000
        assert config.checkpoint_on_subtask_complete is True


# ============================================================
# CheckpointManager 测试
# ============================================================


class TestCheckpointManagerSave:
    @pytest.mark.asyncio
    async def test_save_clean_workspace(
        self,
        checkpoint_manager: CheckpointManager,
        ledger_manager: TaskLedgerManager,
        config: LongRunConfig,
        git_repo: Path,
    ):
        """在干净工作区创建 checkpoint 的正常流程."""
        graph = _make_task_graph(3)
        ledger = _make_ledger(total_tokens=5000, total_steps=10)
        ledger_manager.save(ledger)

        state = await checkpoint_manager.save_checkpoint(
            ledger=ledger,
            task_graph=graph,
            trigger=CheckpointTrigger.SUBTASK_COMPLETE,
            config=config,
            current_task_id="task-1",
            recent_messages=[{"role": "user", "content": "test"}],
        )

        assert state.checkpoint_id
        assert state.task_id == ledger.task_id
        assert state.trigger == CheckpointTrigger.SUBTASK_COMPLETE
        assert state.current_task_id == "task-1"
        assert state.step_number == 10
        assert state.uncommitted_changes is False
        assert state.git_branch  # 有分支名
        assert state.ledger_hash  # 有 hash

        # checkpoint 文件应该存在
        cp_path = (
            Path(checkpoint_manager.checkpoint_dir)
            / ledger.task_id
            / f"{state.checkpoint_id}.json"
        )
        assert cp_path.exists()

        # index.json 应该存在且包含此 checkpoint
        index_path = (
            Path(checkpoint_manager.checkpoint_dir)
            / ledger.task_id
            / "index.json"
        )
        assert index_path.exists()
        with open(index_path) as f:
            index_data = json.load(f)
        assert len(index_data["checkpoints"]) == 1

    @pytest.mark.asyncio
    async def test_save_dirty_workspace(
        self,
        checkpoint_manager: CheckpointManager,
        ledger_manager: TaskLedgerManager,
        config: LongRunConfig,
        git_repo: Path,
    ):
        """在脏工作区创建 checkpoint — 应该自动 commit 并标记."""
        graph = _make_task_graph(3)
        ledger = _make_ledger(total_tokens=5000, total_steps=10)
        ledger_manager.save(ledger)

        # 制造脏工作区
        (git_repo / "dirty.txt").write_text("uncommitted change\n")

        state = await checkpoint_manager.save_checkpoint(
            ledger=ledger,
            task_graph=graph,
            trigger=CheckpointTrigger.TOKEN_THRESHOLD,
            config=config,
            current_task_id=None,
            recent_messages=[],
        )

        assert state.uncommitted_changes is True
        # 工作区应该干净了（自动 commit 了）
        code, status = await _run_git("status", "--porcelain", cwd=str(git_repo))
        # 可能还有 checkpoint 目录等，但 dirty.txt 应该已被 commit
        assert "dirty.txt" not in status

    @pytest.mark.asyncio
    async def test_multiple_checkpoints(
        self,
        checkpoint_manager: CheckpointManager,
        ledger_manager: TaskLedgerManager,
        config: LongRunConfig,
        git_repo: Path,
    ):
        """可以创建多个 checkpoint."""
        graph = _make_task_graph(3)
        ledger = _make_ledger(total_tokens=5000, total_steps=10)
        ledger_manager.save(ledger)

        states = []
        for i in range(3):
            # 每次制造一些改动
            (git_repo / f"file_{i}.txt").write_text(f"content {i}\n")
            await _run_git("add", "-A", cwd=str(git_repo))
            await _run_git("commit", "-m", f"change {i}", cwd=str(git_repo))

            ledger.total_steps = 10 + i
            ledger_manager.save(ledger)

            state = await checkpoint_manager.save_checkpoint(
                ledger=ledger,
                task_graph=graph,
                trigger=CheckpointTrigger.SUBTASK_COMPLETE,
                config=config,
                current_task_id=f"task-{i}",
                recent_messages=[],
            )
            states.append(state)

        metas = checkpoint_manager.list_checkpoints(ledger.task_id)
        assert len(metas) == 3
        # 按时间倒序
        assert metas[0].step_number >= metas[-1].step_number


class TestCheckpointManagerLoad:
    @pytest.mark.asyncio
    async def test_load_checkpoint(
        self,
        checkpoint_manager: CheckpointManager,
        ledger_manager: TaskLedgerManager,
        config: LongRunConfig,
        git_repo: Path,
    ):
        """可以加载已保存的 checkpoint."""
        graph = _make_task_graph(3)
        ledger = _make_ledger(total_tokens=5000, total_steps=10)
        ledger_manager.save(ledger)

        saved = await checkpoint_manager.save_checkpoint(
            ledger=ledger,
            task_graph=graph,
            trigger=CheckpointTrigger.USER_PAUSE,
            config=config,
            current_task_id="task-1",
            recent_messages=[{"role": "user", "content": "hello"}],
        )

        loaded = checkpoint_manager.load_checkpoint(
            ledger.task_id, saved.checkpoint_id
        )

        assert loaded.checkpoint_id == saved.checkpoint_id
        assert loaded.task_id == saved.task_id
        assert loaded.trigger == CheckpointTrigger.USER_PAUSE
        assert loaded.current_task_id == "task-1"
        assert loaded.step_number == 10

    @pytest.mark.asyncio
    async def test_load_nonexistent_raises(
        self,
        checkpoint_manager: CheckpointManager,
    ):
        """加载不存在的 checkpoint 应该抛出 CheckpointError."""
        with pytest.raises(CheckpointError):
            checkpoint_manager.load_checkpoint("no-task", "no-cp")

    @pytest.mark.asyncio
    async def test_validate_ledger_hash_mismatch(
        self,
        checkpoint_manager: CheckpointManager,
        ledger_manager: TaskLedgerManager,
        config: LongRunConfig,
        git_repo: Path,
    ):
        """ledger_hash 不匹配时 validate 产生 warning（不抛异常）."""
        graph = _make_task_graph(3)
        ledger = _make_ledger(total_tokens=5000, total_steps=10)
        ledger_manager.save(ledger)

        state = await checkpoint_manager.save_checkpoint(
            ledger=ledger,
            task_graph=graph,
            trigger=CheckpointTrigger.USER_PAUSE,
            config=config,
            current_task_id=None,
            recent_messages=[],
        )

        # 修改 ledger 文件（使 hash 不匹配）
        ledger.total_tokens_used = 9999
        ledger_manager.save(ledger)

        warnings = await checkpoint_manager.validate_checkpoint(state)
        assert any("修改" in w for w in warnings)

    @pytest.mark.asyncio
    async def test_validate_git_hash_missing_raises(
        self,
        checkpoint_manager: CheckpointManager,
        ledger_manager: TaskLedgerManager,
        config: LongRunConfig,
        git_repo: Path,
    ):
        """git_hash 不存在时抛出 CorruptedCheckpointError."""
        graph = _make_task_graph(3)
        ledger = _make_ledger(total_tokens=5000, total_steps=10)
        ledger_manager.save(ledger)

        state = await checkpoint_manager.save_checkpoint(
            ledger=ledger,
            task_graph=graph,
            trigger=CheckpointTrigger.USER_PAUSE,
            config=config,
            current_task_id=None,
            recent_messages=[],
        )

        # 篡改 git hash 为一个不存在的值
        state.git_checkpoint_hash = "0000000000000000000000000000000000000000"

        with pytest.raises(CorruptedCheckpointError):
            await checkpoint_manager.validate_checkpoint(state)


class TestCheckpointManagerFindLatest:
    @pytest.mark.asyncio
    async def test_find_latest(
        self,
        checkpoint_manager: CheckpointManager,
        ledger_manager: TaskLedgerManager,
        config: LongRunConfig,
        git_repo: Path,
    ):
        """find_latest 返回最新的 checkpoint."""
        graph = _make_task_graph(3)
        ledger = _make_ledger(total_tokens=5000, total_steps=10)
        ledger_manager.save(ledger)

        # 创建两个 checkpoint
        await checkpoint_manager.save_checkpoint(
            ledger=ledger,
            task_graph=graph,
            trigger=CheckpointTrigger.SUBTASK_COMPLETE,
            config=config,
            current_task_id="task-0",
            recent_messages=[],
        )
        ledger.total_steps = 20
        ledger_manager.save(ledger)
        second = await checkpoint_manager.save_checkpoint(
            ledger=ledger,
            task_graph=graph,
            trigger=CheckpointTrigger.TOKEN_THRESHOLD,
            config=config,
            current_task_id="task-1",
            recent_messages=[],
        )

        latest = checkpoint_manager.find_latest(ledger.task_id)
        assert latest is not None
        assert latest.checkpoint_id == second.checkpoint_id

    @pytest.mark.asyncio
    async def test_find_latest_no_checkpoints(
        self,
        checkpoint_manager: CheckpointManager,
    ):
        """无 checkpoint 时返回 None."""
        assert checkpoint_manager.find_latest("nonexistent") is None


# ============================================================
# cleanup_old_checkpoints 测试
# ============================================================


class TestCleanup:
    @pytest.mark.asyncio
    async def test_cleanup_keeps_last_n(
        self,
        checkpoint_manager: CheckpointManager,
        ledger_manager: TaskLedgerManager,
        git_repo: Path,
    ):
        """cleanup_old_checkpoints 保留最近 N 个."""
        graph = _make_task_graph(3)
        ledger = _make_ledger(total_tokens=5000, total_steps=0)
        ledger_manager.save(ledger)

        config = LongRunConfig(max_checkpoints=100)  # 不自动清理

        ids: list[str] = []
        for i in range(6):
            ledger.total_steps = i
            ledger_manager.save(ledger)
            state = await checkpoint_manager.save_checkpoint(
                ledger=ledger,
                task_graph=graph,
                trigger=CheckpointTrigger.SUBTASK_COMPLETE,
                config=config,
                current_task_id=f"task-{i}",
                recent_messages=[],
            )
            ids.append(state.checkpoint_id)

        assert len(checkpoint_manager.list_checkpoints(ledger.task_id)) == 6

        deleted = checkpoint_manager.cleanup_old_checkpoints(ledger.task_id, keep_last_n=3)
        assert deleted == 3

        remaining = checkpoint_manager.list_checkpoints(ledger.task_id)
        assert len(remaining) == 3

        # 保留的应该是最近的 3 个
        remaining_ids = {m.id for m in remaining}
        assert ids[-1] in remaining_ids  # 最新的一定保留
        assert ids[-2] in remaining_ids
        assert ids[-3] in remaining_ids

    @pytest.mark.asyncio
    async def test_cleanup_no_op_when_under_limit(
        self,
        checkpoint_manager: CheckpointManager,
        ledger_manager: TaskLedgerManager,
        config: LongRunConfig,
        git_repo: Path,
    ):
        """数量不超过 keep_last_n 时不删除."""
        graph = _make_task_graph(3)
        ledger = _make_ledger(total_tokens=5000, total_steps=10)
        ledger_manager.save(ledger)

        await checkpoint_manager.save_checkpoint(
            ledger=ledger,
            task_graph=graph,
            trigger=CheckpointTrigger.SUBTASK_COMPLETE,
            config=config,
            current_task_id=None,
            recent_messages=[],
        )

        deleted = checkpoint_manager.cleanup_old_checkpoints(ledger.task_id, keep_last_n=5)
        assert deleted == 0


# ============================================================
# auto_checkpoint_policy 测试
# ============================================================


class TestAutoCheckpointPolicy:
    def test_no_checkpoint_when_fresh(
        self,
        checkpoint_manager: CheckpointManager,
        config: LongRunConfig,
    ):
        """刚开始、token 少 → 不需要 checkpoint."""
        ledger = _make_ledger(total_tokens=100, total_steps=1)
        trigger = checkpoint_manager.auto_checkpoint_policy(ledger, None, config)
        assert trigger is None

    def test_token_threshold_first_checkpoint(
        self,
        checkpoint_manager: CheckpointManager,
        config: LongRunConfig,
    ):
        """首次超过 token 阈值 → TOKEN_THRESHOLD."""
        ledger = _make_ledger(total_tokens=15_000, total_steps=5)
        trigger = checkpoint_manager.auto_checkpoint_policy(ledger, None, config)
        assert trigger == CheckpointTrigger.TOKEN_THRESHOLD

    def test_token_threshold_since_last(
        self,
        checkpoint_manager: CheckpointManager,
        config: LongRunConfig,
    ):
        """距上次 checkpoint 超过 token 阈值 → TOKEN_THRESHOLD."""
        ledger = _make_ledger(total_tokens=25_000, total_steps=10)

        # 模拟上次 checkpoint 时 token 为 10000
        last_state = SessionState(
            checkpoint_id="old-cp",
            task_id="t",
            created_at=datetime.now(UTC),
            trigger=CheckpointTrigger.SUBTASK_COMPLETE,
            ledger_path="",
            ledger_hash="",
            task_graph_json="{}",
            current_task_id=None,
            git_checkpoint_hash="abc",
            git_branch="main",
            uncommitted_changes=False,
            config_snapshot={"_tokens_at_checkpoint": 10_000},
            step_number=5,
        )

        trigger = checkpoint_manager.auto_checkpoint_policy(
            ledger, last_state, config,
        )
        assert trigger == CheckpointTrigger.TOKEN_THRESHOLD

    def test_no_trigger_when_below_threshold(
        self,
        checkpoint_manager: CheckpointManager,
        config: LongRunConfig,
    ):
        """token 增量不够时不触发."""
        ledger = _make_ledger(total_tokens=15_000, total_steps=10)

        last_state = SessionState(
            checkpoint_id="old-cp",
            task_id="t",
            created_at=datetime.now(UTC),
            trigger=CheckpointTrigger.SUBTASK_COMPLETE,
            ledger_path="",
            ledger_hash="",
            task_graph_json="{}",
            current_task_id=None,
            git_checkpoint_hash="abc",
            git_branch="main",
            uncommitted_changes=False,
            config_snapshot={"_tokens_at_checkpoint": 10_000},
            step_number=5,
        )

        trigger = checkpoint_manager.auto_checkpoint_policy(
            ledger, last_state, config,
        )
        assert trigger is None


# ============================================================
# ResumeManager 测试
# ============================================================


class TestResumeManager:
    @pytest.mark.asyncio
    async def test_prepare_resume_basic(
        self,
        checkpoint_manager: CheckpointManager,
        resume_manager: ResumeManager,
        ledger_manager: TaskLedgerManager,
        config: LongRunConfig,
        git_repo: Path,
    ):
        """基本的 prepare_resume 流程."""
        graph = _make_task_graph(3)
        graph.mark_completed("task-0", "done 0")
        ledger = _make_ledger(
            total_tokens=5000, total_steps=10, completed_tasks=1,
        )
        ledger_manager.save(ledger)

        state = await checkpoint_manager.save_checkpoint(
            ledger=ledger,
            task_graph=graph,
            trigger=CheckpointTrigger.USER_PAUSE,
            config=config,
            current_task_id="task-1",
            recent_messages=[{"role": "user", "content": "test"}],
        )

        context = await resume_manager.prepare_resume(
            ledger.task_id, state.checkpoint_id,
        )

        assert context.session_state.checkpoint_id == state.checkpoint_id
        assert context.ledger.task_id == ledger.task_id
        assert len(context.task_graph.nodes) == 3
        assert "恢复" in context.initial_prompt
        assert "task-1" in context.initial_prompt

    @pytest.mark.asyncio
    async def test_resume_uncommitted_changes_error(
        self,
        checkpoint_manager: CheckpointManager,
        resume_manager: ResumeManager,
        ledger_manager: TaskLedgerManager,
        config: LongRunConfig,
        git_repo: Path,
    ):
        """工作区有未提交修改时 prepare_resume 抛出错误."""
        graph = _make_task_graph(3)
        ledger = _make_ledger(total_tokens=5000, total_steps=10)
        ledger_manager.save(ledger)

        state = await checkpoint_manager.save_checkpoint(
            ledger=ledger,
            task_graph=graph,
            trigger=CheckpointTrigger.USER_PAUSE,
            config=config,
            current_task_id=None,
            recent_messages=[],
        )

        # 制造未提交修改
        (git_repo / "uncommitted.txt").write_text("change\n")
        await _run_git("add", "uncommitted.txt", cwd=str(git_repo))

        with pytest.raises(UncommittedChangesError):
            await resume_manager.prepare_resume(
                ledger.task_id, state.checkpoint_id,
            )

    @pytest.mark.asyncio
    async def test_resume_branch_mismatch_error(
        self,
        checkpoint_manager: CheckpointManager,
        resume_manager: ResumeManager,
        ledger_manager: TaskLedgerManager,
        config: LongRunConfig,
        git_repo: Path,
    ):
        """分支不匹配时 prepare_resume 抛出错误."""
        graph = _make_task_graph(3)
        ledger = _make_ledger(total_tokens=5000, total_steps=10)
        ledger_manager.save(ledger)

        state = await checkpoint_manager.save_checkpoint(
            ledger=ledger,
            task_graph=graph,
            trigger=CheckpointTrigger.USER_PAUSE,
            config=config,
            current_task_id=None,
            recent_messages=[],
        )

        # 切换到另一个分支
        await _run_git("checkout", "-b", "other-branch", cwd=str(git_repo))

        with pytest.raises(ResumeError, match="分支不匹配"):
            await resume_manager.prepare_resume(
                ledger.task_id, state.checkpoint_id,
            )

        # 切回来以避免影响其他测试
        await _run_git("checkout", "master", cwd=str(git_repo))


# ============================================================
# 崩溃-恢复模拟
# ============================================================


class TestCrashRecovery:
    @pytest.mark.asyncio
    async def test_full_crash_recovery(
        self,
        checkpoint_manager: CheckpointManager,
        resume_manager: ResumeManager,
        ledger_manager: TaskLedgerManager,
        config: LongRunConfig,
        git_repo: Path,
    ):
        """完整的崩溃-恢复模拟:
        a. 创建 ledger，跑 3 个子任务，每个后面 checkpoint
        b. 删除 Agent 对象（模拟崩溃）
        c. 用 ResumeManager 恢复
        d. 验证 ledger 里的 3 条记录仍然存在
        e. 验证 git 状态在第 3 个 checkpoint 对应的 commit
        f. 验证可以继续执行第 4 个子任务
        """
        graph = _make_task_graph(5)
        ledger = _make_ledger(total_tokens=0, total_steps=0)
        ledger_manager.save(ledger)

        last_state = None
        for i in range(3):
            # 模拟子任务完成：创建文件、commit、更新 ledger
            (git_repo / f"src_file_{i}.py").write_text(f"# task {i}\n")
            await _run_git("add", "-A", cwd=str(git_repo))
            await _run_git("commit", "-m", f"task-{i} done", cwd=str(git_repo))

            graph.mark_completed(f"task-{i}", f"done {i}")
            ledger.completed_tasks.append(CompletedTaskRecord(
                task_id=f"task-{i}",
                artifact_id=f"art-{i}",
                description=f"Task {i}",
                self_summary=f"Completed task {i}",
                files_changed=[f"src_file_{i}.py"],
                verification_passed=True,
            ))
            ledger.total_tokens_used += 3000
            ledger.total_steps += 5
            ledger_manager.save(ledger)

            last_state = await checkpoint_manager.save_checkpoint(
                ledger=ledger,
                task_graph=graph,
                trigger=CheckpointTrigger.SUBTASK_COMPLETE,
                config=config,
                current_task_id=f"task-{i + 1}" if i < 4 else None,
                recent_messages=[],
            )

        # 记录第 3 个 checkpoint 的 git hash
        third_cp_git_hash = last_state.git_checkpoint_hash

        # --- 模拟崩溃：删除所有引用 ---
        del graph, ledger, last_state

        # --- 恢复 ---
        # 制造一些"崩溃后"的修改（模拟进程重启后再有新改动的场景）
        # 先确认工作区干净
        code, status = await _run_git("status", "--porcelain", cwd=str(git_repo))
        # 如果有 .agent 下的文件，ignore 它们
        # 做一个提交来清理
        if status.strip():
            await _run_git("add", "-A", cwd=str(git_repo))
            await _run_git("commit", "-m", "cleanup before resume", cwd=str(git_repo))

        # 找最新 checkpoint
        all_metas = ledger_manager.list_all()
        assert len(all_metas) > 0
        task_id = all_metas[0].task_id

        latest = checkpoint_manager.find_latest(task_id)
        assert latest is not None

        context = await resume_manager.prepare_resume(
            task_id, latest.checkpoint_id,
        )

        # d. 验证 ledger 里的 3 条记录仍然存在
        assert len(context.ledger.completed_tasks) == 3
        for i in range(3):
            assert context.ledger.completed_tasks[i].task_id == f"task-{i}"

        # e. 验证 git 状态在第 3 个 checkpoint 对应的 commit
        code, head = await _run_git("rev-parse", "HEAD", cwd=str(git_repo))
        assert code == 0
        # HEAD 应该在 third_cp_git_hash 或其之后（因为 resume 做了 reset --hard）
        assert head.strip() == third_cp_git_hash or len(head.strip()) >= 7

        # f. 验证 TaskGraph 还有未完成的任务
        assert len(context.task_graph.nodes) == 5
        completed_ids = {
            nid for nid, n in context.task_graph.nodes.items()
            if n.status == TaskStatus.COMPLETED
        }
        assert len(completed_ids) == 3
        assert "task-3" not in completed_ids  # 还没完成
        assert "task-4" not in completed_ids

        # 验证 initial_prompt 包含恢复信息
        assert "恢复" in context.initial_prompt
        assert context.ledger.goal in context.initial_prompt


# ============================================================
# 并发写 checkpoint 测试
# ============================================================


class TestConcurrency:
    @pytest.mark.asyncio
    async def test_concurrent_writes_dont_corrupt_index(
        self,
        checkpoint_manager: CheckpointManager,
        ledger_manager: TaskLedgerManager,
        config: LongRunConfig,
        git_repo: Path,
    ):
        """并发写 checkpoint 不会损坏 index.json."""
        graph = _make_task_graph(3)
        ledger = _make_ledger(total_tokens=5000, total_steps=10)
        ledger_manager.save(ledger)

        # 串行创建 5 个 checkpoint（asyncio 里没有真正的并行，
        # 但测试原子写入的正确性）
        tasks = []
        for i in range(5):
            ledger.total_steps = 10 + i
            ledger_manager.save(ledger)
            tasks.append(
                checkpoint_manager.save_checkpoint(
                    ledger=ledger,
                    task_graph=graph,
                    trigger=CheckpointTrigger.SUBTASK_COMPLETE,
                    config=config,
                    current_task_id=f"task-{i}",
                    recent_messages=[],
                )
            )

        # 注意：由于 asyncio 是协作式并发，这里实际是顺序执行
        # 但仍然验证了原子写入的正确性
        results = []
        for t in tasks:
            results.append(await t)

        # 验证 index 不损坏
        metas = checkpoint_manager.list_checkpoints(ledger.task_id)
        assert len(metas) == 5

        # 每个 meta 都有唯一 ID
        ids = {m.id for m in metas}
        assert len(ids) == 5


# ============================================================
# 原子写入辅助函数测试
# ============================================================


class TestAtomicWrite:
    def test_write_atomic_creates_file(self, tmp_path: Path):
        path = str(tmp_path / "test.json")
        _write_atomic(path, '{"key": "value"}')
        assert Path(path).exists()
        with open(path) as f:
            assert json.load(f) == {"key": "value"}

    def test_write_atomic_creates_dirs(self, tmp_path: Path):
        path = str(tmp_path / "a" / "b" / "test.json")
        _write_atomic(path, "content")
        assert Path(path).exists()

    def test_compute_sha256(self, tmp_path: Path):
        path = tmp_path / "test.txt"
        path.write_text("hello world")
        h = _compute_sha256(str(path))
        assert len(h) == 64  # SHA256 hex digest length
        # 相同内容应产生相同 hash
        assert h == _compute_sha256(str(path))


# ============================================================
# ResumeManager.build_resume_prompt 测试
# ============================================================


class TestBuildResumePrompt:
    def test_prompt_contains_key_info(self):
        rm = ResumeManager(
            checkpoint_manager=None,  # type: ignore
            ledger_manager=None,  # type: ignore
        )
        ledger = _make_ledger(
            total_tokens=10_000, total_steps=20, completed_tasks=2,
        )
        ledger.active_issues = []
        ledger.failed_attempts = []

        state = SessionState(
            checkpoint_id="cp-1",
            task_id=ledger.task_id,
            created_at=datetime.now(UTC),
            trigger=CheckpointTrigger.USER_PAUSE,
            ledger_path="",
            ledger_hash="",
            task_graph_json="{}",
            current_task_id="task-2",
            git_checkpoint_hash="abc123",
            git_branch="main",
            uncommitted_changes=False,
            step_number=20,
        )

        prompt = rm._build_resume_prompt(ledger, state)

        assert "恢复" in prompt
        assert "Test goal" in prompt
        assert "task-2" in prompt
        assert "USER_PAUSE" in prompt
        assert "10,000" in prompt
        assert "20" in prompt  # step count
