"""Ledger 集成测试 — 验证 Ledger 在各组件间的端到端协同.

不依赖 LLM API，通过手动构造 TaskGraph + Ledger 来模拟完整流程。

覆盖场景：
1. 创建 Ledger → 执行子任务 → 更新状态 → 验证 context 注入
2. Context summary 在不同阶段的输出变化
3. Checkpoint/Resume：保存 → 重新加载 → 检查一致性
4. /status 和 /ledger 输出格式验证
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest

from mini_code_agent.core.task_graph import TaskGraph, TaskNode, TaskStatus
from mini_code_agent.longrun import (
    ActiveIssue,
    CompletedTaskRecord,
    DecisionRecord,
    FailedAttemptRecord,
    TaskLedgerManager,
    TaskRunStatus,
)
from mini_code_agent.longrun.task_ledger import TaskLedger


# ──────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────


@pytest.fixture()
def tmp_storage(tmp_path: Path) -> str:
    return str(tmp_path / "ledger_storage")


@pytest.fixture()
def sample_graph() -> TaskGraph:
    """构造一个简单的 3 步 TaskGraph: init → write_code → test."""
    g = TaskGraph()
    g.original_goal = "创建一个 hello.py 文件，打印 Hello World"
    g.add_task(TaskNode(
        id="task-1",
        description="初始化项目目录结构",
        dependencies=[],
        files_involved=["hello.py"],
        verification="检查文件是否存在",
    ))
    g.add_task(TaskNode(
        id="task-2",
        description="编写 hello.py 文件",
        dependencies=["task-1"],
        files_involved=["hello.py"],
        verification="运行 python hello.py 检查输出",
    ))
    g.add_task(TaskNode(
        id="task-3",
        description="验证输出是否为 Hello World",
        dependencies=["task-2"],
        files_involved=["hello.py"],
        verification="输出包含 Hello World",
    ))
    return g


@pytest.fixture()
def manager_and_ledger(
    tmp_storage: str, sample_graph: TaskGraph
) -> tuple[TaskLedgerManager, TaskLedger]:
    mgr = TaskLedgerManager(storage_dir=tmp_storage)
    ledger = mgr.create(
        goal="创建一个 hello.py 文件，打印 Hello World",
        task_graph=sample_graph,
        budget=200_000,
    )
    return mgr, ledger


# ──────────────────────────────────────────────────────────────
# Test: 端到端流程模拟
# ──────────────────────────────────────────────────────────────


class TestEndToEndFlow:
    """模拟完整的 --long-run 执行流程（不调用 LLM）."""

    def test_full_lifecycle(
        self, manager_and_ledger: tuple[TaskLedgerManager, TaskLedger]
    ) -> None:
        mgr, ledger = manager_and_ledger

        # 1. 初始状态
        assert ledger.status == TaskRunStatus.NOT_STARTED
        assert ledger.current_phase == "planning"
        assert ledger.total_tokens_used == 0
        assert ledger.token_budget == 200_000
        assert ledger.token_budget_remaining == 200_000
        assert len(ledger.completed_tasks) == 0

        # 2. 开始执行
        ledger.status = TaskRunStatus.RUNNING
        mgr.update_phase(ledger, "execution")
        assert ledger.status == TaskRunStatus.RUNNING
        assert ledger.current_phase == "execution"

        # 3. 执行 task-1
        mgr.update_current_task(ledger, "task-1")
        assert ledger.current_task_id == "task-1"

        # 模拟 task-1 完成（手动添加 CompletedTaskRecord）
        record1 = CompletedTaskRecord(
            task_id="task-1",
            artifact_id=str(uuid4()),
            description="初始化项目目录结构",
            self_summary="创建了项目目录和 hello.py 空文件",
            files_changed=["hello.py"],
            verification_passed=True,
            confidence="DONE",
            step_number_start=0,
            step_number_end=3,
            token_count=500,
            timestamp=datetime.now(UTC),
        )
        ledger.completed_tasks.append(record1)
        mgr.update_resources(ledger, tokens=500, steps=3, wall_time=2.5)
        assert ledger.total_tokens_used == 500
        assert ledger.total_steps == 3
        assert ledger.token_budget_remaining == 199_500

        # 4. 执行 task-2
        mgr.update_current_task(ledger, "task-2")
        record2 = CompletedTaskRecord(
            task_id="task-2",
            artifact_id=str(uuid4()),
            description="编写 hello.py 文件",
            self_summary="写入 print('Hello World') 到 hello.py",
            files_changed=["hello.py"],
            verification_passed=True,
            confidence="DONE",
            step_number_start=3,
            step_number_end=6,
            token_count=800,
            timestamp=datetime.now(UTC),
        )
        ledger.completed_tasks.append(record2)
        mgr.update_resources(ledger, tokens=800, steps=3, wall_time=3.0)
        assert ledger.total_tokens_used == 1300
        assert ledger.total_steps == 6

        # 5. 执行 task-3
        mgr.update_current_task(ledger, "task-3")
        record3 = CompletedTaskRecord(
            task_id="task-3",
            artifact_id=str(uuid4()),
            description="验证输出是否为 Hello World",
            self_summary="运行 python hello.py 输出正确",
            files_changed=[],
            verification_passed=True,
            confidence="DONE",
            step_number_start=6,
            step_number_end=8,
            token_count=340,
            timestamp=datetime.now(UTC),
        )
        ledger.completed_tasks.append(record3)
        mgr.update_resources(ledger, tokens=340, steps=2, wall_time=1.5)

        # 6. 完成
        ledger.status = TaskRunStatus.COMPLETED
        mgr.update_phase(ledger, "done")
        mgr.update_current_task(ledger, None)
        mgr.save(ledger)

        # 验证最终状态
        assert ledger.status == TaskRunStatus.COMPLETED
        assert ledger.current_phase == "done"
        assert len(ledger.completed_tasks) == 3
        assert ledger.total_tokens_used == 1640
        assert ledger.total_steps == 8
        assert ledger.token_budget_remaining == 200_000 - 1640

    def test_failure_and_retry(
        self, manager_and_ledger: tuple[TaskLedgerManager, TaskLedger]
    ) -> None:
        """模拟子任务失败 → 记录失败 → 重试成功."""
        mgr, ledger = manager_and_ledger
        ledger.status = TaskRunStatus.RUNNING
        mgr.update_phase(ledger, "execution")
        mgr.update_current_task(ledger, "task-1")

        # 第一次尝试失败
        failed = FailedAttemptRecord(
            task_id="task-1",
            artifact_id=str(uuid4()),
            approach_description="尝试直接 touch 创建文件",
            failure_reason="权限不足",
            step_number=1,
            lesson_learned="需要先检查目录权限",
        )
        ledger.failed_attempts.append(failed)
        mgr.save(ledger)
        assert len(ledger.failed_attempts) == 1

        # 记录问题
        issue_id = str(uuid4())
        issue = ActiveIssue(
            id=issue_id,
            description="目标目录权限不足",
            source_task_id="task-1",
            severity="high",
            resolution_attempts=1,
            resolved=False,
        )
        mgr.record_active_issue(ledger, issue)
        assert len(ledger.active_issues) == 1

        # 解决问题并重试成功
        mgr.resolve_issue(ledger, issue_id)
        assert len(ledger.active_issues) == 0
        assert len(ledger.resolved_issues) == 1
        assert ledger.resolved_issues[0].resolved is True

        record = CompletedTaskRecord(
            task_id="task-1",
            artifact_id=str(uuid4()),
            description="初始化项目目录结构",
            self_summary="先 mkdir 再 touch 创建成功",
            files_changed=["hello.py"],
            verification_passed=True,
            confidence="DONE",
            step_number_start=2,
            step_number_end=5,
            token_count=700,
            timestamp=datetime.now(UTC),
        )
        ledger.completed_tasks.append(record)
        mgr.save(ledger)
        assert len(ledger.completed_tasks) == 1


# ──────────────────────────────────────────────────────────────
# Test: Context Summary 随阶段变化
# ──────────────────────────────────────────────────────────────


class TestContextInjection:
    """验证 build_context_summary 在不同阶段输出不同内容."""

    def test_initial_context_has_goal_and_phase(
        self, manager_and_ledger: tuple[TaskLedgerManager, TaskLedger]
    ) -> None:
        """第 1 轮：context 里应该只有 '目标' 和 '阶段' 部分."""
        mgr, ledger = manager_and_ledger
        summary = mgr.build_context_summary(ledger)
        assert "目标" in summary or "创建一个 hello.py" in summary
        assert "planning" in summary or "阶段" in summary
        # 不应该有 "最近完成" 因为还没完成任何任务
        assert "最近完成" not in summary or "0" in summary

    def test_mid_execution_context_has_current_task(
        self, manager_and_ledger: tuple[TaskLedgerManager, TaskLedger]
    ) -> None:
        """中间轮：context 里应该有 '当前任务' 信息."""
        mgr, ledger = manager_and_ledger
        ledger.status = TaskRunStatus.RUNNING
        mgr.update_phase(ledger, "execution")
        mgr.update_current_task(ledger, "task-2")

        # 先完成 task-1
        record = CompletedTaskRecord(
            task_id="task-1",
            artifact_id=str(uuid4()),
            description="初始化项目目录结构",
            self_summary="创建了项目目录",
            files_changed=["hello.py"],
            verification_passed=True,
            confidence="DONE",
            step_number_start=0,
            step_number_end=3,
            token_count=500,
            timestamp=datetime.now(UTC),
        )
        ledger.completed_tasks.append(record)
        mgr.save(ledger)

        summary = mgr.build_context_summary(ledger)
        assert "task-2" in summary
        assert "execution" in summary or "执行" in summary

    def test_completed_context_has_recent_completions(
        self, manager_and_ledger: tuple[TaskLedgerManager, TaskLedger]
    ) -> None:
        """完成后：context 里应该有 '最近完成' 信息."""
        mgr, ledger = manager_and_ledger
        ledger.status = TaskRunStatus.COMPLETED
        mgr.update_phase(ledger, "done")

        for i in range(1, 4):
            record = CompletedTaskRecord(
                task_id=f"task-{i}",
                artifact_id=str(uuid4()),
                description=f"步骤 {i} 的描述",
                self_summary=f"步骤 {i} 已完成",
                files_changed=["hello.py"],
                verification_passed=True,
                confidence="DONE",
                step_number_start=0,
                step_number_end=3,
                token_count=300,
                timestamp=datetime.now(UTC),
            )
            ledger.completed_tasks.append(record)
        mgr.save(ledger)

        summary = mgr.build_context_summary(ledger)
        # 应该包含已完成任务的信息
        assert "task-1" in summary or "步骤" in summary
        assert "COMPLETED" in summary or "done" in summary

    def test_context_with_issues_and_failures(
        self, manager_and_ledger: tuple[TaskLedgerManager, TaskLedger]
    ) -> None:
        """有问题和失败时，context 优先级应该更高."""
        mgr, ledger = manager_and_ledger
        ledger.status = TaskRunStatus.RUNNING
        mgr.update_phase(ledger, "execution")

        # 添加活跃问题
        issue = ActiveIssue(
            id=str(uuid4()),
            description="API 连接超时",
            source_task_id="task-1",
            severity="high",
            resolution_attempts=0,
            resolved=False,
        )
        mgr.record_active_issue(ledger, issue)

        # 添加失败记录
        failed = FailedAttemptRecord(
            task_id="task-1",
            artifact_id=str(uuid4()),
            approach_description="直接调用 API",
            failure_reason="连接超时",
            step_number=1,
            lesson_learned="需要增加重试逻辑",
        )
        ledger.failed_attempts.append(failed)
        mgr.save(ledger)

        summary = mgr.build_context_summary(ledger)
        # 问题和失败教训应该出现在 context 中
        assert "API 连接超时" in summary or "连接超时" in summary


# ──────────────────────────────────────────────────────────────
# Test: Checkpoint/Resume
# ──────────────────────────────────────────────────────────────


class TestCheckpointResume:
    """验证保存→加载→恢复的完整流程."""

    def test_save_and_reload_preserves_state(
        self, manager_and_ledger: tuple[TaskLedgerManager, TaskLedger]
    ) -> None:
        """保存后重新加载，所有状态应该一致."""
        mgr, ledger = manager_and_ledger
        task_id = ledger.task_id

        # 模拟执行到一半
        ledger.status = TaskRunStatus.RUNNING
        mgr.update_phase(ledger, "execution")
        mgr.update_current_task(ledger, "task-2")

        record = CompletedTaskRecord(
            task_id="task-1",
            artifact_id=str(uuid4()),
            description="初始化项目目录结构",
            self_summary="创建了项目目录",
            files_changed=["hello.py"],
            verification_passed=True,
            confidence="DONE",
            step_number_start=0,
            step_number_end=3,
            token_count=500,
            timestamp=datetime.now(UTC),
        )
        ledger.completed_tasks.append(record)
        mgr.update_resources(ledger, tokens=500, steps=3, wall_time=2.5)

        # 添加决策
        decision = DecisionRecord(
            description="使用 print() 而非 sys.stdout.write()",
            reason="更简洁",
            source_task_id="task-1",
            reversible=True,
            step_number=2,
        )
        ledger.decisions_made.append(decision)
        mgr.save(ledger)

        # 重新加载
        loaded = mgr.load(task_id)
        assert loaded.task_id == task_id
        assert loaded.status == TaskRunStatus.RUNNING
        assert loaded.current_phase == "execution"
        assert loaded.current_task_id == "task-2"
        assert len(loaded.completed_tasks) == 1
        assert loaded.completed_tasks[0].task_id == "task-1"
        assert loaded.completed_tasks[0].self_summary == "创建了项目目录"
        assert loaded.total_tokens_used == 500
        assert loaded.total_steps == 3
        assert loaded.token_budget_remaining == 199_500
        assert len(loaded.decisions_made) == 1
        assert loaded.decisions_made[0].description == "使用 print() 而非 sys.stdout.write()"

    def test_list_all_shows_saved_ledger(
        self, manager_and_ledger: tuple[TaskLedgerManager, TaskLedger]
    ) -> None:
        """list_all 应该能列出保存的 ledger."""
        mgr, ledger = manager_and_ledger
        metas = mgr.list_all()
        assert len(metas) == 1
        assert metas[0].task_id == ledger.task_id
        assert metas[0].goal == "创建一个 hello.py 文件，打印 Hello World"

    def test_resume_summary(
        self, manager_and_ledger: tuple[TaskLedgerManager, TaskLedger]
    ) -> None:
        """get_summary_for_resume 应该包含断点恢复信息."""
        mgr, ledger = manager_and_ledger
        ledger.status = TaskRunStatus.RUNNING
        mgr.update_phase(ledger, "execution")
        mgr.update_current_task(ledger, "task-2")

        record = CompletedTaskRecord(
            task_id="task-1",
            artifact_id=str(uuid4()),
            description="初始化项目目录结构",
            self_summary="创建了项目目录",
            files_changed=["hello.py"],
            verification_passed=True,
            confidence="DONE",
            step_number_start=0,
            step_number_end=3,
            token_count=500,
            timestamp=datetime.now(UTC),
        )
        ledger.completed_tasks.append(record)
        mgr.save(ledger)

        summary = mgr.get_summary_for_resume(ledger)
        assert "断点恢复" in summary
        assert "task-2" in summary
        assert "创建了项目目录" in summary

    def test_graph_snapshot_roundtrip(
        self,
        tmp_storage: str,
        sample_graph: TaskGraph,
    ) -> None:
        """TaskGraph snapshot 应该可以完整恢复."""
        mgr = TaskLedgerManager(storage_dir=tmp_storage)
        ledger = mgr.create(
            goal="test graph snapshot",
            task_graph=sample_graph,
            budget=100_000,
        )
        task_id = ledger.task_id

        # 从保存的 ledger 中恢复图
        loaded = mgr.load(task_id)
        snapshot = loaded.task_graph_snapshot
        assert snapshot is not None
        assert "nodes" in snapshot
        assert len(snapshot["nodes"]) == 3

        # 重建 TaskGraph
        restored = TaskGraph()
        restored.original_goal = snapshot.get("original_goal", "")
        for _nid, ndata in snapshot["nodes"].items():
            restored.add_task(TaskNode(
                id=ndata["id"],
                description=ndata["description"],
                dependencies=ndata.get("dependencies", []),
                status=TaskStatus(ndata.get("status", "pending")),
                files_involved=ndata.get("files_involved", []),
                verification=ndata.get("verification", ""),
            ))

        assert len(restored.nodes) == 3
        assert "task-1" in restored.nodes
        assert "task-2" in restored.nodes
        assert "task-3" in restored.nodes
        assert restored.nodes["task-2"].dependencies == ["task-1"]
        assert restored.nodes["task-3"].dependencies == ["task-2"]

    def test_multiple_ledgers_isolated(self, tmp_storage: str) -> None:
        """多个 ledger 互不干扰."""
        mgr = TaskLedgerManager(storage_dir=tmp_storage)

        g1 = TaskGraph()
        g1.original_goal = "任务A"
        g1.add_task(TaskNode(id="a1", description="A-step-1"))

        g2 = TaskGraph()
        g2.original_goal = "任务B"
        g2.add_task(TaskNode(id="b1", description="B-step-1"))

        l1 = mgr.create(goal="任务A", task_graph=g1, budget=50_000)
        l2 = mgr.create(goal="任务B", task_graph=g2, budget=80_000)

        assert l1.task_id != l2.task_id
        metas = mgr.list_all()
        assert len(metas) == 2

        # 分别修改
        mgr.update_resources(l1, tokens=100, steps=1, wall_time=1.0)
        mgr.update_resources(l2, tokens=200, steps=2, wall_time=2.0)

        loaded1 = mgr.load(l1.task_id)
        loaded2 = mgr.load(l2.task_id)
        assert loaded1.total_tokens_used == 100
        assert loaded2.total_tokens_used == 200


# ──────────────────────────────────────────────────────────────
# Test: /status 和 /ledger 的 get_stats 输出
# ──────────────────────────────────────────────────────────────


class TestStatusAndLedger:
    """/status 和 /ledger 输出格式."""

    def test_get_stats_format(
        self, manager_and_ledger: tuple[TaskLedgerManager, TaskLedger]
    ) -> None:
        mgr, ledger = manager_and_ledger
        stats = mgr.get_stats(ledger)

        # 验证 /status 需要的所有字段
        required_keys = [
            "task_id", "goal", "status", "current_phase",
            "current_task_id", "completion_rate", "completed_tasks",
            "failed_attempts", "total_steps", "total_tokens_used",
            "token_budget", "token_budget_remaining",
            "avg_tokens_per_task", "total_wall_time_seconds",
            "issues_open", "issues_resolved", "decisions_count",
            "milestones_reached", "milestones_total",
        ]
        for key in required_keys:
            assert key in stats, f"Missing key in stats: {key}"

        assert stats["goal"] == "创建一个 hello.py 文件，打印 Hello World"
        assert stats["status"] == "NOT_STARTED"
        assert stats["completion_rate"] == "0/3"

    def test_get_stats_after_partial_completion(
        self, manager_and_ledger: tuple[TaskLedgerManager, TaskLedger]
    ) -> None:
        mgr, ledger = manager_and_ledger
        ledger.status = TaskRunStatus.RUNNING

        record = CompletedTaskRecord(
            task_id="task-1",
            artifact_id=str(uuid4()),
            description="步骤1",
            self_summary="已完成",
            files_changed=["hello.py"],
            verification_passed=True,
            confidence="DONE",
            step_number_start=0,
            step_number_end=3,
            token_count=500,
            timestamp=datetime.now(UTC),
        )
        ledger.completed_tasks.append(record)
        mgr.update_resources(ledger, tokens=500, steps=3, wall_time=2.0)

        stats = mgr.get_stats(ledger)
        assert stats["completion_rate"] == "1/3"
        assert stats["completed_tasks"] == 1
        assert stats["total_tokens_used"] == 500
        assert stats["total_steps"] == 3
        assert stats["avg_tokens_per_task"] == 500

    def test_ledger_full_json_roundtrip(
        self, manager_and_ledger: tuple[TaskLedgerManager, TaskLedger]
    ) -> None:
        """/ledger 输出的完整 JSON 应该可以被解析."""
        mgr, ledger = manager_and_ledger
        # to_dict 就是 /ledger 命令输出的数据
        data = ledger.to_dict()
        json_str = json.dumps(data, ensure_ascii=False, indent=2)
        parsed = json.loads(json_str)
        assert parsed["task_id"] == ledger.task_id
        assert parsed["goal"] == ledger.goal

    def test_history_log(
        self, manager_and_ledger: tuple[TaskLedgerManager, TaskLedger]
    ) -> None:
        """每次 save 都应该产生 history 记录."""
        mgr, ledger = manager_and_ledger
        # create 时已 save 了一次
        mgr.update_phase(ledger, "execution")  # save 一次
        mgr.update_resources(ledger, tokens=100, steps=1, wall_time=1.0)  # 又 save 一次

        history = mgr.get_history(ledger.task_id, last_n=10)
        # 至少 3 条：create + update_phase + update_resources
        assert len(history) >= 3
        # 每条应该有 timestamp 和关键指标
        for entry in history:
            assert "timestamp" in entry
            assert "status" in entry
            assert "current_phase" in entry
            assert "total_tokens_used" in entry
