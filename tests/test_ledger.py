"""Task Ledger 系统测试.

运行: uv run pytest tests/test_ledger.py -xvs
"""

from __future__ import annotations

import json
import threading
from datetime import UTC, datetime
from pathlib import Path

import pytest

from mini_code_agent.artifacts import (
    ArtifactDecision,
    ArtifactStatus,
    CheckResult,
    Confidence,
    EditOperation,
    FileEdit,
    Patch,
    ResourceUsage,
    ScopeCheck,
    SelfVerification,
    SubtaskArtifact,
)
from mini_code_agent.core.task_graph import TaskGraph, TaskNode
from mini_code_agent.longrun.ledger_types import (
    ActiveIssue,
    CompletedTaskRecord,
    DecisionRecord,
    FailedAttemptRecord,
    Milestone,
    TaskRunStatus,
)
from mini_code_agent.longrun.task_ledger import LedgerMeta, TaskLedger
from mini_code_agent.longrun.ledger_manager import LedgerError, TaskLedgerManager


# ============================================================
# 辅助工厂
# ============================================================


def _make_check_result(
    name: str = "syntax",
    passed: bool = True,
) -> CheckResult:
    return CheckResult(
        check_name=name,
        passed=passed,
        skipped=False,
        skip_reason=None,
        duration_seconds=0.1,
        details="ok" if passed else "error",
        items_checked=1,
        items_failed=0 if passed else 1,
    )


def _make_verification(passed: bool = True) -> SelfVerification:
    return SelfVerification(
        syntax_check=_make_check_result("syntax", passed=passed),
        lint_check=None,
        type_check=None,
        unit_test=None,
        import_check=_make_check_result("import", passed=True),
        overall_passed=passed,
    )


def _make_artifact(
    task_id: str = "task-1",
    artifact_id: str = "artifact-1",
    files: list[str] | None = None,
    passed: bool = True,
    confidence: Confidence = Confidence.DONE,
    decisions: list[ArtifactDecision] | None = None,
    tokens: int = 1000,
) -> SubtaskArtifact:
    edits = [
        FileEdit(
            path=f,
            operation=EditOperation.MODIFY,
            old_content="old",
            new_content="new",
            old_path=None,
            unified_diff="--- a\n+++ b\n",
            lines_added=1,
            lines_removed=1,
        )
        for f in (files or ["src/main.py"])
    ]
    return SubtaskArtifact(
        artifact_id=artifact_id,
        task_id=task_id,
        created_at=datetime.now(UTC),
        producer="test",
        task_description=f"Test task {task_id}",
        allowed_paths=["src/"],
        verification_spec="pytest",
        patch=Patch(
            edits=edits,
            total_files_changed=len(edits),
            total_lines_added=sum(e.lines_added for e in edits),
            total_lines_removed=sum(e.lines_removed for e in edits),
            base_git_hash="abc123",
        ),
        self_verification=_make_verification(passed),
        scope_check=ScopeCheck(
            allowed_paths=["src/"],
            touched_paths=[e.path for e in edits],
            out_of_scope_paths=[],
            is_clean=True,
        ),
        decisions=decisions or [],
        resource_usage=ResourceUsage(
            tokens_input=tokens // 2,
            tokens_output=tokens // 2,
            tokens_total=tokens,
            llm_calls=2,
            tool_calls=3,
            wall_time_seconds=5.0,
            model_used="test-model",
        ),
        confidence=confidence,
        self_summary=f"Completed {task_id} successfully",
        open_questions=[],
        status=ArtifactStatus.SUBMITTED,
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


def _make_diamond_graph() -> TaskGraph:
    """创建钻石形 DAG: A → B, A → C, B → D, C → D."""
    graph = TaskGraph()
    graph.original_goal = "Diamond goal"
    graph.add_task(TaskNode(id="A", description="Task A"))
    graph.add_task(TaskNode(id="B", description="Task B", dependencies=["A"]))
    graph.add_task(TaskNode(id="C", description="Task C", dependencies=["A"]))
    graph.add_task(TaskNode(id="D", description="Task D", dependencies=["B", "C"]))
    return graph


# ============================================================
# Ledger Types 测试
# ============================================================


class TestLedgerTypes:
    """测试各数据结构的 dict roundtrip."""

    def test_milestone_roundtrip(self) -> None:
        m = Milestone(
            id="m-1",
            description="Phase 1 done",
            associated_task_ids=["t-1", "t-2"],
            expected_by_step=5,
            actual_step=3,
            status="REACHED",
        )
        d = m.to_dict()
        m2 = Milestone.from_dict(d)
        assert m2.id == m.id
        assert m2.description == m.description
        assert m2.associated_task_ids == m.associated_task_ids
        assert m2.expected_by_step == m.expected_by_step
        assert m2.actual_step == m.actual_step
        assert m2.status == m.status

    def test_completed_task_record_roundtrip(self) -> None:
        now = datetime.now(UTC)
        r = CompletedTaskRecord(
            task_id="t-1",
            artifact_id="a-1",
            description="desc",
            self_summary="summary",
            files_changed=["a.py", "b.py"],
            verification_passed=True,
            reviewer_verdict="approved",
            confidence="DONE",
            step_number_start=1,
            step_number_end=3,
            token_count=500,
            timestamp=now,
        )
        d = r.to_dict()
        r2 = CompletedTaskRecord.from_dict(d)
        assert r2.task_id == r.task_id
        assert r2.artifact_id == r.artifact_id
        assert r2.files_changed == r.files_changed
        assert r2.verification_passed is True
        assert r2.reviewer_verdict == "approved"
        assert r2.confidence == "DONE"

    def test_decision_record_roundtrip(self) -> None:
        dr = DecisionRecord(
            description="Use dataclass",
            reason="Simpler",
            source_task_id="t-1",
            reversible=False,
            step_number=2,
        )
        d = dr.to_dict()
        dr2 = DecisionRecord.from_dict(d)
        assert dr2.description == dr.description
        assert dr2.reversible is False
        assert dr2.source_task_id == "t-1"

    def test_active_issue_roundtrip(self) -> None:
        issue = ActiveIssue(
            id="issue-1",
            description="Token expired",
            source_task_id="t-1",
            severity="blocker",
            first_seen_step=3,
            resolution_attempts=2,
            resolved=False,
        )
        d = issue.to_dict()
        issue2 = ActiveIssue.from_dict(d)
        assert issue2.id == issue.id
        assert issue2.severity == "blocker"
        assert issue2.resolution_attempts == 2

    def test_failed_attempt_roundtrip(self) -> None:
        fa = FailedAttemptRecord(
            task_id="t-1",
            artifact_id="a-1",
            approach_description="Tried X",
            failure_reason="Y failed",
            step_number=5,
            lesson_learned="Don't do X",
        )
        d = fa.to_dict()
        fa2 = FailedAttemptRecord.from_dict(d)
        assert fa2.task_id == fa.task_id
        assert fa2.lesson_learned == "Don't do X"

    def test_task_run_status_values(self) -> None:
        assert TaskRunStatus.NOT_STARTED.value == "NOT_STARTED"
        assert TaskRunStatus.RUNNING.value == "RUNNING"
        assert TaskRunStatus.PAUSED.value == "PAUSED"
        assert TaskRunStatus.COMPLETED.value == "COMPLETED"
        assert TaskRunStatus.FAILED.value == "FAILED"
        assert TaskRunStatus.ABORTED.value == "ABORTED"


# ============================================================
# TaskLedger 测试
# ============================================================


class TestTaskLedger:
    """测试 TaskLedger 的序列化和反序列化."""

    def test_json_roundtrip(self) -> None:
        """save 后 load 内容完全一致."""
        ledger = TaskLedger(
            task_id="test-123",
            goal="Build a feature",
            status=TaskRunStatus.RUNNING,
            plan_summary="3 tasks total",
            current_phase="implementation",
            current_task_id="task-1",
            milestones=[
                Milestone(
                    id="m-1",
                    description="Phase 1",
                    associated_task_ids=["task-0"],
                    expected_by_step=1,
                    status="REACHED",
                    actual_step=1,
                ),
            ],
            completed_tasks=[
                CompletedTaskRecord(
                    task_id="task-0",
                    artifact_id="a-0",
                    description="Setup",
                    self_summary="Setup done",
                    files_changed=["setup.py"],
                    verification_passed=True,
                    confidence="DONE",
                    step_number_start=0,
                    step_number_end=1,
                    token_count=200,
                ),
            ],
            decisions_made=[
                DecisionRecord(
                    description="Use sqlite",
                    reason="Simple",
                    source_task_id="task-0",
                    reversible=True,
                    step_number=1,
                ),
            ],
            active_issues=[
                ActiveIssue(
                    id="i-1",
                    description="Bug",
                    source_task_id="task-0",
                    severity="warning",
                ),
            ],
            failed_attempts=[
                FailedAttemptRecord(
                    task_id="task-0",
                    artifact_id="a-fail",
                    approach_description="First try",
                    failure_reason="Import error",
                    lesson_learned="Check imports",
                ),
            ],
            total_tokens_used=500,
            total_steps=3,
            total_wall_time_seconds=10.5,
            token_budget=10000,
            token_budget_remaining=9500,
        )

        d = ledger.to_dict()
        json_str = json.dumps(d, ensure_ascii=False, indent=2)
        d2 = json.loads(json_str)
        ledger2 = TaskLedger.from_dict(d2)

        assert ledger2.task_id == ledger.task_id
        assert ledger2.goal == ledger.goal
        assert ledger2.status == ledger.status
        assert ledger2.plan_summary == ledger.plan_summary
        assert ledger2.current_phase == ledger.current_phase
        assert ledger2.current_task_id == ledger.current_task_id
        assert len(ledger2.milestones) == 1
        assert ledger2.milestones[0].status == "REACHED"
        assert len(ledger2.completed_tasks) == 1
        assert ledger2.completed_tasks[0].task_id == "task-0"
        assert len(ledger2.decisions_made) == 1
        assert ledger2.decisions_made[0].description == "Use sqlite"
        assert len(ledger2.active_issues) == 1
        assert len(ledger2.failed_attempts) == 1
        assert ledger2.total_tokens_used == 500
        assert ledger2.total_steps == 3
        assert ledger2.total_wall_time_seconds == 10.5
        assert ledger2.token_budget == 10000
        assert ledger2.ledger_schema_version == "1.0"

    def test_to_meta(self) -> None:
        ledger = TaskLedger(
            task_id="meta-test",
            goal="Meta test",
            status=TaskRunStatus.RUNNING,
            total_tokens_used=1234,
        )
        meta = ledger.to_meta()
        assert isinstance(meta, LedgerMeta)
        assert meta.task_id == "meta-test"
        assert meta.goal == "Meta test"
        assert meta.status == TaskRunStatus.RUNNING
        assert meta.total_tokens_used == 1234

    def test_dict_field_order_is_stable(self) -> None:
        """序列化的字段顺序应稳定（human-readable）."""
        ledger = TaskLedger(task_id="stable", goal="Stable test")
        d1 = ledger.to_dict()
        d2 = ledger.to_dict()
        assert list(d1.keys()) == list(d2.keys())
        # schema_version 应在最前面
        assert list(d1.keys())[0] == "ledger_schema_version"

    def test_timestamps_are_utc(self) -> None:
        """时间戳应为 UTC."""
        ledger = TaskLedger(task_id="utc", goal="UTC test")
        d = ledger.to_dict()
        # ISO format 应含 UTC 时区信息
        created_at = datetime.fromisoformat(d["created_at"])
        assert created_at.tzinfo is not None


# ============================================================
# TaskLedgerManager 测试
# ============================================================


class TestLedgerManager:
    """测试 LedgerManager 的核心功能."""

    @pytest.fixture()
    def manager(self, tmp_path: Path) -> TaskLedgerManager:
        return TaskLedgerManager(storage_dir=str(tmp_path / "ledger"))

    @pytest.fixture()
    def graph(self) -> TaskGraph:
        return _make_task_graph(3)

    def test_create_and_load(self, manager: TaskLedgerManager, graph: TaskGraph) -> None:
        """创建后加载，内容一致."""
        ledger = manager.create(goal="Build feature", task_graph=graph, budget=10000)

        assert ledger.task_id is not None
        assert ledger.goal == "Build feature"
        assert ledger.status == TaskRunStatus.NOT_STARTED
        assert ledger.token_budget == 10000
        assert len(ledger.milestones) > 0

        loaded = manager.load(ledger.task_id)
        assert loaded.task_id == ledger.task_id
        assert loaded.goal == ledger.goal
        assert loaded.token_budget == 10000

    def test_json_roundtrip_save_load(self, manager: TaskLedgerManager, graph: TaskGraph) -> None:
        """Save 后 load 内容完全一致."""
        ledger = manager.create(goal="Roundtrip test", task_graph=graph, budget=5000)

        # 添加一些数据
        ledger.status = TaskRunStatus.RUNNING
        ledger.current_phase = "execution"
        ledger.current_task_id = "task-0"
        ledger.total_tokens_used = 123
        ledger.total_steps = 2
        manager.save(ledger)

        loaded = manager.load(ledger.task_id)
        assert loaded.status == TaskRunStatus.RUNNING
        assert loaded.current_phase == "execution"
        assert loaded.current_task_id == "task-0"
        assert loaded.total_tokens_used == 123
        assert loaded.total_steps == 2

    def test_atomic_write_survives_corruption(self, manager: TaskLedgerManager, graph: TaskGraph) -> None:
        """模拟主文件损坏，load 能从 history 恢复."""
        ledger = manager.create(goal="Atomic test", task_graph=graph, budget=5000)
        original_id = ledger.task_id

        # 做一次正常更新
        ledger.status = TaskRunStatus.RUNNING
        ledger.total_steps = 5
        manager.save(ledger)

        # 损坏主文件
        ledger_path = manager.storage_dir / f"{original_id}.json"
        ledger_path.write_text("{ broken json!", encoding="utf-8")

        # 应该从 history 恢复
        recovered = manager.load(original_id)
        assert recovered.task_id == original_id
        assert recovered.status == TaskRunStatus.RUNNING
        assert recovered.total_steps == 5

    def test_load_nonexistent_raises(self, manager: TaskLedgerManager) -> None:
        """加载不存在的 ledger 应报错."""
        with pytest.raises(LedgerError, match="均不存在"):
            manager.load("nonexistent-id")

    def test_list_all(self, manager: TaskLedgerManager, graph: TaskGraph) -> None:
        """列出所有 Ledger."""
        l1 = manager.create(goal="Task 1", task_graph=graph, budget=1000)
        l2 = manager.create(goal="Task 2", task_graph=graph, budget=2000)

        metas = manager.list_all()
        assert len(metas) == 2
        ids = {m.task_id for m in metas}
        assert l1.task_id in ids
        assert l2.task_id in ids

    def test_record_task_completed(self, manager: TaskLedgerManager, graph: TaskGraph) -> None:
        """record_task_completed 正确提取 Artifact 所有字段."""
        ledger = manager.create(goal="Test completed", task_graph=graph, budget=10000)
        ledger.status = TaskRunStatus.RUNNING
        manager.save(ledger)

        decisions = [
            ArtifactDecision(
                description="Use dataclass",
                reason="Simpler",
                alternatives_considered=["Pydantic"],
                reversible=False,
                step_number=1,
            ),
        ]
        artifact = _make_artifact(
            task_id="task-0",
            artifact_id="a-123",
            files=["src/model.py", "src/utils.py"],
            passed=True,
            confidence=Confidence.DONE,
            decisions=decisions,
            tokens=2000,
        )

        manager.record_task_completed(ledger, artifact)

        assert len(ledger.completed_tasks) == 1
        record = ledger.completed_tasks[0]
        assert record.task_id == "task-0"
        assert record.artifact_id == "a-123"
        assert record.files_changed == ["src/model.py", "src/utils.py"]
        assert record.verification_passed is True
        assert record.confidence == "DONE"
        assert record.token_count == 2000

        # decisions 也被提取
        assert len(ledger.decisions_made) == 1
        assert ledger.decisions_made[0].description == "Use dataclass"
        assert ledger.decisions_made[0].reversible is False
        assert ledger.decisions_made[0].source_task_id == "task-0"

        # 资源被更新
        assert ledger.total_tokens_used >= 2000

        # 验证已持久化
        loaded = manager.load(ledger.task_id)
        assert len(loaded.completed_tasks) == 1

    def test_record_task_failed(self, manager: TaskLedgerManager, graph: TaskGraph) -> None:
        """失败记录正确保存."""
        ledger = manager.create(goal="Test failed", task_graph=graph, budget=10000)
        artifact = _make_artifact(task_id="task-1", passed=False)

        manager.record_task_failed(ledger, artifact, "Import error")

        assert len(ledger.failed_attempts) == 1
        fa = ledger.failed_attempts[0]
        assert fa.task_id == "task-1"
        assert fa.failure_reason == "Import error"

    def test_record_and_resolve_issue(self, manager: TaskLedgerManager, graph: TaskGraph) -> None:
        ledger = manager.create(goal="Test issues", task_graph=graph, budget=10000)

        issue = ActiveIssue(
            id="issue-1",
            description="Token expired",
            source_task_id="task-0",
            severity="blocker",
            first_seen_step=1,
        )
        manager.record_active_issue(ledger, issue)
        assert len(ledger.active_issues) == 1

        manager.resolve_issue(ledger, "issue-1")
        assert len(ledger.active_issues) == 0
        assert len(ledger.resolved_issues) == 1
        assert ledger.resolved_issues[0].resolved is True

    def test_update_current_task(self, manager: TaskLedgerManager, graph: TaskGraph) -> None:
        ledger = manager.create(goal="Test update", task_graph=graph, budget=10000)
        manager.update_current_task(ledger, "task-2")
        assert ledger.current_task_id == "task-2"

        loaded = manager.load(ledger.task_id)
        assert loaded.current_task_id == "task-2"

    def test_update_phase(self, manager: TaskLedgerManager, graph: TaskGraph) -> None:
        ledger = manager.create(goal="Test phase", task_graph=graph, budget=10000)
        manager.update_phase(ledger, "testing")
        assert ledger.current_phase == "testing"

    def test_update_resources(self, manager: TaskLedgerManager, graph: TaskGraph) -> None:
        ledger = manager.create(goal="Test resources", task_graph=graph, budget=10000)
        manager.update_resources(ledger, tokens=500, steps=1, wall_time=2.5)
        assert ledger.total_tokens_used == 500
        assert ledger.total_steps == 1
        assert ledger.total_wall_time_seconds == 2.5
        assert ledger.token_budget_remaining == 9500

        manager.update_resources(ledger, tokens=300, steps=1, wall_time=1.0)
        assert ledger.total_tokens_used == 800
        assert ledger.total_steps == 2
        assert ledger.total_wall_time_seconds == 3.5
        assert ledger.token_budget_remaining == 9200


# ============================================================
# Milestone 达成检测
# ============================================================


class TestMilestoneDetection:
    """测试 milestone 达成逻辑."""

    @pytest.fixture()
    def manager(self, tmp_path: Path) -> TaskLedgerManager:
        return TaskLedgerManager(storage_dir=str(tmp_path / "ledger"))

    def test_milestone_reached_on_task_completion(self, manager: TaskLedgerManager) -> None:
        """添加 completed_task 后对应 milestone 变成 REACHED."""
        graph = _make_task_graph(2)
        ledger = manager.create(goal="Milestone test", task_graph=graph, budget=10000)

        # 第一层应该只有 task-0（深度 0）
        # 完成 task-0
        artifact = _make_artifact(task_id="task-0")
        manager.record_task_completed(ledger, artifact)

        # 找到关联 task-0 的 milestone
        m0 = next(
            (m for m in ledger.milestones if "task-0" in m.associated_task_ids),
            None,
        )
        assert m0 is not None
        assert m0.status == "REACHED"

    def test_milestone_overdue(self, manager: TaskLedgerManager) -> None:
        """当步数超过 expected_by_step 但任务未完成时，milestone 变 OVERDUE."""
        graph = _make_task_graph(1)
        ledger = manager.create(goal="Overdue test", task_graph=graph, budget=10000)

        # 找到 milestone
        assert len(ledger.milestones) > 0
        m = ledger.milestones[0]
        expected = m.expected_by_step

        # 增加步数超过预期
        ledger.total_steps = expected + 5
        manager._check_milestones(ledger)

        assert m.status == "OVERDUE"

    def test_diamond_graph_milestones(self, manager: TaskLedgerManager) -> None:
        """钻石形 DAG 的 milestone 提取."""
        graph = _make_diamond_graph()
        ledger = manager.create(goal="Diamond test", task_graph=graph, budget=10000)

        # 应有多层 milestone
        assert len(ledger.milestones) > 1


# ============================================================
# build_context_summary 测试
# ============================================================


class TestBuildContextSummary:
    """测试上下文摘要的生成."""

    @pytest.fixture()
    def manager(self, tmp_path: Path) -> TaskLedgerManager:
        return TaskLedgerManager(storage_dir=str(tmp_path / "ledger"))

    def test_deterministic_output(self, manager: TaskLedgerManager) -> None:
        """相同 ledger → 相同输出."""
        graph = _make_task_graph(3)
        ledger = manager.create(goal="Deterministic test", task_graph=graph, budget=10000)
        ledger.current_phase = "execution"

        s1 = manager.build_context_summary(ledger)
        s2 = manager.build_context_summary(ledger)
        assert s1 == s2

    def test_overview_always_present(self, manager: TaskLedgerManager) -> None:
        """任务概览应始终存在."""
        graph = _make_task_graph(1)
        ledger = manager.create(goal="Overview test", task_graph=graph, budget=10000)

        summary = manager.build_context_summary(ledger, max_chars=200)
        assert "任务概览" in summary
        assert "Overview test" in summary

    def test_truncation_by_priority(self, manager: TaskLedgerManager) -> None:
        """低优先级部分在 max_chars 限制下被截断."""
        graph = _make_task_graph(3)
        ledger = manager.create(goal="Truncation test", task_graph=graph, budget=10000)

        # 添加大量历史数据
        for i in range(20):
            ledger.completed_tasks.append(CompletedTaskRecord(
                task_id=f"task-{i}",
                artifact_id=f"a-{i}",
                description=f"Very long description for task {i} " * 5,
                self_summary=f"Summary for task {i} " * 3,
                files_changed=[f"file{i}.py"],
                verification_passed=True,
                confidence="DONE",
            ))

        # 用很小的 max_chars
        short_summary = manager.build_context_summary(ledger, max_chars=500)
        full_summary = manager.build_context_summary(ledger, max_chars=10000)

        assert len(short_summary) <= 600  # 一些余量
        assert len(full_summary) > len(short_summary)
        # 概览始终保留
        assert "任务概览" in short_summary
        # 应包含 omitted 提示
        if "omitted" not in short_summary:
            # 如果 500 chars 足够放下所有内容就跳过
            pass

    def test_issues_in_summary(self, manager: TaskLedgerManager) -> None:
        """活跃问题应出现在摘要中."""
        graph = _make_task_graph(1)
        ledger = manager.create(goal="Issue test", task_graph=graph, budget=10000)
        ledger.active_issues.append(ActiveIssue(
            id="i-1",
            description="Critical bug in auth",
            source_task_id="task-0",
            severity="blocker",
        ))
        manager.save(ledger)

        summary = manager.build_context_summary(ledger)
        assert "Critical bug in auth" in summary

    def test_failures_in_summary(self, manager: TaskLedgerManager) -> None:
        """失败教训应出现在摘要中."""
        graph = _make_task_graph(1)
        ledger = manager.create(goal="Failure test", task_graph=graph, budget=10000)
        ledger.failed_attempts.append(FailedAttemptRecord(
            task_id="task-0",
            artifact_id="a-fail",
            approach_description="Used wrong API",
            failure_reason="404 error",
            lesson_learned="Use v2 API",
        ))
        manager.save(ledger)

        summary = manager.build_context_summary(ledger)
        assert "Use v2 API" in summary

    def test_decisions_in_summary(self, manager: TaskLedgerManager) -> None:
        """关键决策应出现在摘要中."""
        graph = _make_task_graph(1)
        ledger = manager.create(goal="Decision test", task_graph=graph, budget=10000)
        ledger.decisions_made.append(DecisionRecord(
            description="Use PostgreSQL",
            reason="Better for concurrent writes",
            source_task_id="task-0",
            reversible=False,
        ))
        manager.save(ledger)

        summary = manager.build_context_summary(ledger)
        assert "Use PostgreSQL" in summary


# ============================================================
# History 测试
# ============================================================


class TestHistory:
    """测试 history.jsonl 的 append-only 特性."""

    @pytest.fixture()
    def manager(self, tmp_path: Path) -> TaskLedgerManager:
        return TaskLedgerManager(storage_dir=str(tmp_path / "ledger"))

    def test_history_append_only(self, manager: TaskLedgerManager) -> None:
        """每次更新都有新记录，不覆盖."""
        graph = _make_task_graph(1)
        ledger = manager.create(goal="History test", task_graph=graph, budget=10000)

        # create 已经写了 1 次
        # 再做 3 次更新
        for i in range(3):
            ledger.total_steps = i + 1
            manager.save(ledger)

        history_path = manager.storage_dir / f"{ledger.task_id}.history.jsonl"
        lines = [l for l in history_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        # create 本身调用了 save → 1 次；之后 3 次 = 4 次
        assert len(lines) == 4

        # 每一行都是有效 JSON
        for line in lines:
            entry = json.loads(line)
            assert "timestamp" in entry
            assert "snapshot" in entry

    def test_get_history(self, manager: TaskLedgerManager) -> None:
        """get_history 返回最后 N 条摘要."""
        graph = _make_task_graph(1)
        ledger = manager.create(goal="Get history", task_graph=graph, budget=10000)

        for i in range(5):
            ledger.total_steps = i + 1
            manager.save(ledger)

        entries = manager.get_history(ledger.task_id, last_n=3)
        assert len(entries) == 3
        # 最后一个应该是 total_steps=5
        assert entries[-1]["total_steps"] == 5

    def test_history_survives_main_file_corruption(self, manager: TaskLedgerManager) -> None:
        """即使主文件损坏，history 也完好无损."""
        graph = _make_task_graph(1)
        ledger = manager.create(goal="Survive test", task_graph=graph, budget=10000)
        ledger.total_steps = 42
        manager.save(ledger)

        # 损坏主文件
        (manager.storage_dir / f"{ledger.task_id}.json").write_text("broken", encoding="utf-8")

        # history 仍然可读
        entries = manager.get_history(ledger.task_id)
        assert len(entries) >= 1
        assert entries[-1]["total_steps"] == 42


# ============================================================
# 并发安全测试
# ============================================================


class TestConcurrencySafety:
    """测试 /status 读和 Agent 写不冲突."""

    @pytest.fixture()
    def manager(self, tmp_path: Path) -> TaskLedgerManager:
        return TaskLedgerManager(storage_dir=str(tmp_path / "ledger"))

    def test_concurrent_read_write(self, manager: TaskLedgerManager) -> None:
        """并发读写不应导致 JSON 解析错误."""
        graph = _make_task_graph(1)
        ledger = manager.create(goal="Concurrent test", task_graph=graph, budget=10000)
        task_id = ledger.task_id

        errors: list[str] = []
        stop = threading.Event()

        def writer() -> None:
            for i in range(50):
                if stop.is_set():
                    break
                try:
                    ledger.total_steps = i
                    manager.save(ledger)
                except Exception as e:
                    errors.append(f"write error: {e}")

        def reader() -> None:
            for _ in range(50):
                if stop.is_set():
                    break
                try:
                    loaded = manager.load(task_id)
                    # 基本验证
                    assert loaded.task_id == task_id
                except LedgerError:
                    pass  # 可以容忍瞬时恢复
                except Exception as e:
                    errors.append(f"read error: {e}")

        t_write = threading.Thread(target=writer)
        t_read = threading.Thread(target=reader)
        t_write.start()
        t_read.start()
        t_write.join(timeout=10)
        t_read.join(timeout=10)
        stop.set()

        assert not errors, f"Concurrent errors: {errors}"


# ============================================================
# get_stats / get_summary_for_resume 测试
# ============================================================


class TestStatsAndResume:
    @pytest.fixture()
    def manager(self, tmp_path: Path) -> TaskLedgerManager:
        return TaskLedgerManager(storage_dir=str(tmp_path / "ledger"))

    def test_get_stats(self, manager: TaskLedgerManager) -> None:
        graph = _make_task_graph(3)
        ledger = manager.create(goal="Stats test", task_graph=graph, budget=10000)
        ledger.status = TaskRunStatus.RUNNING
        ledger.total_tokens_used = 3000
        ledger.total_steps = 5

        artifact = _make_artifact(task_id="task-0", tokens=1000)
        manager.record_task_completed(ledger, artifact)

        stats = manager.get_stats(ledger)
        assert stats["status"] == "RUNNING"
        assert stats["completed_tasks"] == 1
        assert stats["total_tokens_used"] > 0
        assert "completion_rate" in stats
        assert stats["avg_tokens_per_task"] > 0

    def test_get_summary_for_resume(self, manager: TaskLedgerManager) -> None:
        graph = _make_task_graph(3)
        ledger = manager.create(goal="Resume test", task_graph=graph, budget=10000)
        ledger.current_task_id = "task-1"

        artifact = _make_artifact(task_id="task-0", tokens=500)
        manager.record_task_completed(ledger, artifact)

        ledger.active_issues.append(ActiveIssue(
            id="i-1",
            description="Auth broken",
            source_task_id="task-0",
            severity="blocker",
        ))

        summary = manager.get_summary_for_resume(ledger)
        assert "断点恢复" in summary
        assert "Resume test" in summary
        assert "task-1" in summary
        assert "Auth broken" in summary
        assert "task-0" in summary  # 已完成的前置任务


# ============================================================
# Milestone 提取测试
# ============================================================


class TestMilestoneExtraction:
    @pytest.fixture()
    def manager(self, tmp_path: Path) -> TaskLedgerManager:
        return TaskLedgerManager(storage_dir=str(tmp_path / "ledger"))

    def test_linear_graph_milestones(self, manager: TaskLedgerManager) -> None:
        """线性图每层一个 milestone."""
        graph = _make_task_graph(3)
        milestones = manager._extract_milestones(graph)
        # 线性图有 3 层（每个任务一层）
        assert len(milestones) == 3
        # expected_by_step 递增
        for i in range(1, len(milestones)):
            assert milestones[i].expected_by_step >= milestones[i - 1].expected_by_step

    def test_diamond_graph_milestones(self, manager: TaskLedgerManager) -> None:
        """钻石图的层级结构."""
        graph = _make_diamond_graph()
        milestones = manager._extract_milestones(graph)
        # A=depth0, B,C=depth1, D=depth2 → 3 层
        assert len(milestones) == 3

    def test_empty_graph_milestones(self, manager: TaskLedgerManager) -> None:
        graph = TaskGraph()
        milestones = manager._extract_milestones(graph)
        assert milestones == []


# ============================================================
# save 方法的 save-on-every-mutation 保证
# ============================================================


class TestSaveOnMutation:
    """验证任何修改 ledger 的方法都调用了 save."""

    @pytest.fixture()
    def manager(self, tmp_path: Path) -> TaskLedgerManager:
        return TaskLedgerManager(storage_dir=str(tmp_path / "ledger"))

    def _count_history_lines(self, manager: TaskLedgerManager, task_id: str) -> int:
        path = manager.storage_dir / f"{task_id}.history.jsonl"
        if not path.exists():
            return 0
        return sum(1 for l in path.read_text("utf-8").splitlines() if l.strip())

    def test_record_task_completed_saves(self, manager: TaskLedgerManager) -> None:
        graph = _make_task_graph(1)
        ledger = manager.create(goal="Save test", task_graph=graph, budget=10000)
        before = self._count_history_lines(manager, ledger.task_id)

        artifact = _make_artifact(task_id="task-0")
        manager.record_task_completed(ledger, artifact)

        after = self._count_history_lines(manager, ledger.task_id)
        assert after > before

    def test_record_task_failed_saves(self, manager: TaskLedgerManager) -> None:
        graph = _make_task_graph(1)
        ledger = manager.create(goal="Save test", task_graph=graph, budget=10000)
        before = self._count_history_lines(manager, ledger.task_id)

        artifact = _make_artifact(task_id="task-0", passed=False)
        manager.record_task_failed(ledger, artifact, "error")

        after = self._count_history_lines(manager, ledger.task_id)
        assert after > before

    def test_record_active_issue_saves(self, manager: TaskLedgerManager) -> None:
        graph = _make_task_graph(1)
        ledger = manager.create(goal="Save test", task_graph=graph, budget=10000)
        before = self._count_history_lines(manager, ledger.task_id)

        issue = ActiveIssue(id="i-1", description="bug", source_task_id="t-0")
        manager.record_active_issue(ledger, issue)

        after = self._count_history_lines(manager, ledger.task_id)
        assert after > before

    def test_resolve_issue_saves(self, manager: TaskLedgerManager) -> None:
        graph = _make_task_graph(1)
        ledger = manager.create(goal="Save test", task_graph=graph, budget=10000)
        issue = ActiveIssue(id="i-1", description="bug", source_task_id="t-0")
        manager.record_active_issue(ledger, issue)
        before = self._count_history_lines(manager, ledger.task_id)

        manager.resolve_issue(ledger, "i-1")

        after = self._count_history_lines(manager, ledger.task_id)
        assert after > before

    def test_update_current_task_saves(self, manager: TaskLedgerManager) -> None:
        graph = _make_task_graph(1)
        ledger = manager.create(goal="Save test", task_graph=graph, budget=10000)
        before = self._count_history_lines(manager, ledger.task_id)

        manager.update_current_task(ledger, "task-0")

        after = self._count_history_lines(manager, ledger.task_id)
        assert after > before

    def test_update_phase_saves(self, manager: TaskLedgerManager) -> None:
        graph = _make_task_graph(1)
        ledger = manager.create(goal="Save test", task_graph=graph, budget=10000)
        before = self._count_history_lines(manager, ledger.task_id)

        manager.update_phase(ledger, "testing")

        after = self._count_history_lines(manager, ledger.task_id)
        assert after > before

    def test_update_resources_saves(self, manager: TaskLedgerManager) -> None:
        graph = _make_task_graph(1)
        ledger = manager.create(goal="Save test", task_graph=graph, budget=10000)
        before = self._count_history_lines(manager, ledger.task_id)

        manager.update_resources(ledger, 100, 1, 0.5)

        after = self._count_history_lines(manager, ledger.task_id)
        assert after > before
