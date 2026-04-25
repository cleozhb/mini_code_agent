"""TaskLedgerManager — Ledger 的持久化、增量更新与上下文摘要生成."""

from __future__ import annotations

import fcntl
import json
import logging
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from .ledger_types import (
    ActiveIssue,
    CompletedTaskRecord,
    DecisionRecord,
    FailedAttemptRecord,
    Milestone,
    TaskRunStatus,
)
from .task_ledger import LedgerMeta, TaskLedger

if TYPE_CHECKING:
    from ..artifacts.artifact import SubtaskArtifact
    from ..core.task_graph import TaskGraph

logger = logging.getLogger(__name__)


class LedgerError(Exception):
    """Ledger 操作错误."""


class TaskLedgerManager:
    """Ledger 的管理器：负责持久化、增量更新和上下文摘要.

    文件布局:
        {storage_dir}/
            {task_id}.json              # 当前 Ledger
            {task_id}.history.jsonl     # 每次更新的 append-only 历史
    """

    def __init__(self, storage_dir: str = ".agent/ledger/") -> None:
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    # ─────────────────────────────────────────────────────────────────
    # 创建 / 保存 / 加载
    # ─────────────────────────────────────────────────────────────────

    def create(
        self,
        goal: str,
        task_graph: TaskGraph,
        budget: int,
    ) -> TaskLedger:
        """初始化一个新 Ledger.

        从 task_graph 提取 milestones（每个 phase 对应一个里程碑）。
        """
        task_id = str(uuid.uuid4())
        now = datetime.now(UTC)

        # 从 task_graph 提取 milestones：为每个独立的 task 分组创建
        milestones = self._extract_milestones(task_graph)

        # 序列化 task_graph
        graph_snapshot = self._serialize_task_graph(task_graph)

        ledger = TaskLedger(
            task_id=task_id,
            goal=goal,
            created_at=now,
            updated_at=now,
            status=TaskRunStatus.NOT_STARTED,
            plan_summary=task_graph.original_goal or goal,
            task_graph_snapshot=graph_snapshot,
            current_phase="planning",
            milestones=milestones,
            token_budget=budget,
            token_budget_remaining=budget,
        )

        self.save(ledger)
        return ledger

    def save(self, ledger: TaskLedger) -> None:
        """原子写入 Ledger 到磁盘.

        先写 .tmp 文件然后 rename，避免写入一半崩溃导致文件损坏。
        同时 append 一条记录到 history.jsonl。
        """
        ledger.updated_at = datetime.now(UTC)

        ledger_path = self.storage_dir / f"{ledger.task_id}.json"
        tmp_path = self.storage_dir / f"{ledger.task_id}.json.tmp"
        history_path = self.storage_dir / f"{ledger.task_id}.history.jsonl"

        data = ledger.to_dict()
        json_str = json.dumps(data, ensure_ascii=False, indent=2, sort_keys=False)

        # 原子写入：write → fsync → rename
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(json_str)
            f.flush()
            os.fsync(f.fileno())
        os.replace(str(tmp_path), str(ledger_path))

        # Append-only 历史
        history_entry = {
            "timestamp": ledger.updated_at.isoformat(),
            "snapshot": data,
        }
        with open(history_path, "a", encoding="utf-8") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                f.write(json.dumps(history_entry, ensure_ascii=False) + "\n")
                f.flush()
                os.fsync(f.fileno())
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)

    def load(self, task_id: str) -> TaskLedger:
        """从磁盘加载 Ledger.

        主文件加载失败时，尝试从最近一条 history 记录恢复。
        """
        ledger_path = self.storage_dir / f"{task_id}.json"

        # 尝试主文件
        if ledger_path.exists():
            try:
                with open(ledger_path, "r", encoding="utf-8") as f:
                    fcntl.flock(f, fcntl.LOCK_SH)
                    try:
                        data = json.load(f)
                    finally:
                        fcntl.flock(f, fcntl.LOCK_UN)
                return TaskLedger.from_dict(data)
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning("主文件损坏，尝试从历史恢复: %s", e)

        # 从 history 恢复
        return self._recover_from_history(task_id)

    def list_all(self) -> list[LedgerMeta]:
        """列出所有存在的 Ledger（用于 /resume --list）."""
        metas: list[LedgerMeta] = []
        for path in sorted(self.storage_dir.glob("*.json")):
            if path.name.endswith(".json.tmp"):
                continue
            if ".history." in path.name:
                continue
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                ledger = TaskLedger.from_dict(data)
                metas.append(ledger.to_meta())
            except Exception as e:
                logger.warning("无法读取 Ledger %s: %s", path.name, e)
        return metas

    # ─────────────────────────────────────────────────────────────────
    # 增量更新方法（被 Agent / GraphExecutor 调用的主接口）
    # ─────────────────────────────────────────────────────────────────

    def record_task_completed(
        self,
        ledger: TaskLedger,
        artifact: SubtaskArtifact,
    ) -> None:
        """从 Artifact 提取信息填入 CompletedTaskRecord.

        同时：
        - 从 artifact.decisions 提取 DecisionRecord 追加到 decisions_made
        - 更新资源消耗计数
        - 检查是否有 milestone 达成
        """
        # 提取 files_changed
        files_changed = [e.path for e in artifact.patch.edits]

        record = CompletedTaskRecord(
            task_id=artifact.task_id,
            artifact_id=artifact.artifact_id,
            description=artifact.task_description,
            self_summary=artifact.self_summary,
            files_changed=files_changed,
            verification_passed=artifact.self_verification.overall_passed,
            confidence=artifact.confidence.value,
            step_number_start=ledger.total_steps,
            step_number_end=ledger.total_steps,
            token_count=artifact.resource_usage.tokens_total,
            timestamp=datetime.now(UTC),
        )
        ledger.completed_tasks.append(record)

        # 提取 decisions
        for ad in artifact.decisions:
            dr = DecisionRecord(
                description=ad.description,
                reason=ad.reason,
                source_task_id=artifact.task_id,
                reversible=ad.reversible,
                step_number=ad.step_number,
            )
            ledger.decisions_made.append(dr)

        # 更新资源
        ledger.total_tokens_used += artifact.resource_usage.tokens_total
        ledger.token_budget_remaining = max(
            0, ledger.token_budget - ledger.total_tokens_used
        )

        # 检查 milestone 达成
        self._check_milestones(ledger)

        self.save(ledger)

    def record_task_failed(
        self,
        ledger: TaskLedger,
        artifact: SubtaskArtifact,
        failure_reason: str,
    ) -> None:
        """记录失败尝试."""
        record = FailedAttemptRecord(
            task_id=artifact.task_id,
            artifact_id=artifact.artifact_id,
            approach_description=artifact.self_summary,
            failure_reason=failure_reason,
            step_number=ledger.total_steps,
            lesson_learned=None,
        )
        ledger.failed_attempts.append(record)
        self.save(ledger)

    def record_active_issue(self, ledger: TaskLedger, issue: ActiveIssue) -> None:
        """记录一个活跃问题."""
        ledger.active_issues.append(issue)
        self.save(ledger)

    def resolve_issue(self, ledger: TaskLedger, issue_id: str) -> None:
        """解决一个活跃问题."""
        for i, issue in enumerate(ledger.active_issues):
            if issue.id == issue_id:
                issue.resolved = True
                ledger.resolved_issues.append(issue)
                ledger.active_issues.pop(i)
                break
        self.save(ledger)

    def update_current_task(self, ledger: TaskLedger, task_id: str) -> None:
        """更新当前正在执行的子任务."""
        ledger.current_task_id = task_id
        self.save(ledger)

    def update_phase(self, ledger: TaskLedger, phase_name: str) -> None:
        """更新当前阶段."""
        ledger.current_phase = phase_name
        self.save(ledger)

    def update_resources(
        self,
        ledger: TaskLedger,
        tokens: int,
        steps: int,
        wall_time: float,
    ) -> None:
        """更新资源消耗计数."""
        ledger.total_tokens_used += tokens
        ledger.total_steps += steps
        ledger.total_wall_time_seconds += wall_time
        ledger.token_budget_remaining = max(
            0, ledger.token_budget - ledger.total_tokens_used
        )
        self.save(ledger)

    # ─────────────────────────────────────────────────────────────────
    # 上下文摘要
    # ─────────────────────────────────────────────────────────────────

    def build_context_summary(self, ledger: TaskLedger, max_chars: int = 4000) -> str:
        """生成注入 prompt 的上下文摘要（确定性输出）.

        按优先级排列，超出 max_chars 时截断低优先级部分：
        1. 任务概览（永远保留）
        2. 当前问题（高优先级）
        3. 失败教训（高优先级）
        4. 关键决策（中优先级）
        5. 最近完成的子任务（中优先级）
        6. 历史完成列表（低优先级）
        """
        sections: list[tuple[str, int]] = []  # (content, priority) 数字越小越高

        # 1. 任务概览（永远保留, priority=0）
        total_tasks = len(ledger.task_graph_snapshot.get("nodes", {}))
        completed_count = len(ledger.completed_tasks)
        overview = (
            f"## 任务概览\n"
            f"目标：{ledger.goal}\n"
            f"阶段：{ledger.current_phase}（{completed_count}/{total_tasks} 子任务完成）\n"
            f"资源：已用 {ledger.total_tokens_used:,}/{ledger.token_budget:,} token，"
            f"{ledger.total_steps} 步"
        )
        if ledger.current_task_id:
            overview += f"\n当前子任务：{ledger.current_task_id}"
        sections.append((overview, 0))

        # 2. 当前问题（priority=1）
        if ledger.active_issues:
            lines = ["## 当前问题"]
            for issue in ledger.active_issues:
                lines.append(
                    f"- [{issue.severity}] {issue.description}"
                    f"（来自 {issue.source_task_id}，尝试 {issue.resolution_attempts} 次）"
                )
            sections.append(("\n".join(lines), 1))

        # 3. 失败教训（priority=1）
        recent_failures = ledger.failed_attempts[-5:]
        if recent_failures:
            lines = ["## 失败教训"]
            for fa in recent_failures:
                lesson = fa.lesson_learned or fa.failure_reason
                lines.append(f"- {fa.task_id}: {lesson}")
            sections.append(("\n".join(lines), 1))

        # 4. 关键决策（priority=2，不可逆优先 + 时间倒序）
        if ledger.decisions_made:
            sorted_decisions = sorted(
                ledger.decisions_made,
                key=lambda d: (d.reversible, -d.step_number),
            )
            lines = ["## 关键决策"]
            for dr in sorted_decisions[:10]:
                rev = "可逆" if dr.reversible else "不可逆"
                lines.append(f"- [{rev}] {dr.description}（{dr.reason}）")
            sections.append(("\n".join(lines), 2))

        # 5. 最近完成的子任务（priority=2）
        recent_tasks = ledger.completed_tasks[-5:]
        if recent_tasks:
            lines = ["## 最近完成"]
            for ct in reversed(recent_tasks):
                lines.append(f"- {ct.task_id}: {ct.self_summary}")
            sections.append(("\n".join(lines), 2))

        # 6. 历史完成列表（priority=3）
        older_tasks = ledger.completed_tasks[:-5] if len(ledger.completed_tasks) > 5 else []
        if older_tasks:
            lines = ["## 历史完成"]
            for ct in older_tasks:
                lines.append(f"- {ct.task_id}: {ct.description}")
            sections.append(("\n".join(lines), 3))

        # 按优先级组装，超出 max_chars 时从低优先级截断
        sections.sort(key=lambda s: s[1])
        result_parts: list[str] = []
        used_chars = 0

        for content, priority in sections:
            if used_chars + len(content) + 2 <= max_chars:
                result_parts.append(content)
                used_chars += len(content) + 2  # +2 for "\n\n"
            else:
                remaining = max_chars - used_chars - 2
                if remaining > 50 and priority <= 1:
                    # 高优先级部分至少保留截断版
                    lines = content.splitlines()
                    truncated: list[str] = []
                    char_count = 0
                    for line in lines:
                        if char_count + len(line) + 1 > remaining - 30:
                            break
                        truncated.append(line)
                        char_count += len(line) + 1
                    omitted = len(lines) - len(truncated)
                    if omitted > 0:
                        truncated.append(f"...({omitted} items omitted)")
                    result_parts.append("\n".join(truncated))
                    used_chars += char_count + 30
                else:
                    # 低优先级整节跳过
                    count = content.count("\n")
                    result_parts.append(f"...({count} items omitted)")
                    used_chars += 30
                break

        return "\n\n".join(result_parts)

    def get_summary_for_resume(self, ledger: TaskLedger) -> str:
        """断点恢复时的详细 prompt（比 build_context_summary 更详细）."""
        lines = [
            f"# 断点恢复 — 任务 {ledger.task_id}",
            f"目标：{ledger.goal}",
            f"状态：{ledger.status.value}",
            f"阶段：{ledger.current_phase}",
            f"已完成 {len(ledger.completed_tasks)} 个子任务",
            f"已用 {ledger.total_tokens_used:,} token（预算 {ledger.token_budget:,}）",
        ]

        if ledger.current_task_id:
            lines.append(f"\n## 断点位置")
            lines.append(f"正在执行的子任务：{ledger.current_task_id}")

            # 列出前置依赖的结果
            dep_results = []
            for ct in ledger.completed_tasks:
                dep_results.append(f"- {ct.task_id}: {ct.self_summary}")
            if dep_results:
                lines.append("\n已完成的前置任务：")
                lines.extend(dep_results)

        # 活跃问题
        if ledger.active_issues:
            lines.append("\n## 需要注意的问题")
            for issue in ledger.active_issues:
                lines.append(f"- [{issue.severity}] {issue.description}")

        # 失败教训
        if ledger.failed_attempts:
            lines.append("\n## 之前的失败教训")
            for fa in ledger.failed_attempts[-5:]:
                lesson = fa.lesson_learned or fa.failure_reason
                lines.append(f"- {fa.task_id}: {lesson}")

        return "\n".join(lines)

    def get_stats(self, ledger: TaskLedger) -> dict:
        """纯指标数据（用于 /status 命令和 eval）."""
        total_tasks = len(ledger.task_graph_snapshot.get("nodes", {}))
        completed = len(ledger.completed_tasks)
        failed = len(ledger.failed_attempts)

        avg_tokens = (
            ledger.total_tokens_used // completed if completed > 0 else 0
        )

        return {
            "task_id": ledger.task_id,
            "goal": ledger.goal,
            "status": ledger.status.value,
            "current_phase": ledger.current_phase,
            "current_task_id": ledger.current_task_id,
            "completion_rate": f"{completed}/{total_tasks}" if total_tasks > 0 else "0/0",
            "completed_tasks": completed,
            "failed_attempts": failed,
            "total_steps": ledger.total_steps,
            "total_tokens_used": ledger.total_tokens_used,
            "token_budget": ledger.token_budget,
            "token_budget_remaining": ledger.token_budget_remaining,
            "avg_tokens_per_task": avg_tokens,
            "total_wall_time_seconds": ledger.total_wall_time_seconds,
            "issues_open": len(ledger.active_issues),
            "issues_resolved": len(ledger.resolved_issues),
            "decisions_count": len(ledger.decisions_made),
            "milestones_reached": sum(
                1 for m in ledger.milestones if m.status == "REACHED"
            ),
            "milestones_total": len(ledger.milestones),
        }

    # ─────────────────────────────────────────────────────────────────
    # 内部辅助
    # ─────────────────────────────────────────────────────────────────

    def _extract_milestones(self, task_graph: TaskGraph) -> list[Milestone]:
        """从 TaskGraph 提取 milestones.

        策略：按拓扑层级分组，每组对应一个里程碑。
        """
        if not task_graph.nodes:
            return []

        # 按拓扑深度分层
        depths: dict[str, int] = {}
        for node_id in task_graph.nodes:
            self._compute_depth(node_id, task_graph, depths)

        # 按深度分组
        layers: dict[int, list[str]] = {}
        for node_id, depth in sorted(depths.items(), key=lambda x: x[1]):
            layers.setdefault(depth, []).append(node_id)

        milestones: list[Milestone] = []
        cumulative_tasks = 0
        for depth in sorted(layers.keys()):
            task_ids = layers[depth]
            cumulative_tasks += len(task_ids)
            # 用第一个任务的描述作为里程碑简述
            first_node = task_graph.nodes[task_ids[0]]
            desc = (
                f"完成第 {depth + 1} 层任务"
                if len(task_ids) > 1
                else first_node.description
            )
            milestones.append(Milestone(
                id=f"milestone-{depth + 1}",
                description=desc,
                associated_task_ids=task_ids,
                expected_by_step=cumulative_tasks,
                status="PENDING",
            ))

        return milestones

    def _compute_depth(
        self,
        node_id: str,
        task_graph: TaskGraph,
        depths: dict[str, int],
    ) -> int:
        """计算节点在 DAG 中的拓扑深度（最长路径到根）."""
        if node_id in depths:
            return depths[node_id]
        node = task_graph.nodes[node_id]
        if not node.dependencies:
            depths[node_id] = 0
            return 0
        max_dep_depth = max(
            self._compute_depth(dep_id, task_graph, depths)
            for dep_id in node.dependencies
            if dep_id in task_graph.nodes
        )
        depths[node_id] = max_dep_depth + 1
        return depths[node_id]

    def _serialize_task_graph(self, task_graph: TaskGraph) -> dict:
        """将 TaskGraph 序列化为 dict."""
        nodes: dict[str, dict] = {}
        for node_id, node in task_graph.nodes.items():
            nodes[node_id] = {
                "id": node.id,
                "description": node.description,
                "dependencies": node.dependencies,
                "status": node.status.value,
                "files_involved": node.files_involved,
                "verification": node.verification,
            }
        return {
            "original_goal": task_graph.original_goal,
            "nodes": nodes,
        }

    def _check_milestones(self, ledger: TaskLedger) -> None:
        """检查是否有 milestone 因子任务完成而达成."""
        completed_ids = {ct.task_id for ct in ledger.completed_tasks}

        for milestone in ledger.milestones:
            if milestone.status == "REACHED":
                continue
            # 所有关联任务都完成了 → REACHED
            if milestone.associated_task_ids and all(
                tid in completed_ids for tid in milestone.associated_task_ids
            ):
                milestone.status = "REACHED"
                milestone.actual_step = ledger.total_steps
            # 检查是否 OVERDUE
            elif (
                milestone.status == "PENDING"
                and milestone.expected_by_step > 0
                and ledger.total_steps > milestone.expected_by_step
            ):
                milestone.status = "OVERDUE"

    def _recover_from_history(self, task_id: str) -> TaskLedger:
        """从 history.jsonl 恢复最近一条有效记录."""
        history_path = self.storage_dir / f"{task_id}.history.jsonl"
        if not history_path.exists():
            raise LedgerError(f"Ledger 和历史文件均不存在: {task_id}")

        last_valid: dict | None = None
        with open(history_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    last_valid = entry["snapshot"]
                except (json.JSONDecodeError, KeyError):
                    continue

        if last_valid is None:
            raise LedgerError(f"历史文件中无有效记录: {task_id}")

        logger.info("从历史记录恢复 Ledger: %s", task_id)
        ledger = TaskLedger.from_dict(last_valid)

        # 恢复后重新写入主文件
        self.save(ledger)
        return ledger

    def get_history(self, task_id: str, last_n: int = 20) -> list[dict]:
        """读取 history.jsonl 的最后 N 条记录."""
        history_path = self.storage_dir / f"{task_id}.history.jsonl"
        if not history_path.exists():
            return []

        entries: list[dict] = []
        with open(history_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    # 只保留时间戳和关键指标，不返回完整 snapshot
                    snapshot = entry.get("snapshot", {})
                    entries.append({
                        "timestamp": entry.get("timestamp", ""),
                        "status": snapshot.get("status", ""),
                        "current_phase": snapshot.get("current_phase", ""),
                        "current_task_id": snapshot.get("current_task_id"),
                        "completed_tasks": len(snapshot.get("completed_tasks", [])),
                        "total_tokens_used": snapshot.get("total_tokens_used", 0),
                        "total_steps": snapshot.get("total_steps", 0),
                    })
                except (json.JSONDecodeError, KeyError):
                    continue

        return entries[-last_n:]
