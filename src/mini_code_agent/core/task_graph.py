"""Task Graph — DAG 式任务依赖图，用于多步骤编程任务的规划与执行."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum


# ---------------------------------------------------------------------------
# 异常
# ---------------------------------------------------------------------------


class CyclicDependencyError(Exception):
    """添加依赖时检测到环."""


# ---------------------------------------------------------------------------
# TaskStatus
# ---------------------------------------------------------------------------


class TaskStatus(str, Enum):
    """任务状态枚举."""

    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    BLOCKED = "blocked"


# ---------------------------------------------------------------------------
# TaskNode
# ---------------------------------------------------------------------------


@dataclass
class TaskNode:
    """任务图中的单个节点."""

    id: str
    description: str
    dependencies: list[str] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    files_involved: list[str] = field(default_factory=list)
    verification: str = ""
    result: str | None = None
    error: str | None = None
    retry_count: int = 0


# ---------------------------------------------------------------------------
# TaskGraph
# ---------------------------------------------------------------------------


class TaskGraph:
    """DAG 式任务依赖图.

    核心能力：
    - 添加任务和依赖关系（带环检测）
    - 获取就绪任务（拓扑排序的"下一批"）
    - 标记任务完成/失败/跳过
    - 判断整体完成/阻塞状态
    - 找关键路径（最长依赖链）
    - 导出 Mermaid 图表
    """

    def __init__(self) -> None:
        self.nodes: dict[str, TaskNode] = {}
        self.original_goal: str = ""

    def add_task(self, node: TaskNode) -> None:
        """添加任务节点.

        如果节点声明了 dependencies，会同时校验这些依赖不会形成环。
        """
        if node.id in self.nodes:
            raise ValueError(f"任务 ID 重复: {node.id}")
        self.nodes[node.id] = node
        # 校验所有已声明依赖是否形成环
        for dep_id in node.dependencies:
            if dep_id in self.nodes:
                self._check_cycle(node.id, dep_id)

    def add_dependency(self, task_id: str, depends_on_id: str) -> None:
        """为已存在的任务添加依赖关系.

        Raises:
            KeyError: task_id 或 depends_on_id 不存在
            CyclicDependencyError: 添加后会形成环
        """
        if task_id not in self.nodes:
            raise KeyError(f"任务不存在: {task_id}")
        if depends_on_id not in self.nodes:
            raise KeyError(f"依赖任务不存在: {depends_on_id}")

        self._check_cycle(task_id, depends_on_id)

        task = self.nodes[task_id]
        if depends_on_id not in task.dependencies:
            task.dependencies.append(depends_on_id)

    def _check_cycle(self, task_id: str, new_dep_id: str) -> None:
        """DFS 检测添加 task_id -> new_dep_id 的依赖后是否形成环.

        如果 new_dep_id 能（通过已有依赖链）到达 task_id，就会形成环。
        """
        if task_id == new_dep_id:
            raise CyclicDependencyError(
                f"自依赖: {task_id} -> {new_dep_id}"
            )

        # 从 new_dep_id 出发，沿依赖方向（反向：找 new_dep_id 依赖了谁）看能否到达 task_id
        # 不对——应该沿"被依赖方向"：从 task_id 出发，看谁依赖了 task_id，能否到达 new_dep_id
        # 其实更简单：如果 new_dep_id 能通过**它的**依赖链到达 task_id，就有环
        visited: set[str] = set()
        stack = [new_dep_id]
        while stack:
            current = stack.pop()
            if current == task_id:
                raise CyclicDependencyError(
                    f"环依赖: {task_id} -> {new_dep_id} -> ... -> {task_id}"
                )
            if current in visited:
                continue
            visited.add(current)
            node = self.nodes.get(current)
            if node:
                stack.extend(node.dependencies)

    def get_ready_tasks(self) -> list[TaskNode]:
        """返回所有依赖已满足（COMPLETED）且自身是 PENDING 的任务.

        这是 DAG 拓扑排序的"当前可执行批次"。
        """
        ready = []
        for node in self.nodes.values():
            if node.status != TaskStatus.PENDING:
                continue
            # 所有依赖都必须已完成
            deps_met = all(
                self.nodes[dep_id].status == TaskStatus.COMPLETED
                for dep_id in node.dependencies
                if dep_id in self.nodes
            )
            if deps_met:
                ready.append(node)
        return ready

    def mark_running(self, task_id: str) -> None:
        """将任务标记为运行中."""
        self.nodes[task_id].status = TaskStatus.RUNNING

    def mark_completed(self, task_id: str, result: str) -> None:
        """将任务标记为完成."""
        node = self.nodes[task_id]
        node.status = TaskStatus.COMPLETED
        node.result = result

    def mark_failed(self, task_id: str, error: str) -> None:
        """将任务标记为失败，并将依赖它的下游任务标记为 BLOCKED."""
        node = self.nodes[task_id]
        node.status = TaskStatus.FAILED
        node.error = error
        # 传播阻塞状态到下游
        self._propagate_blocked(task_id)

    def mark_skipped(self, task_id: str) -> None:
        """将任务标记为跳过."""
        self.nodes[task_id].status = TaskStatus.SKIPPED

    def _propagate_blocked(self, failed_id: str) -> None:
        """将所有直接或间接依赖 failed_id 的 PENDING 任务标记为 BLOCKED."""
        queue = deque([failed_id])
        visited: set[str] = {failed_id}
        while queue:
            current_id = queue.popleft()
            # 找所有依赖 current_id 的下游任务
            for node in self.nodes.values():
                if node.id in visited:
                    continue
                if current_id in node.dependencies:
                    if node.status == TaskStatus.PENDING:
                        node.status = TaskStatus.BLOCKED
                    visited.add(node.id)
                    queue.append(node.id)

    def is_complete(self) -> bool:
        """所有任务都是 COMPLETED 或 SKIPPED."""
        return all(
            n.status in (TaskStatus.COMPLETED, TaskStatus.SKIPPED)
            for n in self.nodes.values()
        )

    def is_blocked(self) -> bool:
        """存在 FAILED 的任务，且有其他任务依赖它（即有 BLOCKED 任务）."""
        has_failed = any(n.status == TaskStatus.FAILED for n in self.nodes.values())
        has_blocked = any(n.status == TaskStatus.BLOCKED for n in self.nodes.values())
        return has_failed and has_blocked

    def get_critical_path(self) -> list[TaskNode]:
        """找出最长的依赖链（关键路径）.

        使用 DFS+记忆化，对每个节点计算以它为终点的最长路径长度。
        """
        if not self.nodes:
            return []

        memo: dict[str, list[str]] = {}  # node_id -> 以它为终点的最长路径 id 列表

        def _longest_path(node_id: str) -> list[str]:
            if node_id in memo:
                return memo[node_id]

            node = self.nodes[node_id]
            if not node.dependencies:
                memo[node_id] = [node_id]
                return memo[node_id]

            best: list[str] = []
            for dep_id in node.dependencies:
                if dep_id not in self.nodes:
                    continue
                path = _longest_path(dep_id)
                if len(path) > len(best):
                    best = path

            memo[node_id] = best + [node_id]
            return memo[node_id]

        # 对所有节点计算，取最长的
        longest: list[str] = []
        for nid in self.nodes:
            path = _longest_path(nid)
            if len(path) > len(longest):
                longest = path

        return [self.nodes[nid] for nid in longest]

    def to_mermaid(self) -> str:
        """导出为 Mermaid 流程图语法.

        状态用不同样式标注：
        - COMPLETED: 绿色
        - FAILED: 红色
        - RUNNING: 蓝色
        - BLOCKED: 灰色
        """
        lines = ["graph TD"]

        # 状态到样式类的映射
        status_class = {
            TaskStatus.COMPLETED: ":::completed",
            TaskStatus.FAILED: ":::failed",
            TaskStatus.RUNNING: ":::running",
            TaskStatus.BLOCKED: ":::blocked",
            TaskStatus.SKIPPED: ":::skipped",
            TaskStatus.PENDING: "",
            TaskStatus.READY: "",
        }

        for node in self.nodes.values():
            label = node.description.replace('"', "'")
            cls = status_class.get(node.status, "")
            lines.append(f'    {node.id}["{node.id}: {label}"]{cls}')

        for node in self.nodes.values():
            for dep_id in node.dependencies:
                lines.append(f"    {dep_id} --> {node.id}")

        # 样式定义
        lines.extend([
            "",
            "    classDef completed fill:#2ecc71,stroke:#27ae60,color:#fff",
            "    classDef failed fill:#e74c3c,stroke:#c0392b,color:#fff",
            "    classDef running fill:#3498db,stroke:#2980b9,color:#fff",
            "    classDef blocked fill:#95a5a6,stroke:#7f8c8d,color:#fff",
            "    classDef skipped fill:#f39c12,stroke:#e67e22,color:#fff",
        ])

        return "\n".join(lines)

    def summary(self) -> dict[str, int]:
        """返回各状态的任务计数."""
        counts: dict[str, int] = {}
        for status in TaskStatus:
            count = sum(1 for n in self.nodes.values() if n.status == status)
            if count > 0:
                counts[status.value] = count
        return counts

    def __len__(self) -> int:
        return len(self.nodes)
