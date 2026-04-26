"""ResumeManager — 从 Checkpoint 恢复长程任务."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Callable

from ..core.task_graph import TaskGraph
from ..tools.git import _run_git
from .checkpoint_manager import CheckpointManager, CorruptedCheckpointError, _compute_sha256
from .ledger_manager import TaskLedgerManager
from .session_state import SessionState
from .task_ledger import TaskLedger

logger = logging.getLogger(__name__)


class ResumeError(Exception):
    """恢复操作错误."""


class UncommittedChangesError(ResumeError):
    """当前工作区有未提交的修改，恢复会丢失这些修改."""


@dataclass
class ResumeContext:
    """恢复上下文 — prepare_resume 的返回值."""

    session_state: SessionState
    ledger: TaskLedger
    task_graph: TaskGraph
    initial_prompt: str         # 重建 Agent 后给它的第一条消息
    warnings: list[str] = field(default_factory=list)


class ResumeManager:
    """从 Checkpoint 恢复长程任务.

    依赖：
    - CheckpointManager: 加载 checkpoint
    - TaskLedgerManager: 加载 ledger
    - GitCheckpoint: git 回滚
    """

    def __init__(
        self,
        checkpoint_manager: CheckpointManager,
        ledger_manager: TaskLedgerManager,
        cwd: str | None = None,
    ) -> None:
        self.checkpoint_manager = checkpoint_manager
        self.ledger_manager = ledger_manager
        self.cwd = cwd

    async def prepare_resume(
        self,
        task_id: str,
        checkpoint_id: str,
    ) -> ResumeContext:
        """准备恢复上下文.

        执行步骤：
        1. 加载 SessionState
        2. 加载 Ledger（+ 一致性检查）
        3. Git 回滚
        4. 重建 TaskGraph
        5. 构造恢复 prompt
        """
        warnings: list[str] = []

        # 步骤 1 — 加载 SessionState
        state = self.checkpoint_manager.load_checkpoint(task_id, checkpoint_id)

        # 验证 checkpoint 完整性
        try:
            validation_warnings = await self.checkpoint_manager.validate_checkpoint(state)
            warnings.extend(validation_warnings)
        except CorruptedCheckpointError:
            raise

        # 步骤 2 — 加载 Ledger
        ledger = self.ledger_manager.load(state.task_id)

        # 验证 ledger 和 session_state 一致
        if state.ledger_path:
            from pathlib import Path
            ledger_file = Path(state.ledger_path)
            if ledger_file.exists():
                current_hash = _compute_sha256(state.ledger_path)
                if current_hash != state.ledger_hash:
                    warnings.append(
                        "Ledger 在 checkpoint 创建后被修改过。"
                        "恢复时会使用当前的 Ledger，请检查是否符合预期。"
                    )

        # 步骤 3 — Git 回滚（关键步骤）
        code, current_branch = await _run_git(
            "rev-parse", "--abbrev-ref", "HEAD", cwd=self.cwd
        )
        current_branch = current_branch.strip() if code == 0 else ""

        if current_branch and state.git_branch and current_branch != state.git_branch:
            raise ResumeError(
                f"分支不匹配：当前 {current_branch}，"
                f"checkpoint {state.git_branch}"
            )

        # 检查是否有未提交的修改
        code, status_output = await _run_git("status", "--porcelain", cwd=self.cwd)
        if code == 0 and status_output.strip():
            raise UncommittedChangesError(
                "当前工作区有未提交的修改。恢复会丢失这些修改。\n"
                "请先 git stash 或 commit，然后重试 /resume。"
            )

        # 回滚到 checkpoint 时的 commit
        if state.git_checkpoint_hash:
            code, out = await _run_git(
                "reset", "--hard", state.git_checkpoint_hash, cwd=self.cwd
            )
            if code != 0:
                raise ResumeError(f"Git reset 失败: {out}")
            logger.info("已回滚到 %s", state.git_checkpoint_hash[:8])

        # 步骤 4 — 重建 TaskGraph
        task_graph = TaskGraph.from_json(state.task_graph_json)

        # 步骤 5 — 构造恢复 prompt
        initial_prompt = self._build_resume_prompt(ledger, state)

        return ResumeContext(
            session_state=state,
            ledger=ledger,
            task_graph=task_graph,
            initial_prompt=initial_prompt,
            warnings=warnings,
        )

    def _build_resume_prompt(self, ledger: TaskLedger, state: SessionState) -> str:
        """构造恢复 prompt — 比普通的 build_context_summary 更聚焦."""
        now = datetime.now(UTC)
        delta = now - state.created_at
        if delta.total_seconds() < 60:
            time_ago = f"{int(delta.total_seconds())} 秒"
        elif delta.total_seconds() < 3600:
            time_ago = f"{int(delta.total_seconds() / 60)} 分钟"
        elif delta.total_seconds() < 86400:
            time_ago = f"{delta.total_seconds() / 3600:.1f} 小时"
        else:
            time_ago = f"{delta.days} 天"

        total_tasks = len(ledger.task_graph_snapshot.get("nodes", {}))
        completed_count = len(ledger.completed_tasks)

        lines = [
            "你正在恢复一个之前暂停的长程任务。",
            "",
            "== 任务恢复信息 ==",
            f"任务 ID: {ledger.task_id}",
            f"原始目标: {ledger.goal}",
            f"暂停于: {state.created_at.isoformat()}（{time_ago}前）",
            f"暂停原因: {state.trigger.value}",
            "",
            "== 当前进度 ==",
            f"阶段: {ledger.current_phase}",
            f"已完成子任务: {completed_count} / {total_tasks}",
            f"当前子任务: {state.current_task_id or '未开始新子任务'}",
            "",
            f"已用资源: {ledger.total_tokens_used:,} token / 预算 {ledger.token_budget:,}",
            f"已执行步数: {ledger.total_steps}",
        ]

        # 最近完成
        recent_completed = ledger.completed_tasks[-3:]
        if recent_completed:
            lines.append("")
            lines.append("== 最近完成 ==")
            for ct in recent_completed:
                lines.append(f"- {ct.description}: {ct.self_summary}")

        # 当前未解决问题
        if ledger.active_issues:
            lines.append("")
            lines.append("== 当前未解决问题 ==")
            for issue in ledger.active_issues:
                lines.append(f"- [{issue.severity}] {issue.description}")

        # 需要避免的错误
        recent_failures = ledger.failed_attempts[-3:]
        if recent_failures:
            lines.append("")
            lines.append("== 需要避免的错误 ==")
            for fa in recent_failures:
                lesson = fa.lesson_learned or fa.failure_reason
                lines.append(f"- {lesson}")

        lines.append("")
        lines.append("== 恢复指令 ==")
        if state.current_task_id:
            lines.append(f"请从子任务 {state.current_task_id} 继续执行。")
        else:
            lines.append("请从下一个就绪的子任务继续执行。")
        lines.append("你不需要重做任何已完成的工作。")
        lines.append("如果有需要，可以用 /status 查看详细状态。")

        return "\n".join(lines)

    async def execute_resume(
        self,
        context: ResumeContext,
        agent_factory: Callable[..., Any],
    ) -> Any:
        """执行恢复：用 agent_factory 创建 Agent 并注入 initial_prompt.

        agent_factory 是一个函数，负责构造 Agent 实例（因为 ResumeManager
        不应该知道 Agent 的具体构造参数）。

        Returns:
            创建好的 Agent 实例（已注入 initial_prompt）。
        """
        agent = agent_factory(
            ledger=context.ledger,
            task_graph=context.task_graph,
        )

        # 把 initial_prompt 作为第一条 user message 注入
        if hasattr(agent, "inject_initial_message"):
            agent.inject_initial_message(context.initial_prompt)

        return agent
