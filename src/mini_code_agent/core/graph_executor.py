"""图执行器 — 按 DAG 顺序执行 Task Graph 中的任务."""

from __future__ import annotations

import asyncio
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ..llm.base import TokenUsage
from .agent import Agent, AgentResult
from .task_graph import TaskGraph, TaskNode, TaskStatus

if TYPE_CHECKING:
    from ..artifacts import SubtaskArtifact
    from ..longrun.checkpoint_manager import CheckpointManager
    from ..longrun.config import LongRunConfig
    from ..longrun.ledger_manager import TaskLedgerManager
    from ..longrun.session_state import SessionState
    from ..longrun.task_ledger import TaskLedger
    from .subtask_runner import SubtaskRunner

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 执行结果
# ---------------------------------------------------------------------------


@dataclass
class GraphResult:
    """Task Graph 整体执行结果."""

    graph: TaskGraph
    total_steps: int = 0
    total_tokens: int = 0
    wall_time: float = 0.0
    tasks_completed: int = 0
    tasks_failed: int = 0
    tasks_skipped: int = 0


# ---------------------------------------------------------------------------
# 验证运行器
# ---------------------------------------------------------------------------


async def run_verification(command: str, cwd: str | None = None) -> tuple[bool, str]:
    """运行验证命令，返回 (是否通过, 输出信息).

    使用 asyncio.create_subprocess_shell 执行，超时 60 秒。
    如果 command 不像合法的 shell 命令（含中文、纯自然语言描述），自动跳过。
    """
    if not command.strip():
        return True, ""

    if not _is_shell_command(command):
        logger.info("verification 不是可执行命令，跳过: %s", command[:80])
        return True, ""

    try:
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=60)
        output = (stdout or b"").decode("utf-8", errors="replace")
        err_output = (stderr or b"").decode("utf-8", errors="replace")
        combined = output
        if err_output:
            combined += "\n" + err_output

        passed = proc.returncode == 0
        return passed, combined.strip()

    except asyncio.TimeoutError:
        return False, "验证命令超时（60s）"
    except Exception as e:
        return False, f"验证命令执行失败: {type(e).__name__}: {e}"


# 常见 shell 命令前缀——verification 第一个 token 应该是这些之一
_SHELL_CMD_PREFIXES = frozenset({
    "python", "python3", "pytest", "node", "npm", "npx", "yarn",
    "bash", "sh", "zsh", "cat", "ls", "test", "echo", "grep",
    "head", "tail", "wc", "diff", "make", "cargo", "go", "java",
    "javac", "gcc", "g++", "ruby", "perl", "php", "curl", "wget",
    "git", "docker", "cd", "mv", "cp", "rm", "mkdir", "touch",
    "find", "which", "env", "true", "false", "exit", "pip", "uv",
    "[",  # test 的别名 `[ -f file ]`
})


def _is_shell_command(text: str) -> bool:
    """启发式判断 text 是否像一条可执行的 shell 命令.

    规则：
    1. 包含中文字符 → 大概率是自然语言描述，不是命令
    2. 第一个 token 是已知的命令名或路径 → 是命令
    3. 以 ./ 或 / 开头 → 是命令（执行脚本）
    """
    text = text.strip()
    if not text:
        return False

    # 含中文字符 → 自然语言
    if re.search(r"[\u4e00-\u9fff]", text):
        return False

    first_token = text.split()[0].lower()

    # 绝对路径或相对路径开头
    if first_token.startswith(("/", "./")):
        return True

    # 去掉路径前缀取 basename（如 /usr/bin/python -> python）
    basename = first_token.rsplit("/", 1)[-1]
    if basename in _SHELL_CMD_PREFIXES:
        return True

    # 允许带版本号的命令如 python3.11, node18
    base_no_ver = re.sub(r"[\d.]+$", "", basename)
    if base_no_ver in _SHELL_CMD_PREFIXES:
        return True

    return False


# ---------------------------------------------------------------------------
# 用户交互选择
# ---------------------------------------------------------------------------

# 回调类型：当执行被阻塞时，询问用户选择
# 返回 "replan" | "skip" | "abort"
GraphBlockedCallback = None  # 由调用方提供，避免 import 循环


# ---------------------------------------------------------------------------
# GraphExecutor
# ---------------------------------------------------------------------------


class GraphExecutor:
    """按 DAG 依赖顺序执行 Task Graph.

    执行逻辑：
    1. 获取所有就绪任务（依赖已完成）
    2. 串行执行每个就绪任务（后续可扩展为并行）
    3. 执行完后运行验证
    4. 失败的任务最多重试 2 次
    5. 如果有任务失败导致阻塞，通过回调询问用户
    """

    def __init__(
        self,
        agent_factory: AgentFactory | None = None,
        project_path: str | None = None,
        max_retries: int = 2,
        blocked_callback: object | None = None,
        progress_callback: object | None = None,
        subtask_runner: "SubtaskRunner | None" = None,
        ledger_manager: "TaskLedgerManager | None" = None,
        checkpoint_manager: "CheckpointManager | None" = None,
        longrun_config: "LongRunConfig | None" = None,
    ) -> None:
        """
        Args:
            agent_factory: 创建独立 Agent 的工厂函数。如果不提供，则使用传入的 agent。
            project_path: 项目根路径，用于 cwd。
            max_retries: 每个子任务最大重试次数。
            blocked_callback: async (graph, failed, blocked) -> "replan" | "skip" | "abort"
            progress_callback: async (task_index, total, task, phase) -> None
            subtask_runner: 走 Artifact 协议的子任务执行器；提供它时 execute_with_ledger 可用。
            ledger_manager: execute_with_ledger 需要 — 即时把任务结果写进 Ledger。
            checkpoint_manager: execute_with_ledger 需要 — 子任务完成后判断是否 checkpoint。
            longrun_config: execute_with_ledger 需要 — checkpoint 阈值与重试上限。
        """
        self.agent_factory = agent_factory
        self.project_path = project_path
        self.max_retries = max_retries
        self.blocked_callback = blocked_callback
        self.progress_callback = progress_callback
        self.subtask_runner = subtask_runner
        self.ledger_manager = ledger_manager
        self.checkpoint_manager = checkpoint_manager
        self.longrun_config = longrun_config
        self.last_checkpoint: "SessionState | None" = None

    async def execute(
        self,
        graph: TaskGraph,
        agent: Agent,
    ) -> GraphResult:
        """执行 Task Graph.

        每个子任务使用独立的 Agent 消息历史（通过 reset），
        但将已完成任务的结果摘要传递给后续任务。

        Args:
            graph: 要执行的任务图
            agent: Agent 实例（用于执行子任务）

        Returns:
            GraphResult 包含最终状态和统计信息
        """
        start_time = time.monotonic()
        total_tokens = 0
        total_steps = 0

        while not graph.is_complete() and not graph.is_blocked():
            ready_tasks = graph.get_ready_tasks()
            if not ready_tasks:
                break

            for task in ready_tasks:
                total_steps += 1
                completed_count = sum(
                    1 for n in graph.nodes.values()
                    if n.status == TaskStatus.COMPLETED
                )
                total_count = len(graph.nodes)

                # 进度回调
                if self.progress_callback:
                    await self.progress_callback(
                        completed_count + 1, total_count, task, "start"
                    )

                graph.mark_running(task.id)

                # 构建子任务 prompt
                task_prompt = self._build_task_prompt(graph, task)

                # 用独立的消息历史执行子任务
                agent.reset()
                try:
                    result: AgentResult = await agent.run(task_prompt)
                except Exception as e:
                    logger.exception("子任务 %s 执行异常", task.id)
                    result = AgentResult(
                        content=f"执行异常: {type(e).__name__}: {e}",
                        stop_reason="error",
                    )

                total_tokens += result.usage.total_tokens

                # 从 result 中提取摘要
                result_summary = self._extract_summary(result.content)

                # 运行验证
                verification_passed = True
                verification_output = ""
                if task.verification:
                    verification_passed, verification_output = await run_verification(
                        task.verification, cwd=self.project_path
                    )

                if verification_passed and result.stop_reason == "ok":
                    graph.mark_completed(task.id, result_summary)
                    if self.progress_callback:
                        await self.progress_callback(
                            completed_count + 1, total_count, task, "end_ok"
                        )
                else:
                    error_info = ""
                    if result.stop_reason != "ok":
                        error_info = f"Agent 停止原因: {result.stop_reason}. "
                    if not verification_passed:
                        error_info += f"验证失败: {verification_output}"

                    if task.retry_count < self.max_retries:
                        task.retry_count += 1
                        task.status = TaskStatus.PENDING
                        logger.info(
                            "子任务 %s 失败，重试 (%d/%d): %s",
                            task.id, task.retry_count, self.max_retries, error_info,
                        )
                        if self.progress_callback:
                            await self.progress_callback(
                                completed_count + 1, total_count, task, "retry"
                            )
                    else:
                        graph.mark_failed(task.id, error_info)
                        if self.progress_callback:
                            await self.progress_callback(
                                completed_count + 1, total_count, task, "end_fail"
                            )

        # 如果有任务失败导致阻塞，询问用户
        if graph.is_blocked() and self.blocked_callback:
            failed = [n for n in graph.nodes.values() if n.status == TaskStatus.FAILED]
            blocked = [n for n in graph.nodes.values() if n.status == TaskStatus.BLOCKED]
            choice = await self.blocked_callback(graph, failed, blocked)

            if choice == "skip":
                # 跳过失败任务和被阻塞的任务中不依赖失败任务的
                for node in blocked:
                    graph.mark_skipped(node.id)
                for node in failed:
                    node.status = TaskStatus.SKIPPED
                # 继续执行不依赖失败任务的剩余任务
                if not graph.is_complete():
                    sub_result = await self.execute(graph, agent)
                    total_tokens += sub_result.total_tokens
                    total_steps += sub_result.total_steps

            elif choice == "abort":
                pass  # 直接返回当前状态

        wall_time = time.monotonic() - start_time

        return GraphResult(
            graph=graph,
            total_steps=total_steps,
            total_tokens=total_tokens,
            wall_time=wall_time,
            tasks_completed=sum(
                1 for n in graph.nodes.values() if n.status == TaskStatus.COMPLETED
            ),
            tasks_failed=sum(
                1 for n in graph.nodes.values() if n.status == TaskStatus.FAILED
            ),
            tasks_skipped=sum(
                1 for n in graph.nodes.values() if n.status == TaskStatus.SKIPPED
            ),
        )

    async def execute_with_ledger(
        self,
        graph: TaskGraph,
        ledger: "TaskLedger",
        project_path: str,
    ) -> GraphResult:
        """长程任务执行路径：经 SubtaskRunner 产出 Artifact，即时写 Ledger.

        与 execute() 的区别：
        - 不直接调 Agent.run，而是经 SubtaskRunner 走 Artifact 协议
        - 子任务完成后立即把 CompletedTaskRecord / FailedAttemptRecord 写入 Ledger
        - 根据 Artifact.confidence 分别处理（DONE / UNCERTAIN / PARTIAL / STUCK）
        - 自动触发 checkpoint
        """
        if self.subtask_runner is None or self.ledger_manager is None:
            raise RuntimeError(
                "execute_with_ledger 需要 subtask_runner 和 ledger_manager；"
                "未提供时请使用 execute()"
            )

        from ..artifacts import Confidence
        from ..longrun.ledger_types import ActiveIssue, TaskRunStatus

        start_time = time.monotonic()
        total_steps = 0
        total_tokens = 0
        max_retries = self.max_retries
        budget_exhausted = False

        # 进入 RUNNING 状态
        ledger.status = TaskRunStatus.RUNNING
        self.ledger_manager.update_phase(ledger, "execution")

        while not graph.is_complete() and not graph.is_blocked():
            # Budget 阻断
            if (
                ledger.token_budget > 0
                and ledger.total_tokens_used >= ledger.token_budget
            ):
                logger.warning(
                    "token 预算耗尽 (%d/%d)，中止执行",
                    ledger.total_tokens_used, ledger.token_budget,
                )
                self.ledger_manager.record_active_issue(
                    ledger,
                    ActiveIssue(
                        id=str(uuid.uuid4()),
                        description=(
                            f"Token 预算已耗尽 ({ledger.total_tokens_used:,}"
                            f"/{ledger.token_budget:,})，剩余子任务未执行。"
                        ),
                        source_task_id=ledger.current_task_id or "",
                        severity="blocker",
                        first_seen_step=ledger.total_steps,
                    ),
                )
                budget_exhausted = True
                break

            ready = graph.get_ready_tasks()
            if not ready:
                break

            # 每轮一个任务（串行）
            task = ready[0]
            total_steps += 1
            completed_count = sum(
                1 for n in graph.nodes.values() if n.status == TaskStatus.COMPLETED
            )
            total_count = len(graph.nodes)

            if self.progress_callback:
                await self.progress_callback(
                    completed_count + 1, total_count, task, "start"
                )

            graph.mark_running(task.id)
            self.ledger_manager.update_current_task(ledger, task.id)

            # 构造 GraphContext
            from .subtask_runner import GraphContext
            ctx = GraphContext(
                original_goal=graph.original_goal or ledger.goal,
                completed_summaries=[
                    ct.self_summary for ct in ledger.completed_tasks[-5:]
                ],
                project_path=project_path,
                allowed_paths=self.subtask_runner.derive_allowed_paths(task),
            )

            # 执行子任务
            try:
                artifact = await self.subtask_runner.run(task, ctx)
            except Exception as e:
                # 基础设施错误
                logger.exception("子任务 %s 基础设施错误", task.id)
                graph.mark_failed(task.id, f"infrastructure error: {e}")
                if self.progress_callback:
                    await self.progress_callback(
                        completed_count + 1, total_count, task, "end_fail"
                    )
                continue

            total_tokens += artifact.resource_usage.tokens_total

            # 根据 confidence 处理
            if artifact.confidence == Confidence.DONE:
                self.ledger_manager.record_task_completed(ledger, artifact)
                graph.mark_completed(task.id, artifact.self_summary)
                if self.progress_callback:
                    await self.progress_callback(
                        completed_count + 1, total_count, task, "end_ok"
                    )

            elif artifact.confidence == Confidence.UNCERTAIN:
                self.ledger_manager.record_task_completed(ledger, artifact)
                self.ledger_manager.record_active_issue(
                    ledger,
                    ActiveIssue(
                        id=str(uuid.uuid4()),
                        description=(
                            f"子任务 {task.id} 验证未通过但 Agent 认为已完成: "
                            f"{artifact.self_summary[:120]}"
                        ),
                        source_task_id=task.id,
                        severity="warning",
                        first_seen_step=ledger.total_steps,
                    ),
                )
                graph.mark_completed(task.id, artifact.self_summary)
                if self.progress_callback:
                    await self.progress_callback(
                        completed_count + 1, total_count, task, "end_ok"
                    )

            elif artifact.confidence == Confidence.PARTIAL:
                self.ledger_manager.record_task_completed(ledger, artifact)
                for q in artifact.open_questions:
                    self.ledger_manager.record_active_issue(
                        ledger,
                        ActiveIssue(
                            id=str(uuid.uuid4()),
                            description=q,
                            source_task_id=task.id,
                            severity="info",
                            first_seen_step=ledger.total_steps,
                        ),
                    )
                graph.mark_completed(task.id, artifact.self_summary)
                if self.progress_callback:
                    await self.progress_callback(
                        completed_count + 1, total_count, task, "end_ok"
                    )

            else:  # Confidence.STUCK
                self.ledger_manager.record_task_failed(
                    ledger, artifact, failure_reason="agent declared STUCK",
                )
                if task.retry_count < max_retries:
                    task.retry_count += 1
                    task.status = TaskStatus.PENDING
                    if self.progress_callback:
                        await self.progress_callback(
                            completed_count + 1, total_count, task, "retry"
                        )
                else:
                    graph.mark_failed(task.id, "max retries exhausted (STUCK)")
                    if self.progress_callback:
                        await self.progress_callback(
                            completed_count + 1, total_count, task, "end_fail"
                        )

            # 自动 checkpoint
            await self._maybe_checkpoint(ledger, graph)

        wall_time = time.monotonic() - start_time

        # 收尾 Ledger — 不再依赖调用方
        ledger.current_task_id = None
        tasks_failed_count = sum(
            1 for n in graph.nodes.values() if n.status == TaskStatus.FAILED
        )
        if budget_exhausted:
            ledger.status = TaskRunStatus.PAUSED
        elif tasks_failed_count > 0:
            ledger.status = TaskRunStatus.FAILED
        else:
            ledger.status = TaskRunStatus.COMPLETED
        self.ledger_manager.update_phase(ledger, "done")
        # update_resources 既会保存 ledger，也会刷新 token_budget_remaining
        self.ledger_manager.update_resources(
            ledger, tokens=0, steps=0, wall_time=wall_time,
        )

        return GraphResult(
            graph=graph,
            total_steps=total_steps,
            total_tokens=total_tokens,
            wall_time=wall_time,
            tasks_completed=sum(
                1 for n in graph.nodes.values() if n.status == TaskStatus.COMPLETED
            ),
            tasks_failed=tasks_failed_count,
            tasks_skipped=sum(
                1 for n in graph.nodes.values() if n.status == TaskStatus.SKIPPED
            ),
        )

    async def _maybe_checkpoint(
        self,
        ledger: "TaskLedger",
        graph: TaskGraph,
    ) -> None:
        if (
            self.checkpoint_manager is None
            or self.longrun_config is None
        ):
            return
        try:
            trigger = self.checkpoint_manager.auto_checkpoint_policy(
                ledger, self.last_checkpoint, self.longrun_config,
            )
        except Exception as e:  # noqa: BLE001
            logger.debug("auto_checkpoint_policy error: %s", e)
            return
        if trigger is None:
            return
        try:
            self.last_checkpoint = await self.checkpoint_manager.save_checkpoint(
                ledger=ledger,
                task_graph=graph,
                trigger=trigger,
                config=self.longrun_config,
                current_task_id=ledger.current_task_id,
                recent_messages=[],
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("自动 checkpoint 失败: %s", e)

    def _build_task_prompt(self, graph: TaskGraph, task: TaskNode) -> str:
        """构建子任务的 prompt，包含上下文信息."""
        # 收集已完成的依赖任务的结果摘要
        completed_deps: list[str] = []
        for dep_id in task.dependencies:
            dep_node = graph.nodes.get(dep_id)
            if dep_node and dep_node.status == TaskStatus.COMPLETED and dep_node.result:
                completed_deps.append(
                    f"- [{dep_node.id}] {dep_node.description}: {dep_node.result}"
                )

        lines = [
            "你正在执行一个大任务的子任务。",
            "",
            f"总体目标：{graph.original_goal}",
            f"当前子任务：{task.description}",
        ]

        if task.files_involved:
            lines.append(f"涉及文件：{', '.join(task.files_involved)}")

        if completed_deps:
            lines.append("")
            lines.append("已完成的相关任务：")
            lines.extend(completed_deps)

        if task.verification:
            lines.append("")
            lines.append(f"完成后我会用以下方式验证：{task.verification}")

        retry_note = ""
        if task.retry_count > 0:
            retry_note = f"\n\n注意：这是第 {task.retry_count + 1} 次尝试。"
            if task.error:
                retry_note += f"\n上次失败原因：{task.error}"

        lines.append("")
        lines.append("请只做这个子任务，不要做其他的。")

        if retry_note:
            lines.append(retry_note)

        return "\n".join(lines)

    def _extract_summary(self, content: str, max_length: int = 200) -> str:
        """从 Agent 回复中提取摘要."""
        if not content:
            return "(无输出)"
        # 取前几行作为摘要
        lines = content.strip().splitlines()
        summary = lines[0]
        if len(summary) > max_length:
            summary = summary[:max_length] + "..."
        return summary


# 类型别名（避免循环 import）
AgentFactory = object  # Callable[[], Agent]
