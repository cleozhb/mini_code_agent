"""主 REPL 循环 — 用 prompt_toolkit 输入 + Rich 流式输出."""

from __future__ import annotations

import asyncio
import json
import sys
from functools import partial
from typing import Any

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

from ..core.agent import Agent, AgentEvent, AgentEventType
from ..llm.base import LLMClient, TokenUsage, ToolCall
from ..llm.pricing import estimate_cost
from ..tools.base import ToolRegistry
from .confirm import confirm_tool_call

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..artifacts import ArtifactStore


_WRITE_TOOL_NAMES = {"WriteFile", "EditFile", "write_file", "edit_file"}


def _changed_path_from_subagent_event(event: AgentEvent) -> str | None:
    """从 SubAgent 内部工具结果事件中提取成功写入的文件路径."""
    if event.type != AgentEventType.TOOL_RESULT:
        return None
    if event.tool_call is None or event.tool_result is None:
        return None
    if event.tool_call.name not in _WRITE_TOOL_NAMES:
        return None
    if event.tool_result.is_error:
        return None

    args = event.tool_call.arguments
    path = args.get("path") or args.get("file_path")
    if not isinstance(path, str) or not path:
        return None
    return path


# ---------------------------------------------------------------------------
# REPL
# ---------------------------------------------------------------------------


class REPL:
    """Agent 交互式 REPL."""

    def __init__(
        self,
        agent: Agent,
        console: Console | None = None,
        artifact_store: ArtifactStore | None = None,
    ) -> None:
        self.agent = agent
        self.console = console or Console()
        self._artifact_store = artifact_store

        # 斜杠命令补全
        self._completer = WordCompleter(
            [
                "/quit", "/exit", "/q", "/clear", "/cost", "/model",
                "/memory", "/save",
                "/undo", "/checkpoints", "/diff",
                "/goal", "/plan", "/exec",
            ],
            sentence=True,
        )

        # prompt_toolkit session，带文件历史和命令补全
        self._prompt_session = PromptSession[str](
            history=FileHistory(".agent_history"),
            completer=self._completer,
        )

    async def run(self) -> None:
        """启动 REPL 主循环."""
        self._print_welcome()

        while True:
            try:
                user_input = await self._get_input()
            except (EOFError, KeyboardInterrupt):
                self.console.print("\n[dim]再见！[/dim]")
                break

            if not user_input.strip():
                continue

            # 处理特殊命令
            if user_input.startswith("/"):
                should_continue = await self._handle_command(user_input.strip())
                if not should_continue:
                    break
                continue

            await self._run_agent_stream(user_input)

    async def _get_input(self) -> str:
        """获取用户输入，支持多行（Alt+Enter 提交，Enter 换行默认单行）."""
        # 创建 key bindings: Enter 提交, Alt+Enter / Esc+Enter 换行
        bindings = KeyBindings()

        @bindings.add("escape", "enter")
        def _newline(event):
            event.current_buffer.insert_text("\n")

        result = await self._prompt_session.prompt_async(
            HTML("<b><ansiblue>>>> </ansiblue></b>"),
            key_bindings=bindings,
            multiline=False,
        )
        return result

    async def _handle_command(self, cmd: str) -> bool:
        """处理 / 开头的特殊命令. 返回 False 表示退出 REPL."""
        parts = cmd.split(maxsplit=1)
        command = parts[0].lower()

        if command in ("/quit", "/exit", "/q"):
            self.console.print("[dim]再见！[/dim]")
            return False

        if command == "/clear":
            self.agent.reset()
            self.console.clear()
            self.console.print("[green]对话历史已清空[/green]")
            return True

        if command == "/cost":
            self._show_cost()
            return True

        if command == "/model":
            model_arg = parts[1].strip() if len(parts) > 1 else ""
            self._switch_model(model_arg)
            return True

        if command == "/memory":
            self._show_memory()
            return True

        if command == "/save":
            save_arg = parts[1].strip() if len(parts) > 1 else ""
            self._save_memory(save_arg)
            return True

        if command == "/undo":
            await self._undo()
            return True

        if command == "/checkpoints":
            await self._show_checkpoints()
            return True

        if command == "/diff":
            await self._show_agent_diff()
            return True

        if command == "/goal":
            goal_arg = parts[1].strip() if len(parts) > 1 else ""
            await self._handle_goal_command(goal_arg)
            return True

        if command == "/plan":
            plan_arg = parts[1].strip() if len(parts) > 1 else ""
            await self._handle_plan_command(plan_arg)
            return True

        if command == "/exec":
            exec_arg = parts[1].strip() if len(parts) > 1 else ""
            await self._handle_exec_command(exec_arg)
            return True

        self.console.print(f"[red]未知命令: {command}[/red]")
        self.console.print(
            "[dim]可用命令: /quit  /clear  /cost  /model  /memory  /save"
            "  /undo  /checkpoints  /diff  /goal  /plan  /exec[/dim]"
        )
        return True

    async def _undo(self) -> None:
        """回滚最近一次 Agent 的所有修改."""
        cp = self.agent.git_checkpoint
        if cp is None:
            self.console.print("[red]Git checkpoint 未启用[/red]")
            return

        if not await cp.is_git_repo():
            self.console.print("[red]当前目录不是 git 仓库，无法回滚[/red]")
            return

        checkpoints = await cp.list_checkpoints()
        if not checkpoints:
            self.console.print("[yellow]没有找到 checkpoint，无法回滚[/yellow]")
            return

        latest = checkpoints[0]
        self.console.print(
            f"[dim]将回滚到 checkpoint 之前的状态: "
            f"{latest.message} ({latest.commit_hash[:8]})[/dim]"
        )

        success = await cp.rollback_last()
        if success:
            self.console.print("[green]回滚成功[/green]")
        else:
            self.console.print("[red]回滚失败[/red]")

    async def _show_checkpoints(self) -> None:
        """列出所有 agent checkpoint."""
        cp = self.agent.git_checkpoint
        if cp is None:
            self.console.print("[red]Git checkpoint 未启用[/red]")
            return

        checkpoints = await cp.list_checkpoints()
        if not checkpoints:
            self.console.print("[dim]暂无 git checkpoint[/dim]")
        else:
            lines = []
            for i, c in enumerate(checkpoints):
                lines.append(
                    f"  {i + 1}. [{c.commit_hash[:8]}] {c.message}  "
                    f"[dim]({c.timestamp})[/dim]"
                )

            self.console.print(Panel(
                "\n".join(lines),
                title=f"[bold]Git Checkpoints ({len(checkpoints)})[/bold]",
                border_style="cyan",
            ))

    async def _show_agent_diff(self) -> None:
        """查看 Agent 自最近 checkpoint 以来的所有修改."""
        cp = self.agent.git_checkpoint
        if cp is None:
            self.console.print("[red]Git checkpoint 未启用[/red]")
            return

        checkpoints = await cp.list_checkpoints()
        if not checkpoints:
            self.console.print("[dim]暂无 checkpoint，无法显示 diff[/dim]")
            return

        # 找最早的 "before:" checkpoint 作为基准
        base_hash = checkpoints[-1].commit_hash
        from ..tools.git import _run_git

        code, diff_output = await _run_git(
            "diff", f"{base_hash}~1", "HEAD", cwd=cp.cwd
        )
        if code != 0:
            # 如果 ~1 不存在（首个 commit），直接 show
            code, diff_output = await _run_git(
                "diff", base_hash, "HEAD", cwd=cp.cwd
            )

        if not diff_output.strip():
            self.console.print("[dim]无改动[/dim]")
            return

        lines = diff_output.splitlines()
        if len(lines) > 200:
            display = "\n".join(lines[:200])
            display += f"\n\n... [截断：共 {len(lines)} 行，仅显示前 200 行]"
        else:
            display = diff_output

        from rich.syntax import Syntax
        self.console.print(Syntax(display, "diff", theme="monokai"))

    def _show_cost(self) -> None:
        """显示 token 消耗和费用估算."""
        usage = self.agent.total_usage
        model = self.agent.llm_client.model
        cost = estimate_cost(model, usage)

        lines = [
            f"模型: {model}",
            f"输入 tokens: {usage.input_tokens:,}",
            f"输出 tokens: {usage.output_tokens:,}",
            f"总计 tokens: {usage.total_tokens:,}",
        ]
        if usage.cached_input_tokens:
            lines.append(f"OpenAI cached input tokens: {usage.cached_input_tokens:,}")
        if usage.reasoning_tokens:
            lines.append(f"OpenAI reasoning tokens: {usage.reasoning_tokens:,}")
        if usage.cache_creation_input_tokens:
            lines.append(
                "Anthropic cache creation tokens: "
                f"{usage.cache_creation_input_tokens:,}"
            )
        if usage.cache_read_input_tokens:
            lines.append(
                f"Anthropic cache read tokens: {usage.cache_read_input_tokens:,}"
            )
        if cost is not None:
            suffix = "" if cost.is_complete else "（部分估算）"
            lines.append(f"估算费用: ${cost.total_cost:.4f}{suffix}")
            if cost.missing_price_items:
                lines.append(
                    "缺失价格项: " + ", ".join(cost.missing_price_items)
                )
        else:
            lines.append("估算费用: (该模型暂无价格数据)")

        self.console.print(Panel(
            "\n".join(lines),
            title="[bold]会话消耗[/bold]",
            border_style="cyan",
        ))

    def _show_memory(self) -> None:
        """显示项目长期记忆和对话 token 统计."""
        lines: list[str] = []

        # 对话 token 统计
        conv = self.agent.conversation
        lines.append(f"对话 token 数: {conv.token_count:,}")
        lines.append(f"对话消息数: {len(conv.messages)}")
        threshold = int(conv.max_tokens * conv.compress_ratio)
        lines.append(f"压缩阈值: {threshold:,} tokens")

        # 项目记忆
        pm = self.agent.project_memory
        if pm:
            data = pm.data
            lines.append("")
            lines.append(f"项目约定: {len(data.conventions)} 条")
            for c in data.conventions:
                lines.append(f"  - {c}")
            lines.append(f"技术决策: {len(data.decisions)} 条")
            for d in data.decisions:
                lines.append(f"  - [{d.date}] {d.decision}")
            lines.append(f"已知问题: {len(data.known_issues)} 条")
            for ki in data.known_issues:
                lines.append(f"  - {ki.issue}")
        else:
            lines.append("\n(项目记忆未启用)")

        self.console.print(Panel(
            "\n".join(lines),
            title="[bold]记忆状态[/bold]",
            border_style="magenta",
        ))

    def _save_memory(self, text: str) -> None:
        """手动保存一条信息到项目记忆（约定类型）."""
        pm = self.agent.project_memory
        if not pm:
            self.console.print("[red]项目记忆未启用[/red]")
            return
        if not text:
            self.console.print("[dim]用法: /save <要记住的信息>[/dim]")
            return
        pm.add_convention(text)
        self.console.print(f"[green]已保存到项目记忆: {text}[/green]")

    def _switch_model(self, model_name: str) -> None:
        """切换模型."""
        if not model_name:
            self.console.print(f"[dim]当前模型: {self.agent.llm_client.model}[/dim]")
            self.console.print("[dim]用法: /model <模型名称>[/dim]")
            return
        old = self.agent.llm_client.model
        self.agent.llm_client.model = model_name
        self.console.print(f"[green]模型已切换: {old} → {model_name}[/green]")

    async def _handle_plan_command(self, arg: str) -> None:
        """/plan [--base file] <需求描述> — 派发 plan 子 Agent 输出方案.

        同会话内自动基于上次 /plan 生成的文件迭代；
        也可用 --base plan-xxx.md 显式指定基准计划。
        """
        goal = arg.strip()
        if not goal:
            self.console.print("[dim]用法: /plan <需求描述>[/dim]")
            self.console.print("[dim]  可选: /plan --base plan-xxx.md <需求描述>[/dim]")
            return

        from ..tools.subagent import SubAgentTool
        from ..core.agent import AgentEvent, AgentEventType
        from pathlib import Path

        project_path = str(Path(".").resolve())

        # 解析 --base 参数
        base_file: Path | None = None
        if goal.startswith("--base "):
            parts = goal[len("--base "):].split(maxsplit=1)
            if len(parts) < 2:
                self.console.print("[red]用法: /plan --base <文件路径> <需求描述>[/red]")
                return
            base_path_str, goal = parts[0], parts[1]
            candidate = Path(base_path_str)
            if not candidate.is_absolute():
                candidate = Path(project_path) / ".agent" / "plans" / candidate
            if candidate.is_file():
                base_file = candidate
            else:
                self.console.print(f"[red]计划文件不存在: {base_path_str}[/red]")
                return

        # 同会话内自动关联上次 /plan 生成的文件
        if base_file is None and hasattr(self, "_last_plan_file"):
            p = Path(self._last_plan_file)
            if p.is_file():
                base_file = p

        # 构建 context：包含历史输入 + 计划文件
        context_parts: list[str] = []

        # 追加历史用户输入（多轮对话上下文）
        if not hasattr(self, "_plan_history"):
            self._plan_history: list[str] = []
        if self._plan_history:
            history_text = "\n".join(
                f"第{i+1}轮用户需求: {h}" for i, h in enumerate(self._plan_history)
            )
            context_parts.append(f"以下是同 session 内之前的 /plan 需求历史：\n{history_text}")

        if base_file is not None:
            content = base_file.read_text(encoding="utf-8")
            context_parts.append(
                f"以下是当前已有的计划文件（{base_file.name}），请在此基础上修改/扩展，"
                f"而不是从零开始：\n\n{content}"
            )
            self.console.print(f"[dim]基于已有计划: {base_file.name}[/dim]")

        context = "\n\n".join(context_parts)

        sub_tool = SubAgentTool(
            llm_client=self.agent.llm_client,
            project_path=project_path,
            confirm_callback=self.agent.confirm_callback,
            event_callback=self._render_subagent_event,
            lsp_manager=getattr(self.agent, "lsp_manager", None),
            _plan_file_override=str(base_file) if base_file else None,
        )

        self.console.print(f"[bold]正在规划: {goal}[/bold]\n")
        result = await sub_tool.execute(goal=goal, context=context, type="plan")

        if result.error:
            self.console.print(f"[red]{result.error}[/red]")
        else:
            self.console.print(Markdown(result.output))
            # 记录本次 goal 到历史
            self._plan_history.append(goal)
            # 记录本次生成的计划文件路径（从 result 中提取）
            import re
            m = re.search(r"\[plan_file: (.+?)\]", result.output)
            if m:
                self._last_plan_file = m.group(1)

    async def _handle_exec_command(self, arg: str) -> None:
        """/exec <path> — 读取方案文件，作为上下文传给主 Agent 执行."""
        file_path = arg.strip()
        if not file_path:
            self.console.print("[dim]用法: /exec <方案文件路径>[/dim]")
            return

        from pathlib import Path as P
        target = P(file_path)
        if not target.is_file():
            self.console.print(f"[red]文件不存在: {file_path}[/red]")
            return

        content = target.read_text(encoding="utf-8")
        prompt = (
            f"请按照以下方案文件执行实施：\n\n"
            f"---\n{content}\n---\n\n"
            f"逐步执行方案中的步骤，完成后简要总结。"
        )
        await self._run_agent_stream(prompt)

    async def _handle_goal_command(self, arg: str) -> None:
        """/goal [目标|status|pause|resume|cancel] — Goal-Driven 编排模式."""
        from ..core.agent import AgentObserver
        from ..core.goal_prompt import build_goal_driven_prompt
        from ..tools.subagent import SubAgentTool

        # 子命令分派
        sub_cmd = arg.strip().split(maxsplit=1)[0].lower() if arg.strip() else ""
        if sub_cmd == "status":
            self._show_goal_status()
            return
        if sub_cmd == "pause":
            self.console.print("[yellow]如需暂停，请直接按 Ctrl+C[/yellow]")
            return
        if sub_cmd == "resume":
            await self._goal_resume()
            return
        if sub_cmd == "cancel":
            self._goal_cancel()
            return

        # 收集 goal
        goal = arg.strip()
        if not goal:
            self.console.print("[bold]Goal-Driven 模式[/bold] — 请输入目标：")
            try:
                goal = await self._prompt_session.prompt_async(
                    HTML("<b><ansigreen>目标: </ansigreen></b>"),
                )
            except (EOFError, KeyboardInterrupt):
                self.console.print("[yellow]已取消[/yellow]")
                return
            goal = goal.strip()
            if not goal:
                self.console.print("[yellow]目标为空，已取消[/yellow]")
                return

        # 收集 criteria
        self.console.print("[dim]请输入成功标准（如何判定目标已达成）：[/dim]")
        try:
            criteria = await self._prompt_session.prompt_async(
                HTML("<b><ansicyan>标准: </ansicyan></b>"),
            )
        except (EOFError, KeyboardInterrupt):
            self.console.print("[yellow]已取消[/yellow]")
            return
        criteria = criteria.strip()
        if not criteria:
            self.console.print("[yellow]标准为空，已取消[/yellow]")
            return

        # 构建 master Agent
        from pathlib import Path

        from ..core.agent import Agent
        from ..safety.loop_guard import LoopGuard
        from ..tools.base import ToolRegistry
        from ..tools.file_ops import ReadFileTool
        from ..tools.git import GitLogTool, GitStatusTool
        from ..tools.search import GrepTool, ListDirTool
        from ..tools.shell import BashTool

        project_path = str(Path(".").resolve())
        llm_client = self.agent.llm_client
        subagent_files_changed: list[str] = []

        async def _relay_subagent_event(event: AgentEvent) -> None:
            changed_path = _changed_path_from_subagent_event(event)
            if changed_path and changed_path not in subagent_files_changed:
                subagent_files_changed.append(changed_path)
            self._render_subagent_event(event)

        sub_agent_tool = SubAgentTool(
            llm_client=llm_client,
            project_path=project_path,
            system_prompt="你是一个编程助手，按指令完成任务。完成后简要说明做了什么。",
            confirm_callback=self.agent.confirm_callback,
            event_callback=_relay_subagent_event,
            lsp_manager=getattr(self.agent, "lsp_manager", None),
        )

        registry = ToolRegistry()
        registry.register(sub_agent_tool)
        registry.register(BashTool(cwd=project_path))
        registry.register(ReadFileTool())
        registry.register(GrepTool())
        registry.register(ListDirTool())
        registry.register(GitStatusTool())
        registry.register(GitLogTool())

        system_prompt = build_goal_driven_prompt(goal, criteria, project_path)
        loop_guard = LoopGuard(max_rounds=200)

        # B1: 构建 checkpoint 基础设施
        from ..longrun.checkpoint_manager import CheckpointManager
        from ..longrun.config import LongRunConfig

        longrun_config = LongRunConfig()
        checkpoint_manager = None
        manager = self.agent.ledger_manager
        if manager is not None and self.agent.git_checkpoint is not None:
            checkpoint_manager = CheckpointManager(
                checkpoint_dir=str(Path(project_path) / ".agent" / "checkpoints"),
                ledger_manager=manager,
                git_checkpoint=self.agent.git_checkpoint,
                cwd=project_path,
            )

        master_agent = Agent(
            llm_client=llm_client,
            tool_registry=registry,
            system_prompt=system_prompt,
            confirm_callback=self.agent.confirm_callback,
            loop_guard=loop_guard,
            project_path=project_path,
            git_checkpoint=self.agent.git_checkpoint,
            checkpoint_manager=checkpoint_manager,
            longrun_config=longrun_config,
        )

        # 创建 Ledger（如果有 manager）
        ledger = None
        if manager is not None:
            from ..longrun.ledger_types import TaskRunStatus

            ledger = manager.create(goal=goal, budget=500_000)
            ledger.status = TaskRunStatus.RUNNING
            ledger.current_phase = "execution"
            ledger.current_task_id = "goal"
            ledger.task_graph_snapshot.setdefault("nodes", {})["goal"] = {
                "id": "goal", "description": goal, "status": "running",
            }
            ledger.token_budget_remaining = max(
                0, ledger.token_budget - ledger.total_tokens_used
            )
            manager.save(ledger)
            master_agent.ledger = ledger
            master_agent.ledger_manager = manager
            # B3: 持久化 criteria 到 Ledger
            ledger.task_graph_snapshot["criteria"] = criteria
            manager.save(ledger)
            self.console.print(
                f"[green]Ledger 已创建: {ledger.task_id[:8]}[/green]"
            )

        # 挂 Observer：监听 SubAgent tool 调用结果，即时写入 Ledger
        if ledger is not None and manager is not None:
            from datetime import datetime, UTC
            from ..longrun.ledger_types import CompletedTaskRecord, FailedAttemptRecord

            class GoalLedgerObserver(AgentObserver):
                def __init__(self, ledger, manager, files_changed_buffer: list[str]):
                    self._ledger = ledger
                    self._manager = manager
                    self._files_changed_buffer = files_changed_buffer
                    self._sub_count = 0

                def on_tool_call(self, name: str, args: dict, result) -> None:
                    if name != "SubAgent":
                        return
                    from ..tools.base import ToolResult as TR
                    if not isinstance(result, TR):
                        return
                    self._sub_count += 1
                    task_id = f"sub-{self._sub_count}"
                    goal_desc = args.get("goal", "")[:200]
                    output = result.output or ""
                    stop_reason = "unknown"
                    first_line = output.splitlines()[0] if output else ""
                    if first_line.startswith("[stop_reason:") and first_line.endswith("]"):
                        stop_reason = first_line[len("[stop_reason:"):-1].strip()
                    token_count = 0
                    for line in output.splitlines()[:3]:
                        if line.startswith("[usage:") and line.endswith("]"):
                            for part in line[len("[usage:"):-1].strip().split():
                                if part.startswith("total_tokens="):
                                    try:
                                        token_count = int(part.split("=", 1)[1])
                                    except ValueError:
                                        token_count = 0
                                    break
                    step_start = self._ledger.total_steps
                    step_end = step_start + 1
                    sub_failed = result.is_error or stop_reason not in {"ok", "unknown"}
                    files_changed = list(self._files_changed_buffer)
                    self._files_changed_buffer.clear()

                    if sub_failed:
                        reason = result.error or f"stop_reason={stop_reason}"
                        self._ledger.failed_attempts.append(FailedAttemptRecord(
                            task_id=task_id,
                            artifact_id="",
                            approach_description=goal_desc,
                            failure_reason=reason,
                            step_number=step_end,
                        ))
                    else:
                        self._ledger.completed_tasks.append(CompletedTaskRecord(
                            task_id=task_id,
                            artifact_id="",
                            description=goal_desc,
                            self_summary=output[:300],
                            files_changed=files_changed,
                            verification_passed=False,
                            confidence="DONE",
                            step_number_start=step_start,
                            step_number_end=step_end,
                            token_count=token_count,
                            timestamp=datetime.now(UTC),
                        ))
                    self._ledger.total_steps = step_end
                    self._ledger.total_tokens_used += token_count
                    self._ledger.token_budget_remaining = max(
                        0, self._ledger.token_budget - self._ledger.total_tokens_used
                    )
                    self._manager.save(self._ledger)

                def on_llm_call(self, tokens_in: int, tokens_out: int, model: str) -> None:
                    self._ledger.total_tokens_used += tokens_in + tokens_out
                    self._ledger.token_budget_remaining = max(
                        0, self._ledger.token_budget - self._ledger.total_tokens_used
                    )
                    self._manager.save(self._ledger)

            master_agent.observers.append(
                GoalLedgerObserver(ledger, manager, subagent_files_changed)
            )

        self.console.print(
            f"\n[bold]Goal-Driven 模式启动[/bold]\n"
            f"[dim]目标: {goal}[/dim]\n"
            f"[dim]标准: {criteria}[/dim]\n"
        )

        first_prompt = (
            f"开始执行。当前项目根目录是 {project_path}。"
            "先用相对路径检查当前项目状态，然后决定第一步。"
        )
        await self._run_goal_loop(
            goal=goal,
            criteria=criteria,
            ledger=ledger,
            manager=manager,
            master_agent=master_agent,
            checkpoint_manager=checkpoint_manager,
            longrun_config=longrun_config,
            initial_prompt=first_prompt,
        )

    async def _run_goal_loop(
        self,
        goal: str,
        criteria: str,
        ledger,
        manager=None,
        master_agent=None,
        checkpoint_manager=None,
        longrun_config=None,
        initial_prompt: str = "",
        restore_messages: list[dict] | None = None,
    ) -> None:
        """Goal 模式状态机循环 — /goal 和 /goal resume 共用."""
        from ..core.agent import AgentEventType

        # 如果是 resume 调用（没有传 master_agent），需要自己构建
        if master_agent is None:
            from pathlib import Path
            from ..core.agent import Agent
            from ..core.goal_prompt import build_goal_driven_prompt
            from ..longrun.checkpoint_manager import CheckpointManager
            from ..longrun.config import LongRunConfig
            from ..safety.loop_guard import LoopGuard
            from ..tools.base import ToolRegistry
            from ..tools.file_ops import ReadFileTool
            from ..tools.git import GitLogTool, GitStatusTool
            from ..tools.search import GrepTool, ListDirTool
            from ..tools.shell import BashTool
            from ..tools.subagent import SubAgentTool

            project_path = str(Path(".").resolve())
            llm_client = self.agent.llm_client

            async def _relay(event) -> None:
                self._render_subagent_event(event)

            sub_agent_tool = SubAgentTool(
                llm_client=llm_client,
                project_path=project_path,
                system_prompt="你是一个编程助手，按指令完成任务。完成后简要说明做了什么。",
                confirm_callback=self.agent.confirm_callback,
                event_callback=_relay,
                lsp_manager=getattr(self.agent, "lsp_manager", None),
            )
            registry = ToolRegistry()
            registry.register(sub_agent_tool)
            registry.register(BashTool(cwd=project_path))
            registry.register(ReadFileTool())
            registry.register(GrepTool())
            registry.register(ListDirTool())
            registry.register(GitStatusTool())
            registry.register(GitLogTool())

            system_prompt = build_goal_driven_prompt(goal, criteria, project_path)
            if longrun_config is None:
                longrun_config = LongRunConfig()
            if manager is None:
                manager = self.agent.ledger_manager
            if checkpoint_manager is None and manager is not None and self.agent.git_checkpoint is not None:
                checkpoint_manager = CheckpointManager(
                    checkpoint_dir=str(Path(project_path) / ".agent" / "checkpoints"),
                    ledger_manager=manager,
                    git_checkpoint=self.agent.git_checkpoint,
                    cwd=project_path,
                )

            master_agent = Agent(
                llm_client=llm_client,
                tool_registry=registry,
                system_prompt=system_prompt,
                confirm_callback=self.agent.confirm_callback,
                loop_guard=LoopGuard(max_rounds=200),
                project_path=project_path,
                git_checkpoint=self.agent.git_checkpoint,
                checkpoint_manager=checkpoint_manager,
                longrun_config=longrun_config,
            )
            if ledger is not None and manager is not None:
                master_agent.ledger = ledger
                master_agent.ledger_manager = manager
                from ..longrun.ledger_types import TaskRunStatus as _TRS
                ledger.status = _TRS.RUNNING
                ledger.current_phase = "execution"
                manager.save(ledger)

        # 注入恢复对话历史
        if restore_messages:
            for msg in restore_messages:
                role = msg.get("role", "")
                content = msg.get("content", "")
                if role == "user":
                    master_agent.inject_initial_message(content)
                elif role == "assistant":
                    from ..llm.base import Message
                    master_agent.conversation.append(Message.assistant(content))

        def _set_goal_node_status(status: str) -> None:
            if ledger is None:
                return
            nodes = ledger.task_graph_snapshot.get("nodes", {})
            if "goal" in nodes:
                nodes["goal"]["status"] = status

        def _mark_goal_milestones_reached() -> None:
            if ledger is None:
                return
            for milestone in ledger.milestones:
                if "goal" in milestone.associated_task_ids:
                    milestone.status = "REACHED"
                    milestone.actual_step = ledger.total_steps

        def _refresh_budget() -> None:
            if ledger is None:
                return
            ledger.token_budget_remaining = max(
                0, ledger.token_budget - ledger.total_tokens_used
            )

        # 流式执行 master agent（状态机循环）
        import re
        import sys
        from time import monotonic

        started_at = monotonic()
        run_finished = False
        finish_stop_reason = ""
        run_interrupted = False
        run_error: Exception | None = None
        goal_status = "active"
        text_accumulator = ""

        async def _run_one_turn(prompt: str) -> str:
            """执行 master agent 一轮，返回文本输出."""
            nonlocal run_finished, finish_stop_reason, run_interrupted, run_error
            nonlocal text_accumulator
            text_accumulator = ""
            try:
                async for event in master_agent.run_stream(prompt):
                    if event.type == AgentEventType.TEXT_DELTA:
                        sys.stdout.write(event.content)
                        sys.stdout.flush()
                        text_accumulator += event.content
                    elif event.type == AgentEventType.TOOL_CALL_START:
                        sys.stdout.write("\n")
                        sys.stdout.flush()
                        self._render_tool_call_start(event.tool_call)
                    elif event.type == AgentEventType.TOOL_CALL_END:
                        self._render_tool_call_args(event.tool_call)
                    elif event.type == AgentEventType.TOOL_RESULT:
                        self._render_tool_result(event.tool_call, event.tool_result)
                    elif event.type == AgentEventType.FINISH:
                        run_finished = True
                        finish_stop_reason = event.content or "ok"
                        sys.stdout.write("\n")
                        sys.stdout.flush()
                        if event.usage:
                            self._render_usage_brief(event.usage)
            except (KeyboardInterrupt, asyncio.CancelledError):
                run_interrupted = True
                self.console.print("\n[yellow]已中断（Goal 已暂停）[/yellow]")
            except Exception as e:
                run_error = e
                self.console.print(f"\n[red]错误: {type(e).__name__}: {e}[/red]")
            return text_accumulator

        def _parse_goal_status(text: str) -> str:
            matches = list(
                re.finditer(
                    r"^\s*\[?\s*goal_status\s*:\s*(active|complete|blocked)\b\s*\]?\s*$",
                    text,
                    flags=re.IGNORECASE | re.MULTILINE,
                )
            )
            return matches[-1].group(1).lower() if matches else "active"

        # 首轮
        output = await _run_one_turn(initial_prompt)
        goal_status = _parse_goal_status(output)

        # 状态机循环
        while goal_status == "active" and not run_interrupted and run_error is None:
            output = await _run_one_turn("继续执行下一步。")
            if run_interrupted or run_error is not None:
                break
            goal_status = _parse_goal_status(output)

        if goal_status == "blocked" and not run_interrupted and run_error is None:
            self.console.print(
                "\n[yellow]Goal 被阻塞，等待用户输入。"
                "请提供指示后使用 /goal resume 继续。[/yellow]"
            )

        # 更新 Ledger 状态
        if manager is not None and ledger is not None:
            from ..longrun.ledger_types import TaskRunStatus
            ledger.total_wall_time_seconds += monotonic() - started_at
            if run_interrupted:
                ledger.status = TaskRunStatus.PAUSED
                ledger.current_phase = "paused"
                ledger.current_task_id = "goal"
                _set_goal_node_status("running")
            elif run_error is not None:
                ledger.status = TaskRunStatus.FAILED
                ledger.current_phase = "failed"
                ledger.current_task_id = None
                _set_goal_node_status("failed")
            elif goal_status == "complete":
                ledger.status = TaskRunStatus.COMPLETED
                ledger.current_phase = "done"
                ledger.current_task_id = None
                _set_goal_node_status("completed")
                _mark_goal_milestones_reached()
                for record in ledger.completed_tasks:
                    record.verification_passed = True
            elif goal_status == "blocked":
                ledger.status = TaskRunStatus.PAUSED
                ledger.current_phase = "blocked"
                ledger.current_task_id = "goal"
                _set_goal_node_status("running")
            else:
                ledger.status = TaskRunStatus.FAILED
                ledger.current_phase = "failed"
                ledger.current_task_id = None
                _set_goal_node_status("failed")
            _refresh_budget()
            manager.save(ledger)

        # B2: 中断时 best-effort 保存 checkpoint
        if run_interrupted and checkpoint_manager is not None and ledger is not None:
            from ..longrun.session_state import CheckpointTrigger
            try:
                msg_dicts: list[dict] = []
                for m in master_agent.messages:
                    entry: dict = {"role": m.role.value if hasattr(m.role, "value") else str(m.role)}
                    if isinstance(m.content, str):
                        entry["content"] = m.content[:500]
                    else:
                        entry["content"] = str(m.content)[:500]
                    msg_dicts.append(entry)
                await checkpoint_manager.save_checkpoint(
                    ledger=ledger,
                    trigger=CheckpointTrigger.USER_PAUSE,
                    config=longrun_config,
                    current_task_id="goal",
                    recent_messages=msg_dicts,
                )
                self.console.print("[green]Checkpoint 已保存[/green]")
            except Exception:
                pass

        self.console.print("\n[dim]Goal-Driven 模式结束[/dim]")

    def _show_goal_status(self) -> None:
        """/goal status — 展示当前 Goal 的 Ledger 进展."""
        manager = self.agent.ledger_manager
        if manager is None:
            self.console.print("[dim]Ledger 未启用[/dim]")
            return
        ledgers = manager.list_all()
        if not ledgers:
            self.console.print("[dim]暂无 Goal 记录[/dim]")
            return
        latest = ledgers[0]
        lines = [
            f"Task ID: {latest.task_id[:8]}",
            f"状态: {latest.status.value}",
            f"阶段: {latest.current_phase}",
            f"已完成子任务: {len(latest.completed_tasks)}",
            f"失败尝试: {len(latest.failed_attempts)}",
            f"总步骤: {latest.total_steps}",
            f"Token 用量: {latest.total_tokens_used:,}",
            f"耗时: {latest.total_wall_time_seconds:.1f}s",
        ]
        self.console.print(Panel(
            "\n".join(lines),
            title="[bold]Goal Status[/bold]",
            border_style="cyan",
        ))

    async def _goal_resume(self) -> None:
        """/goal resume — 基于 checkpoint + Ledger 恢复暂停的 Goal."""
        manager = self.agent.ledger_manager
        if manager is None:
            self.console.print("[red]Ledger 未启用[/red]")
            return
        from ..longrun.ledger_types import TaskRunStatus
        ledgers = manager.list_all()
        paused = [l for l in ledgers if l.status == TaskRunStatus.PAUSED]
        if not paused:
            self.console.print("[dim]没有暂停中的 Goal[/dim]")
            return
        ledger = manager.load(paused[0].task_id)

        # 取回 goal + criteria
        goal = ledger.task_graph_snapshot.get("original_goal", "")
        if not goal:
            goal = ledger.goal
        criteria = ledger.task_graph_snapshot.get("criteria", "")
        if not goal:
            self.console.print("[red]无法恢复：Goal 信息缺失[/red]")
            return

        # 查找最新 checkpoint
        from pathlib import Path
        from ..longrun.checkpoint_manager import CheckpointManager
        from ..longrun.config import LongRunConfig

        project_path = str(Path(".").resolve())
        checkpoint_manager = None
        restore_messages: list[dict] | None = None

        if self.agent.git_checkpoint is not None:
            checkpoint_manager = CheckpointManager(
                checkpoint_dir=str(Path(project_path) / ".agent" / "checkpoints"),
                ledger_manager=manager,
                git_checkpoint=self.agent.git_checkpoint,
                cwd=project_path,
            )
            checkpoints = checkpoint_manager.list_checkpoints(ledger.task_id)
            if checkpoints:
                latest = checkpoints[0]
                try:
                    state = checkpoint_manager.load_checkpoint(ledger.task_id, latest.id)
                    restore_messages = state.recent_messages_full
                    self.console.print(
                        f"[green]已加载 checkpoint {latest.id[:8]}[/green]"
                    )
                except Exception:
                    pass

        # 构建恢复 prompt
        resume_summary = manager.get_summary_for_resume(ledger)
        if criteria:
            initial_prompt = (
                f"{resume_summary}\n\n"
                f"成功标准: {criteria}\n\n"
                f"请从中断处继续执行。不需要重做已完成的工作。"
            )
        else:
            initial_prompt = (
                f"{resume_summary}\n\n"
                f"请从中断处继续执行。不需要重做已完成的工作。"
            )

        self.console.print(f"[dim]恢复 Goal: {goal}[/dim]")
        await self._run_goal_loop(
            goal=goal,
            criteria=criteria,
            ledger=ledger,
            checkpoint_manager=checkpoint_manager,
            initial_prompt=initial_prompt,
            restore_messages=restore_messages,
        )

    def _goal_cancel(self) -> None:
        """/goal cancel — 取消当前 Goal."""
        manager = self.agent.ledger_manager
        if manager is None:
            self.console.print("[dim]Ledger 未启用[/dim]")
            return
        ledgers = manager.list_all()
        from ..longrun.ledger_types import TaskRunStatus
        active = [l for l in ledgers if l.status in (TaskRunStatus.RUNNING, TaskRunStatus.PAUSED)]
        if not active:
            self.console.print("[dim]没有活跃的 Goal[/dim]")
            return
        ledger = manager.load(active[0].task_id)
        ledger.status = TaskRunStatus.FAILED
        ledger.current_phase = "cancelled"
        manager.save(ledger)
        self.console.print(f"[green]Goal {ledger.task_id[:8]} 已取消[/green]")

    async def _run_agent_stream(self, user_input: str) -> None:
        """流式执行 Agent 并实时渲染输出."""
        self.console.print()  # 空行分隔

        text_buffer = ""
        in_text = False

        try:
            async for event in self.agent.run_stream(user_input):
                if event.type == AgentEventType.TEXT_DELTA:
                    # 逐 token 打印文本
                    if not in_text:
                        in_text = True
                    sys.stdout.write(event.content)
                    sys.stdout.flush()
                    text_buffer += event.content

                elif event.type == AgentEventType.TOOL_CALL_START:
                    # 结束之前的文本块
                    if in_text:
                        sys.stdout.write("\n")
                        sys.stdout.flush()
                        in_text = False
                        text_buffer = ""

                    self._render_tool_call_start(event.tool_call)

                elif event.type == AgentEventType.TOOL_CALL_END:
                    self._render_tool_call_args(event.tool_call)

                elif event.type == AgentEventType.TOOL_RESULT:
                    self._render_tool_result(event.tool_call, event.tool_result)

                elif event.type == AgentEventType.FINISH:
                    if in_text:
                        sys.stdout.write("\n")
                        sys.stdout.flush()
                        in_text = False
                    # 显示 checkpoint 信息
                    cp = self.agent.git_checkpoint
                    if cp is not None:
                        cps = await cp.list_checkpoints()
                        if cps:
                            latest = cps[0]
                            self.console.print(
                                f"  [dim]checkpoint: {latest.message} "
                                f"({latest.commit_hash[:8]})[/dim]"
                            )
                    if event.usage:
                        self._render_usage_brief(event.usage)

        except KeyboardInterrupt:
            self.console.print("\n[yellow]已中断[/yellow]")
        except Exception as e:
            if in_text:
                sys.stdout.write("\n")
                sys.stdout.flush()
            self.console.print(f"\n[red]错误: {type(e).__name__}: {e}[/red]")

    def _render_subagent_event(self, event: AgentEvent) -> None:
        """渲染 SubAgent 内部事件，让 Goal 模式下的执行过程可见."""
        if event.type == AgentEventType.TOOL_CALL_START:
            self._render_tool_call_start(event.tool_call, prefix="    ↳ ")
        elif event.type == AgentEventType.TOOL_CALL_END:
            self._render_tool_call_args(event.tool_call, prefix="      ")
        elif event.type == AgentEventType.TOOL_RESULT:
            self._render_tool_result(event.tool_call, event.tool_result, prefix="      ")
        elif event.type == AgentEventType.FINISH and event.usage:
            self.console.print(
                f"      [dim]SubAgent tokens: "
                f"{event.usage.input_tokens:,} in / "
                f"{event.usage.output_tokens:,} out[/dim]"
            )

    def _render_tool_call_start(
        self,
        tool_call: ToolCall | None,
        prefix: str = "  ",
    ) -> None:
        """渲染工具调用开始."""
        if not tool_call:
            return
        self.console.print(
            f"{prefix}[bold yellow]⚡ {tool_call.name}[/bold yellow]",
            highlight=False,
        )

    def _render_tool_call_args(
        self,
        tool_call: ToolCall | None,
        prefix: str = "    ",
    ) -> None:
        """渲染工具调用的参数."""
        if not tool_call:
            return
        args = tool_call.arguments
        # 为常见工具做特殊展示
        if tool_call.name == "Bash" and "command" in args:
            self.console.print(f"{prefix}[dim]$ {args['command']}[/dim]")
        elif tool_call.name == "ReadFile" and "path" in args:
            extra = ""
            if "start_line" in args:
                extra = f" (行 {args['start_line']}-{args.get('end_line', '末尾')})"
            self.console.print(f"{prefix}[dim]📄 {args['path']}{extra}[/dim]")
        elif tool_call.name == "WriteFile" and "path" in args:
            content = args.get("content", "")
            lines_count = len(content.splitlines())
            self.console.print(f"{prefix}[dim]✏️  {args['path']} ({lines_count} 行)[/dim]")
        elif tool_call.name == "Grep" and "pattern" in args:
            path = args.get("path", ".")
            self.console.print(f"{prefix}[dim]🔍 '{args['pattern']}' in {path}[/dim]")
        elif tool_call.name == "ListDir":
            path = args.get("path", ".")
            self.console.print(f"{prefix}[dim]📁 {path}[/dim]")
        elif tool_call.name == "GitStatus":
            path = args.get("path", ".")
            self.console.print(f"{prefix}[dim]📋 git status ({path})[/dim]")
        elif tool_call.name == "GitDiff":
            staged = args.get("staged", False)
            label = "staged" if staged else "unstaged"
            self.console.print(f"{prefix}[dim]📋 git diff ({label})[/dim]")
        elif tool_call.name == "GitCommit":
            msg = args.get("message", "")
            self.console.print(f"{prefix}[dim]📝 git commit -m \"{msg}\"[/dim]")
        elif tool_call.name == "GitLog":
            count = args.get("count", 10)
            self.console.print(f"{prefix}[dim]📋 git log -{count}[/dim]")
        elif tool_call.name == "SubAgent":
            agent_type = args.get("type", "coder")
            goal_text = args.get("goal", "")
            self.console.print(f"{prefix}[dim]🤖 SubAgent({agent_type}): {goal_text}[/dim]")
        else:
            compact = json.dumps(args, ensure_ascii=False)
            if len(compact) > 120:
                compact = compact[:117] + "..."
            self.console.print(f"{prefix}[dim]{compact}[/dim]")

    def _render_tool_result(
        self,
        tool_call: ToolCall | None,
        result: Any,
        prefix: str = "    ",
    ) -> None:
        """渲染工具执行结果（简要）."""
        if result is None:
            return

        from ..tools.base import ToolResult as ExecToolResult

        if not isinstance(result, ExecToolResult):
            return

        if result.is_error:
            error_text = result.error or "未知错误"
            if len(error_text) > 200:
                error_text = error_text[:197] + "..."
            self.console.print(f"{prefix}[red]✗ {error_text}[/red]")
        else:
            output = result.output
            lines = output.splitlines()
            if len(lines) > 5:
                preview = "\n".join(lines[:3])
                self.console.print(f"{prefix}[green]✓[/green] [dim]({len(lines)} 行输出)[/dim]")
            elif output:
                # 短输出直接显示
                if len(output) > 200:
                    output = output[:197] + "..."
                self.console.print(f"{prefix}[green]✓[/green] [dim]{output}[/dim]")
            else:
                self.console.print(f"{prefix}[green]✓[/green] [dim](空输出)[/dim]")

    def _render_usage_brief(self, usage: TokenUsage) -> None:
        """在回复末尾简要显示 token 用量."""
        self.console.print(
            f"\n[dim]tokens: {usage.input_tokens:,} in / {usage.output_tokens:,} out[/dim]"
        )

    def _print_welcome(self) -> None:
        """打印欢迎信息."""
        model = self.agent.llm_client.model
        self.console.print(Panel(
            f"[bold]Mini Code Agent[/bold]\n"
            f"模型: {model}\n"
            f"输入消息开始对话，特殊命令：\n"
            f"  /quit         — 退出\n"
            f"  /clear        — 清空对话\n"
            f"  /cost         — 查看 token 消耗\n"
            f"  /model        — 切换模型\n"
            f"  /memory       — 查看记忆状态\n"
            f"  /save         — 保存信息到项目记忆\n"
            f"  /goal         — 启动 Goal-Driven 编排模式\n"
            f"  /plan         — 派发 plan 子 Agent 规划方案\n"
            f"  /exec         — 执行方案文件\n"
            f"  /undo         — 回滚最近一次 Agent 修改\n"
            f"  /checkpoints  — 列出所有 checkpoint\n"
            f"  /diff         — 查看 Agent 的所有修改\n"
            f"  Ctrl+C  — 中断当前操作\n"
            f"  多行输入：Alt+Enter 换行，Enter 提交",
            border_style="blue",
        ))
