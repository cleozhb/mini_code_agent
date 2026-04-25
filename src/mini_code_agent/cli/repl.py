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
from ..tools.base import ToolRegistry
from .confirm import confirm_tool_call

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..longrun.ledger_manager import TaskLedgerManager

# ---------------------------------------------------------------------------
# Token 费用估算（每百万 token 美元价格, 可按模型调整）
# ---------------------------------------------------------------------------

_PRICE_TABLE: dict[str, tuple[float, float]] = {
    # (input_price_per_1m, output_price_per_1m)
    "gpt-4o": (2.5, 10.0),
    "gpt-4o-mini": (0.15, 0.6),
    "gpt-4-turbo": (10.0, 30.0),
    "claude-sonnet-4-20250514": (3.0, 15.0),
    "claude-opus-4-20250514": (15.0, 75.0),
    "deepseek-chat": (0.27, 1.10),
    "deepseek-reasoner": (0.55, 2.19),
}


def _estimate_cost(model: str, usage: TokenUsage) -> float | None:
    """估算费用（美元），模型不在表中返回 None."""
    prices = _PRICE_TABLE.get(model)
    if not prices:
        # 尝试前缀匹配
        for key, val in _PRICE_TABLE.items():
            if model.startswith(key):
                prices = val
                break
    if not prices:
        return None
    input_cost = usage.input_tokens / 1_000_000 * prices[0]
    output_cost = usage.output_tokens / 1_000_000 * prices[1]
    return input_cost + output_cost


# ---------------------------------------------------------------------------
# REPL
# ---------------------------------------------------------------------------


class REPL:
    """Agent 交互式 REPL."""

    def __init__(
        self,
        agent: Agent,
        console: Console | None = None,
        graph_planner: object | None = None,
        graph_executor: object | None = None,
        graph_mode: bool = False,
        pending_graph: object | None = None,
        long_run_deferred: bool = False,
        token_budget: int = 500_000,
    ) -> None:
        self.agent = agent
        self.console = console or Console()
        self.graph_planner = graph_planner
        self.graph_executor = graph_executor
        self.graph_mode = graph_mode
        self._current_graph = None  # 当前 TaskGraph（用于 /graph 查看）
        self._pending_graph = pending_graph  # --long-run / --resume 预生成的图
        self._long_run_deferred = long_run_deferred  # --long-run 不带目标
        self._token_budget = token_budget

        # 斜杠命令补全
        self._completer = WordCompleter(
            [
                "/quit", "/exit", "/q", "/clear", "/cost", "/model",
                "/memory", "/save", "/plan",
                "/undo", "/checkpoints", "/diff",
                "/graph", "/graph-export",
                "/status", "/ledger",
                "/longrun",
            ],
            sentence=True,  # 整条匹配，不拆词
        )

        # prompt_toolkit session，带文件历史和命令补全
        self._prompt_session = PromptSession[str](
            history=FileHistory(".agent_history"),
            completer=self._completer,
        )

    async def run(self) -> None:
        """启动 REPL 主循环."""
        self._print_welcome()

        # --long-run / --resume：有预生成的 TaskGraph，直接执行
        if self._pending_graph is not None:
            await self._execute_pending_graph()

        # --long-run (无目标)：交互式输入后进入长程任务流程
        elif self._long_run_deferred:
            self._long_run_deferred = False
            await self._prompt_and_start_longrun()

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

            # Graph mode 用非流式执行；Plan mode 用非流式；否则走流式
            if self.graph_mode and self.graph_planner is not None:
                await self._run_agent_graph(user_input)
            elif self.agent.plan_mode and self.agent.planner is not None:
                await self._run_agent_plan(user_input)
            else:
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

        if command == "/plan":
            plan_arg = parts[1].strip().lower() if len(parts) > 1 else ""
            self._toggle_plan_mode(plan_arg)
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

        if command == "/graph":
            graph_arg = parts[1].strip().lower() if len(parts) > 1 else ""
            self._handle_graph_command(graph_arg)
            return True

        if command == "/graph-export":
            self._export_graph_mermaid()
            return True

        if command == "/status":
            self._show_ledger_status()
            return True

        if command == "/ledger":
            ledger_arg = parts[1].strip() if len(parts) > 1 else ""
            self._handle_ledger_command(ledger_arg)
            return True

        if command == "/longrun":
            longrun_arg = parts[1].strip() if len(parts) > 1 else ""
            await self._handle_longrun_command(longrun_arg)
            return True

        self.console.print(f"[red]未知命令: {command}[/red]")
        self.console.print(
            "[dim]可用命令: /quit  /clear  /cost  /model  /memory  /save  /plan"
            "  /undo  /checkpoints  /diff  /graph  /graph-export"
            "  /status  /ledger  /longrun[/dim]"
        )
        return True

    def _toggle_plan_mode(self, arg: str) -> None:
        """开关 plan mode."""
        if self.agent.planner is None:
            self.console.print("[red]Plan mode 未初始化（缺少 Planner）[/red]")
            return

        if arg in ("on", "enable", "true", "1"):
            self.agent.plan_mode = True
        elif arg in ("off", "disable", "false", "0"):
            self.agent.plan_mode = False
        elif arg == "":
            self.agent.plan_mode = not self.agent.plan_mode
        else:
            self.console.print(f"[red]无法识别的参数: {arg}[/red]")
            self.console.print("[dim]用法: /plan [on|off][/dim]")
            return

        status = "开启" if self.agent.plan_mode else "关闭"
        color = "green" if self.agent.plan_mode else "yellow"
        self.console.print(f"[{color}]Plan mode 已{status}[/{color}]")

    def _handle_graph_command(self, arg: str) -> None:
        """处理 /graph 命令：无参查看当前图，on/off 切换模式."""
        if arg in ("on", "enable", "true", "1"):
            if self.graph_planner is None:
                self.console.print("[red]Graph mode 未初始化（缺少 GraphPlanner）[/red]")
                return
            self.graph_mode = True
            self.console.print("[green]Graph mode 已开启[/green]")
        elif arg in ("off", "disable", "false", "0"):
            self.graph_mode = False
            self.console.print("[yellow]Graph mode 已关闭[/yellow]")
        elif arg == "":
            # 无参数：如果有当前图则展示，否则显示开关状态
            if self._current_graph is not None:
                from .graph_display import render_graph_table
                render_graph_table(self._current_graph, self.console)
            else:
                status = "开启" if self.graph_mode else "关闭"
                self.console.print(f"[dim]Graph mode: {status}[/dim]")
                self.console.print("[dim]用法: /graph [on|off]  — 无任务图时切换模式[/dim]")
        else:
            self.console.print(f"[red]无法识别的参数: {arg}[/red]")
            self.console.print("[dim]用法: /graph [on|off][/dim]")

    def _export_graph_mermaid(self) -> None:
        """导出当前 TaskGraph 为 Mermaid 格式."""
        if self._current_graph is None:
            self.console.print("[dim]当前没有任务图可导出[/dim]")
            return
        from .graph_display import render_mermaid
        render_mermaid(self._current_graph, self.console)

    def _show_ledger_status(self) -> None:
        """展示 ledger 当前状态（/status 命令）."""
        ledger = self.agent.ledger
        manager = self.agent.ledger_manager
        if ledger is None or manager is None:
            self.console.print("[dim]当前没有活跃的 Task Ledger[/dim]")
            return

        stats = manager.get_stats(ledger)
        summary = manager.build_context_summary(ledger)

        lines = [
            f"Task ID: {stats['task_id']}",
            f"目标: {stats['goal']}",
            f"状态: {stats['status']}",
            f"阶段: {stats['current_phase']}",
            f"当前子任务: {stats['current_task_id'] or '(无)'}",
            f"完成: {stats['completion_rate']}",
            f"失败尝试: {stats['failed_attempts']}",
            f"步数: {stats['total_steps']}",
            f"Token: {stats['total_tokens_used']:,} / {stats['token_budget']:,}"
            f" (剩余 {stats['token_budget_remaining']:,})",
            f"平均 token/任务: {stats['avg_tokens_per_task']:,}",
            f"墙钟: {stats['total_wall_time_seconds']:.1f}s",
            f"问题: {stats['issues_open']} 个未解决, {stats['issues_resolved']} 个已解决",
            f"决策: {stats['decisions_count']} 个",
            f"里程碑: {stats['milestones_reached']}/{stats['milestones_total']}",
        ]

        self.console.print(Panel(
            "\n".join(lines),
            title="[bold]Task Ledger 状态[/bold]",
            border_style="cyan",
        ))

        if summary:
            self.console.print()
            self.console.print(Panel(
                summary,
                title="[bold]上下文摘要[/bold]",
                border_style="dim",
            ))

    def _handle_ledger_command(self, arg: str) -> None:
        """处理 /ledger 命令."""
        ledger = self.agent.ledger
        manager = self.agent.ledger_manager
        if ledger is None or manager is None:
            self.console.print("[dim]当前没有活跃的 Task Ledger[/dim]")
            return

        if not arg:
            # 打印完整 JSON
            data = ledger.to_dict()
            self.console.print_json(json.dumps(data, ensure_ascii=False, indent=2))
            return

        parts = arg.split(maxsplit=1)
        subcmd = parts[0].lower()

        if subcmd == "export":
            export_path = parts[1].strip() if len(parts) > 1 else ""
            if not export_path:
                self.console.print("[dim]用法: /ledger export <path>[/dim]")
                return
            data = ledger.to_dict()
            try:
                from pathlib import Path
                p = Path(export_path)
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(
                    json.dumps(data, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                self.console.print(f"[green]Ledger 已导出到: {export_path}[/green]")
            except Exception as e:
                self.console.print(f"[red]导出失败: {e}[/red]")

        elif subcmd == "history":
            entries = manager.get_history(ledger.task_id)
            if not entries:
                self.console.print("[dim]暂无历史记录[/dim]")
                return
            lines = []
            for i, entry in enumerate(entries):
                lines.append(
                    f"  {i + 1}. [{entry['timestamp']}] "
                    f"status={entry['status']} "
                    f"phase={entry['current_phase']} "
                    f"tasks={entry['completed_tasks']} "
                    f"tokens={entry['total_tokens_used']:,} "
                    f"steps={entry['total_steps']}"
                )
            self.console.print(Panel(
                "\n".join(lines),
                title=f"[bold]Ledger 变更历史（最近 {len(entries)} 条）[/bold]",
                border_style="cyan",
            ))
        else:
            self.console.print(f"[red]未知子命令: {subcmd}[/red]")
            self.console.print("[dim]用法: /ledger [export <path> | history][/dim]")

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
            self.console.print("[dim]暂无 checkpoint[/dim]")
            return

        lines = []
        for i, c in enumerate(checkpoints):
            lines.append(
                f"  {i + 1}. [{c.commit_hash[:8]}] {c.message}  "
                f"[dim]({c.timestamp})[/dim]"
            )

        self.console.print(Panel(
            "\n".join(lines),
            title=f"[bold]Agent Checkpoints ({len(checkpoints)})[/bold]",
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
        cost = _estimate_cost(model, usage)

        lines = [
            f"模型: {model}",
            f"输入 tokens: {usage.input_tokens:,}",
            f"输出 tokens: {usage.output_tokens:,}",
            f"总计 tokens: {usage.total_tokens:,}",
        ]
        if cost is not None:
            lines.append(f"估算费用: ${cost:.4f}")
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

    async def _run_agent_graph(self, user_input: str) -> None:
        """Graph mode：生成 Task Graph → 展示给用户 → 按 DAG 执行."""
        self.console.print()
        self.console.print("[dim]进入 Graph mode，正在生成任务图...[/dim]")

        from .graph_display import render_graph_table, render_graph_result

        try:
            graph = await self.graph_planner.plan_as_graph(user_input)
        except Exception as e:
            self.console.print(f"\n[red]生成任务图失败: {type(e).__name__}: {e}[/red]")
            self.console.print("[dim]回退到普通模式执行[/dim]")
            await self._run_agent_stream(user_input)
            return

        # 展示任务图
        render_graph_table(graph, self.console)
        self._current_graph = graph

        # 简单确认
        self.console.print(
            "\n[bold]操作：[/bold] "
            "[green]y[/green]=执行  "
            "[red]n[/red]=放弃"
        )
        try:
            ans = await self._prompt_session.prompt_async(
                HTML("<b>执行任务图？ [y/n]: </b>")
            )
        except (EOFError, KeyboardInterrupt):
            self.console.print("[yellow]已取消[/yellow]")
            return

        if (ans or "").strip().lower() not in ("y", "yes", ""):
            self.console.print("[yellow]已放弃[/yellow]")
            return

        # 执行
        try:
            result = await self.graph_executor.execute(graph, self.agent)
        except KeyboardInterrupt:
            self.console.print("\n[yellow]已中断[/yellow]")
            return
        except Exception as e:
            self.console.print(f"\n[red]执行错误: {type(e).__name__}: {e}[/red]")
            return

        # 展示结果
        self._current_graph = result.graph
        render_graph_result(result, self.console)

    async def _execute_pending_graph(self) -> None:
        """执行 --long-run / --resume 预生成的 TaskGraph（跳过 planner 生成阶段）."""
        graph = self._pending_graph
        self._pending_graph = None  # 只执行一次
        self._current_graph = graph
        await self._execute_graph_with_ledger(graph)

    async def _execute_graph_with_ledger(self, graph) -> None:
        """确认并执行 TaskGraph，更新 Ledger 状态.

        供 _execute_pending_graph / _start_longrun 复用。
        """
        from .graph_display import render_graph_result

        # 确认
        self.console.print(
            "\n[bold]操作：[/bold] "
            "[green]y[/green]=开始执行  "
            "[red]n[/red]=进入 REPL 手动操作"
        )
        try:
            ans = await self._prompt_session.prompt_async(
                HTML("<b>开始执行任务图？ [y/n]: </b>")
            )
        except (EOFError, KeyboardInterrupt):
            self.console.print("[yellow]已跳过，进入 REPL[/yellow]")
            return

        if (ans or "").strip().lower() not in ("y", "yes", ""):
            self.console.print("[dim]已跳过自动执行，可在 REPL 中手动操作[/dim]")
            return

        # 更新 Ledger 状态
        ledger = self.agent.ledger
        manager = self.agent.ledger_manager
        if ledger and manager:
            from mini_code_agent.longrun.ledger_types import TaskRunStatus
            ledger.status = TaskRunStatus.RUNNING
            manager.update_phase(ledger, "execution")

        # 执行
        self.console.print()
        try:
            result = await self.graph_executor.execute(graph, self.agent)
        except KeyboardInterrupt:
            self.console.print("\n[yellow]已中断，进入 REPL 可用 /status 查看进度[/yellow]")
            return
        except Exception as e:
            self.console.print(f"\n[red]执行错误: {type(e).__name__}: {e}[/red]")
            return

        # 展示结果
        self._current_graph = result.graph
        render_graph_result(result, self.console)

        # 更新 Ledger 完成状态
        if ledger and manager:
            from mini_code_agent.core.task_graph import TaskStatus as GraphTaskStatus
            from mini_code_agent.longrun.ledger_types import TaskRunStatus

            # 从 graph 同步已完成子任务到 ledger.completed_tasks
            self._sync_completed_tasks_to_ledger(result.graph, ledger, manager)

            if result.tasks_failed == 0:
                ledger.status = TaskRunStatus.COMPLETED
            else:
                ledger.status = TaskRunStatus.FAILED
            manager.update_phase(ledger, "done")
            manager.update_resources(
                ledger,
                tokens=result.total_tokens,
                steps=result.total_steps,
                wall_time=result.wall_time,
            )

        self.console.print(
            "\n[dim]任务图执行完毕，进入 REPL。"
            "可用 /status 查看最终状态[/dim]"
        )

    async def _start_longrun(self, goal: str, token_budget: int) -> None:
        """从目标字符串开始完整的长程任务流程：planning → confirm → execute."""
        if self.graph_planner is None:
            self.console.print("[red]GraphPlanner 未初始化[/red]")
            return

        self.console.print(f"[dim]正在为长程任务生成 TaskGraph...[/dim]")

        try:
            graph = await self.graph_planner.plan_as_graph(goal)
        except Exception as e:
            self.console.print(f"[red]生成 TaskGraph 失败: {type(e).__name__}: {e}[/red]")
            return

        from .graph_display import render_graph_table
        render_graph_table(graph, self.console)
        self._current_graph = graph

        # 创建 Ledger
        manager = self.agent.ledger_manager
        if manager is None:
            self.console.print("[red]TaskLedgerManager 未初始化[/red]")
            return

        ledger = manager.create(
            goal=goal,
            task_graph=graph,
            budget=token_budget,
        )
        self.console.print(
            f"[green]Ledger 已创建: {ledger.task_id[:8]}[/green]\n"
            f"[dim]Token 预算: {token_budget:,}  "
            f"里程碑: {len(ledger.milestones)} 个[/dim]"
        )

        # 注入到 Agent
        self.agent.ledger = ledger
        self.agent.ledger_manager = manager

        # 执行
        await self._execute_graph_with_ledger(graph)

    async def _prompt_and_start_longrun(self) -> None:
        """交互式输入长程任务目标，然后走 planning + execution."""
        self.console.print(
            "\n[bold]长程任务模式[/bold] — 请输入任务目标："
        )
        try:
            goal = await self._prompt_session.prompt_async(
                HTML("<b><ansigreen>目标: </ansigreen></b>"),
            )
        except (EOFError, KeyboardInterrupt):
            self.console.print("[yellow]已取消，进入普通 REPL[/yellow]")
            return

        goal = goal.strip()
        if not goal:
            self.console.print("[yellow]目标为空，已取消[/yellow]")
            return

        await self._start_longrun(goal, self._token_budget)

    async def _handle_longrun_command(self, arg: str) -> None:
        """处理 /longrun [goal] 命令."""
        if self.agent.ledger is not None:
            self.console.print(
                "[yellow]当前已有活跃的长程任务。"
                "请先完成或用 /status 查看状态[/yellow]"
            )
            return

        goal = arg.strip()
        if not goal:
            # 交互式输入
            self.console.print("[bold]请输入长程任务目标：[/bold]")
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

        await self._start_longrun(goal, self._token_budget)

    def _sync_completed_tasks_to_ledger(self, graph, ledger, manager) -> None:
        """从已执行的 TaskGraph 同步已完成/失败的子任务到 Ledger.

        GraphExecutor 不直接操作 Ledger，所以在图执行完成后需要
        从 graph 的 COMPLETED 节点补充 CompletedTaskRecord。
        """
        from uuid import uuid4

        from mini_code_agent.core.task_graph import TaskStatus as GraphTaskStatus
        from mini_code_agent.longrun.ledger_types import (
            CompletedTaskRecord,
            FailedAttemptRecord,
        )

        existing_ids = {ct.task_id for ct in ledger.completed_tasks}

        for node in graph.nodes.values():
            if node.status == GraphTaskStatus.COMPLETED and node.id not in existing_ids:
                record = CompletedTaskRecord(
                    task_id=node.id,
                    artifact_id=str(uuid4()),  # 占位 — Artifact Protocol 完善后替换
                    description=node.description,
                    self_summary=node.result or node.description,
                    files_changed=node.files_involved,
                    verification_passed=True,
                    confidence="DONE",
                    step_number_start=0,
                    step_number_end=0,
                    token_count=0,
                )
                ledger.completed_tasks.append(record)

            elif node.status == GraphTaskStatus.FAILED and node.id not in {
                fa.task_id for fa in ledger.failed_attempts
            }:
                record = FailedAttemptRecord(
                    task_id=node.id,
                    artifact_id=str(uuid4()),
                    approach_description=node.description,
                    failure_reason=node.error or "未知",
                    step_number=0,
                    lesson_learned=None,
                )
                ledger.failed_attempts.append(record)

        # 检查里程碑
        manager._check_milestones(ledger)

    async def _run_agent_plan(self, user_input: str) -> None:
        """Plan mode：调用非流式 Agent.run()，中间通过回调渲染进度."""
        self.console.print()
        self.console.print(
            "[dim]进入 Plan mode，正在生成执行计划...[/dim]"
        )
        try:
            result = await self.agent.run(user_input)
        except KeyboardInterrupt:
            self.console.print("\n[yellow]已中断[/yellow]")
            return
        except Exception as e:
            self.console.print(f"\n[red]错误: {type(e).__name__}: {e}[/red]")
            return

        # 展示最终回复
        if result.content:
            self.console.print()
            self.console.print(result.content)
        if result.usage:
            self._render_usage_brief(result.usage)

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

    def _render_tool_call_start(self, tool_call: ToolCall | None) -> None:
        """渲染工具调用开始."""
        if not tool_call:
            return
        self.console.print(
            f"  [bold yellow]⚡ {tool_call.name}[/bold yellow]",
            highlight=False,
        )

    def _render_tool_call_args(self, tool_call: ToolCall | None) -> None:
        """渲染工具调用的参数."""
        if not tool_call:
            return
        args = tool_call.arguments
        # 为常见工具做特殊展示
        if tool_call.name == "Bash" and "command" in args:
            self.console.print(f"    [dim]$ {args['command']}[/dim]")
        elif tool_call.name == "ReadFile" and "path" in args:
            extra = ""
            if "start_line" in args:
                extra = f" (行 {args['start_line']}-{args.get('end_line', '末尾')})"
            self.console.print(f"    [dim]📄 {args['path']}{extra}[/dim]")
        elif tool_call.name == "WriteFile" and "path" in args:
            content = args.get("content", "")
            lines_count = len(content.splitlines())
            self.console.print(f"    [dim]✏️  {args['path']} ({lines_count} 行)[/dim]")
        elif tool_call.name == "Grep" and "pattern" in args:
            path = args.get("path", ".")
            self.console.print(f"    [dim]🔍 '{args['pattern']}' in {path}[/dim]")
        elif tool_call.name == "ListDir":
            path = args.get("path", ".")
            self.console.print(f"    [dim]📁 {path}[/dim]")
        elif tool_call.name == "GitStatus":
            path = args.get("path", ".")
            self.console.print(f"    [dim]📋 git status ({path})[/dim]")
        elif tool_call.name == "GitDiff":
            staged = args.get("staged", False)
            label = "staged" if staged else "unstaged"
            self.console.print(f"    [dim]📋 git diff ({label})[/dim]")
        elif tool_call.name == "GitCommit":
            msg = args.get("message", "")
            self.console.print(f"    [dim]📝 git commit -m \"{msg}\"[/dim]")
        elif tool_call.name == "GitLog":
            count = args.get("count", 10)
            self.console.print(f"    [dim]📋 git log -{count}[/dim]")
        else:
            compact = json.dumps(args, ensure_ascii=False)
            if len(compact) > 120:
                compact = compact[:117] + "..."
            self.console.print(f"    [dim]{compact}[/dim]")

    def _render_tool_result(self, tool_call: ToolCall | None, result: Any) -> None:
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
            self.console.print(f"    [red]✗ {error_text}[/red]")
        else:
            output = result.output
            lines = output.splitlines()
            if len(lines) > 5:
                preview = "\n".join(lines[:3])
                self.console.print(f"    [green]✓[/green] [dim]({len(lines)} 行输出)[/dim]")
            elif output:
                # 短输出直接显示
                if len(output) > 200:
                    output = output[:197] + "..."
                self.console.print(f"    [green]✓[/green] [dim]{output}[/dim]")
            else:
                self.console.print(f"    [green]✓[/green] [dim](空输出)[/dim]")

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
            f"  /plan         — 切换 Plan 模式（先规划后执行）\n"
            f"  /graph        — 切换 Graph 模式 / 查看当前任务图\n"
            f"  /graph-export — 导出 Mermaid 图表\n"
            f"  /status       — 查看 Task Ledger 状态\n"
            f"  /ledger       — 查看/导出 Ledger 详情\n"
            f"  /longrun      — 启动长程任务（交互输入或 /longrun 目标）\n"
            f"  /undo         — 回滚最近一次 Agent 修改\n"
            f"  /checkpoints  — 列出所有 checkpoint\n"
            f"  /diff         — 查看 Agent 的所有修改\n"
            f"  Ctrl+C  — 中断当前操作\n"
            f"  多行输入：Alt+Enter 换行，Enter 提交",
            border_style="blue",
        ))
