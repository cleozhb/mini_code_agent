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
    ) -> None:
        self.agent = agent
        self.console = console or Console()

        # 斜杠命令补全
        self._completer = WordCompleter(
            [
                "/quit", "/exit", "/q", "/clear", "/cost", "/model",
                "/memory", "/save", "/plan",
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

            # Plan mode 用非流式（在步骤间渲染进度）；否则走流式
            if self.agent.plan_mode and self.agent.planner is not None:
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

        self.console.print(f"[red]未知命令: {command}[/red]")
        self.console.print(
            "[dim]可用命令: /quit  /clear  /cost  /model  /memory  /save  /plan[/dim]"
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
            f"  /quit   — 退出\n"
            f"  /clear  — 清空对话\n"
            f"  /cost   — 查看 token 消耗\n"
            f"  /model  — 切换模型\n"
            f"  /memory — 查看记忆状态\n"
            f"  /save   — 保存信息到项目记忆\n"
            f"  /plan   — 切换 Plan 模式（先规划后执行）\n"
            f"  Ctrl+C  — 中断当前操作\n"
            f"  多行输入：Alt+Enter 换行，Enter 提交",
            border_style="blue",
        ))
