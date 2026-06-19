"""ConsoleRenderer — Rich Console 实现的 Renderer，从 REPL 中抽取."""

from __future__ import annotations

import json
import sys

from rich.console import Console

from ..core.agent import AgentEvent, AgentEventType
from ..llm.base import TokenUsage, ToolCall
from ..tools.base import ToolResult


class ConsoleRenderer:
    """基于 Rich Console + sys.stdout 的渲染实现."""

    def __init__(self, console: Console | None = None) -> None:
        self.console = console or Console()
        self._in_text = False

    def render_text_delta(self, text: str) -> None:
        self._in_text = True
        sys.stdout.write(text)
        sys.stdout.flush()

    def render_text_end(self) -> None:
        if self._in_text:
            sys.stdout.write("\n")
            sys.stdout.flush()
            self._in_text = False

    def render_tool_call_start(self, tool_call: ToolCall | None, prefix: str = "  ") -> None:
        if not tool_call:
            return
        self.render_text_end()
        self.console.print(
            f"{prefix}[bold yellow]⚡ {tool_call.name}[/bold yellow]",
            highlight=False,
        )

    def render_tool_call_args(self, tool_call: ToolCall | None, prefix: str = "    ") -> None:
        if not tool_call:
            return
        args = tool_call.arguments
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

    def render_tool_result(self, tool_call: ToolCall | None, result: ToolResult | None, prefix: str = "    ") -> None:
        if result is None:
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
                self.console.print(f"{prefix}[green]✓[/green] [dim]({len(lines)} 行输出)[/dim]")
            elif output:
                if len(output) > 200:
                    output = output[:197] + "..."
                self.console.print(f"{prefix}[green]✓[/green] [dim]{output}[/dim]")
            else:
                self.console.print(f"{prefix}[green]✓[/green] [dim](空输出)[/dim]")

    def render_subagent_event(self, event: AgentEvent) -> None:
        if event.type == AgentEventType.TOOL_CALL_START:
            self.render_tool_call_start(event.tool_call, prefix="    ↳ ")
        elif event.type == AgentEventType.TOOL_CALL_END:
            self.render_tool_call_args(event.tool_call, prefix="      ")
        elif event.type == AgentEventType.TOOL_RESULT:
            self.render_tool_result(event.tool_call, event.tool_result, prefix="      ")
        elif event.type == AgentEventType.FINISH and event.usage:
            self.console.print(
                f"      [dim]SubAgent tokens: "
                f"{event.usage.input_tokens:,} in / "
                f"{event.usage.output_tokens:,} out[/dim]"
            )

    def render_system(self, msg: str) -> None:
        self.console.print(f"[dim]{msg}[/dim]")

    def render_error(self, msg: str) -> None:
        self.console.print(f"[red]{msg}[/red]")

    def render_finish(self, usage: TokenUsage | None) -> None:
        self.render_text_end()
        if usage:
            self.console.print(
                f"\n[dim]tokens: {usage.input_tokens:,} in / {usage.output_tokens:,} out[/dim]"
            )

    def dispatch_event(self, event: AgentEvent) -> None:
        """按 event.type 分发到具体渲染方法."""
        if event.type == AgentEventType.TEXT_DELTA:
            self.render_text_delta(event.content)
        elif event.type == AgentEventType.TOOL_CALL_START:
            self.render_tool_call_start(event.tool_call)
        elif event.type == AgentEventType.TOOL_CALL_END:
            self.render_tool_call_args(event.tool_call)
        elif event.type == AgentEventType.TOOL_RESULT:
            self.render_tool_result(event.tool_call, event.tool_result)
        elif event.type == AgentEventType.FINISH:
            self.render_finish(event.usage)
        elif event.type == AgentEventType.INTERRUPTED:
            self.render_text_end()
            self.console.print("\n[yellow]已暂停[/yellow]")
