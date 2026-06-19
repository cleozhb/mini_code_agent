"""TUIRenderer — 将 Agent 事件写入 Textual RichLog widget."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from rich.text import Text

from ..core.agent import AgentEvent, AgentEventType

if TYPE_CHECKING:
    from textual.widgets import RichLog

    from ..llm.base import TokenUsage, ToolCall
    from ..tools.base import ToolResult


class _RichLogConsoleProxy:
    """让 CommandHandler 能通过 renderer.console.print(...) 写入 RichLog."""

    def __init__(self, output: RichLog) -> None:
        self._output = output

    def print(self, *args: Any, **kwargs: Any) -> None:
        for item in args:
            self._output.write(item)


class TUIRenderer:
    """基于 Textual RichLog 的渲染实现."""

    def __init__(self, output: RichLog) -> None:
        self._output = output
        self._line_buffer = ""
        self.console = _RichLogConsoleProxy(output)

    def _flush_buffer(self) -> None:
        if self._line_buffer:
            self._output.write(Text(self._line_buffer))
            self._line_buffer = ""

    def render_text_delta(self, text: str) -> None:
        parts = (self._line_buffer + text).split("\n")
        for line in parts[:-1]:
            self._output.write(Text(line))
        self._line_buffer = parts[-1]

    def render_text_end(self) -> None:
        self._flush_buffer()

    def render_tool_call_start(self, tool_call: ToolCall | None, prefix: str = "  ") -> None:
        if not tool_call:
            return
        self._flush_buffer()
        self._output.write(Text(f"{prefix}⚡ {tool_call.name}", style="bold yellow"))

    def render_tool_call_args(self, tool_call: ToolCall | None, prefix: str = "    ") -> None:
        if not tool_call:
            return
        args = tool_call.arguments
        if tool_call.name == "Bash" and "command" in args:
            self._output.write(Text(f"{prefix}$ {args['command']}", style="dim"))
        elif tool_call.name == "ReadFile" and "path" in args:
            self._output.write(Text(f"{prefix}📄 {args['path']}", style="dim"))
        elif tool_call.name == "WriteFile" and "path" in args:
            lines_count = len(args.get("content", "").splitlines())
            self._output.write(Text(f"{prefix}✏️  {args['path']} ({lines_count} 行)", style="dim"))
        else:
            compact = json.dumps(args, ensure_ascii=False)
            if len(compact) > 120:
                compact = compact[:117] + "..."
            self._output.write(Text(f"{prefix}{compact}", style="dim"))

    def render_tool_result(self, tool_call: ToolCall | None, result: ToolResult | None, prefix: str = "    ") -> None:
        if result is None:
            return
        if result.is_error:
            error_text = result.error or "未知错误"
            if len(error_text) > 200:
                error_text = error_text[:197] + "..."
            self._output.write(Text(f"{prefix}✗ {error_text}", style="red"))
        else:
            output = result.output
            lines = output.splitlines()
            if len(lines) > 5:
                self._output.write(Text(f"{prefix}✓ ({len(lines)} 行输出)", style="green"))
            elif output:
                if len(output) > 200:
                    output = output[:197] + "..."
                self._output.write(Text(f"{prefix}✓ {output}", style="green"))
            else:
                self._output.write(Text(f"{prefix}✓ (空输出)", style="green"))

    def render_subagent_event(self, event: AgentEvent) -> None:
        if event.type == AgentEventType.TOOL_CALL_START:
            self.render_tool_call_start(event.tool_call, prefix="    ↳ ")
        elif event.type == AgentEventType.TOOL_CALL_END:
            self.render_tool_call_args(event.tool_call, prefix="      ")
        elif event.type == AgentEventType.TOOL_RESULT:
            self.render_tool_result(event.tool_call, event.tool_result, prefix="      ")

    def render_user_input(self, msg: str) -> None:
        self._output.write(Text(f"> {msg}", style="bold green"))

    def render_system(self, msg: str) -> None:
        self._output.write(Text(msg, style="dim"))

    def render_error(self, msg: str) -> None:
        self._output.write(Text(msg, style="red"))

    def render_finish(self, usage: TokenUsage | None) -> None:
        self._flush_buffer()
        if usage:
            self._output.write(Text(
                f"tokens: {usage.input_tokens:,} in / {usage.output_tokens:,} out",
                style="dim",
            ))

    def dispatch_event(self, event: AgentEvent) -> None:
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
            self._flush_buffer()
            self._output.write(Text("已暂停", style="yellow"))
