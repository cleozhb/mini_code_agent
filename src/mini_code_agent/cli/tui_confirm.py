"""TUI Confirm Screen — 工具调用确认弹窗，支持 edit."""

from __future__ import annotations

import json
from typing import Any

from rich.syntax import Syntax
from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.screen import ModalScreen
from textual.widgets import RichLog, TextArea

from ..llm.base import ToolCall
from ..safety.command_filter import SafetyLevel

_SAFETY_BADGES: dict[SafetyLevel, str] = {
    SafetyLevel.SAFE: "🟢 安全",
    SafetyLevel.NEEDS_CONFIRM: "🟡 需确认",
    SafetyLevel.BLOCKED: "🔴 已拦截",
}


class EditScreen(ModalScreen[tuple[bool, dict[str, Any] | None]]):
    """编辑工具参数后确认."""

    BINDINGS = [
        Binding("escape", "cancel", "取消"),
        Binding("ctrl+s", "save", "保存并确认"),
    ]

    DEFAULT_CSS = """
    EditScreen {
        align: center middle;
    }
    #edit-area {
        width: 80%;
        height: 70%;
        border: thick $accent;
    }
    #edit-hint {
        width: 80%;
        height: 3;
        dock: bottom;
    }
    """

    def __init__(self, tool_name: str, tool_call: ToolCall) -> None:
        super().__init__()
        self._tool_name = tool_name
        self._tool_call = tool_call

    def compose(self) -> ComposeResult:
        args = self._tool_call.arguments
        if self._tool_name == "Bash" and "command" in args:
            initial = args["command"]
            lang = "bash"
        else:
            initial = json.dumps(args, ensure_ascii=False, indent=2)
            lang = "json"
        yield TextArea(initial, id="edit-area", language=lang)
        hint = RichLog(id="edit-hint", wrap=True)
        yield hint

    def on_mount(self) -> None:
        hint = self.query_one("#edit-hint", RichLog)
        hint.write(Text("Ctrl+S 保存并确认 | Escape 取消", style="bold"))

    def action_save(self) -> None:
        area = self.query_one("#edit-area", TextArea)
        edited = area.text
        args = self._tool_call.arguments
        if self._tool_name == "Bash" and "command" in args:
            self.dismiss((True, {"command": edited}))
        else:
            try:
                parsed = json.loads(edited)
                self.dismiss((True, parsed))
            except json.JSONDecodeError:
                hint = self.query_one("#edit-hint", RichLog)
                hint.write(Text("JSON 解析失败，请修正后重试", style="red"))

    def action_cancel(self) -> None:
        self.dismiss((False, None))


class ConfirmScreen(ModalScreen[tuple[bool, dict[str, Any] | None]]):
    """展示工具调用详情，y 确认 / n 拒绝 / e 编辑."""

    BINDINGS = [
        Binding("y", "approve", "确认"),
        Binding("n", "deny", "拒绝"),
        Binding("e", "edit", "编辑"),
    ]

    DEFAULT_CSS = """
    ConfirmScreen {
        align: center middle;
    }
    #confirm-panel {
        width: 80%;
        max-height: 80%;
        border: thick $accent;
        background: $surface;
        padding: 1 2;
    }
    """

    def __init__(self, tool_name: str, tool_call: ToolCall, safety_level: SafetyLevel) -> None:
        super().__init__()
        self._tool_name = tool_name
        self._tool_call = tool_call
        self._safety_level = safety_level

    def compose(self) -> ComposeResult:
        log = RichLog(id="confirm-panel", wrap=True)
        yield log

    def on_mount(self) -> None:
        log = self.query_one("#confirm-panel", RichLog)
        badge = _SAFETY_BADGES.get(self._safety_level, "🟡 需确认")
        log.write(Text(f"{badge}  {self._tool_name}", style="bold"))
        log.write(Text(""))

        args = self._tool_call.arguments
        if self._tool_name == "Bash" and "command" in args:
            log.write(Syntax(args["command"], "bash", theme="monokai"))
        elif self._tool_name == "WriteFile" and "path" in args:
            content = args.get("content", "")
            lines = content.splitlines()
            preview = "\n".join(lines[:50])
            if len(lines) > 50:
                preview += f"\n... (共 {len(lines)} 行)"
            log.write(Text(f"文件: {args['path']}", style="bold"))
            log.write(Syntax(preview, "text", theme="monokai"))
        else:
            log.write(Syntax(
                json.dumps(args, ensure_ascii=False, indent=2),
                "json",
                theme="monokai",
            ))

        log.write(Text(""))
        edit_hint = "" if self._tool_name == "WriteFile" else "  [e] 编辑"
        log.write(Text(f"[y] 确认  [n] 拒绝{edit_hint}", style="bold"))

    def action_approve(self) -> None:
        self.dismiss((True, None))

    def action_deny(self) -> None:
        self.dismiss((False, None))

    def action_edit(self) -> None:
        if self._tool_name == "WriteFile":
            return
        self.app.push_screen(
            EditScreen(self._tool_name, self._tool_call),
            callback=self._on_edit_done,
        )

    def _on_edit_done(self, result: tuple[bool, dict[str, Any] | None]) -> None:
        if result[0]:
            self.dismiss(result)
