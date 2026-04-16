"""确认机制 — 对需要 CONFIRM 权限的工具操作进行用户确认."""

from __future__ import annotations

import difflib
import json
from pathlib import Path
from typing import Any

from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text

from ..llm.base import ToolCall


async def confirm_tool_call(
    tool_name: str,
    tool_call: ToolCall,
    console: Console,
    prompt_session: PromptSession,
) -> tuple[bool, dict[str, Any] | None]:
    """展示待执行的操作并请求用户确认.

    Args:
        tool_name: 工具注册名称
        tool_call: 工具调用对象
        console: Rich console 用于渲染
        prompt_session: prompt_toolkit session 用于输入

    Returns:
        (approved, edited_args_or_none)
        - approved=True, None: 用户同意，使用原参数
        - approved=True, new_args: 用户编辑后同意
        - approved=False, None: 用户拒绝
    """
    args = tool_call.arguments

    if tool_name == "WriteFile":
        _render_write_file(args, console)
    elif tool_name == "Bash":
        _render_bash(args, console)
    else:
        _render_generic(tool_name, args, console)

    # 等待用户输入
    while True:
        try:
            answer = await prompt_session.prompt_async(
                HTML("<b>[y]确认 / [n]拒绝 / [e]编辑: </b>"),
            )
        except (EOFError, KeyboardInterrupt):
            return False, None

        answer = answer.strip().lower()
        if answer in ("y", "yes", ""):
            return True, None
        if answer in ("n", "no"):
            return False, None
        if answer in ("e", "edit"):
            edited = await _edit_args(tool_name, args, console, prompt_session)
            if edited is not None:
                return True, edited
            # 编辑取消，重新提示
            continue

        console.print("[dim]请输入 y / n / e[/dim]")


def _render_write_file(args: dict[str, Any], console: Console) -> None:
    """展示 WriteFile 操作：对已有文件显示 diff，新文件显示内容."""
    path_str = args.get("path", "")
    content = args.get("content", "")
    path = Path(path_str).expanduser()

    console.print()
    if path.exists() and path.is_file():
        # 显示 diff
        try:
            original = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, PermissionError):
            original = ""

        diff_lines = list(difflib.unified_diff(
            original.splitlines(keepends=True),
            content.splitlines(keepends=True),
            fromfile=f"a/{path_str}",
            tofile=f"b/{path_str}",
            lineterm="",
        ))

        if diff_lines:
            diff_text = "\n".join(diff_lines)
            syntax = Syntax(diff_text, "diff", theme="monokai", line_numbers=False)
            console.print(Panel(
                syntax,
                title=f"[bold yellow]WriteFile[/bold yellow] → {path_str}",
                border_style="yellow",
            ))
        else:
            console.print(Panel(
                "[dim]文件内容无变化[/dim]",
                title=f"[bold yellow]WriteFile[/bold yellow] → {path_str}",
                border_style="yellow",
            ))
    else:
        # 新文件：显示内容预览
        # 猜测语法高亮
        suffix = path.suffix.lstrip(".")
        lexer = _suffix_to_lexer(suffix)
        preview = content
        if len(content.splitlines()) > 50:
            lines = content.splitlines()
            preview = "\n".join(lines[:50]) + f"\n... (共 {len(lines)} 行)"

        syntax = Syntax(preview, lexer, theme="monokai", line_numbers=True)
        console.print(Panel(
            syntax,
            title=f"[bold green]WriteFile (新文件)[/bold green] → {path_str}",
            border_style="green",
        ))


def _render_bash(args: dict[str, Any], console: Console) -> None:
    """展示 Bash 命令."""
    command = args.get("command", "")
    console.print()
    syntax = Syntax(command, "bash", theme="monokai", line_numbers=False)
    console.print(Panel(
        syntax,
        title="[bold red]Bash[/bold red]",
        border_style="red",
    ))


def _render_generic(tool_name: str, args: dict[str, Any], console: Console) -> None:
    """展示通用工具调用."""
    console.print()
    args_text = json.dumps(args, ensure_ascii=False, indent=2)
    syntax = Syntax(args_text, "json", theme="monokai")
    console.print(Panel(
        syntax,
        title=f"[bold yellow]{tool_name}[/bold yellow]",
        border_style="yellow",
    ))


async def _edit_args(
    tool_name: str,
    args: dict[str, Any],
    console: Console,
    prompt_session: PromptSession,
) -> dict[str, Any] | None:
    """允许用户编辑工具参数.

    对 Bash: 编辑 command
    对 WriteFile: 编辑 content
    其他: 编辑 JSON
    """
    if tool_name == "Bash":
        console.print("[dim]输入新命令（留空取消）：[/dim]")
        try:
            new_cmd = await prompt_session.prompt_async("$ ")
        except (EOFError, KeyboardInterrupt):
            return None
        if new_cmd.strip():
            return {**args, "command": new_cmd.strip()}
        return None

    if tool_name == "WriteFile":
        console.print("[dim]编辑文件内容暂不支持，请选择 y 确认或 n 拒绝[/dim]")
        return None

    # 通用：编辑 JSON
    console.print("[dim]输入修改后的 JSON（留空取消）：[/dim]")
    try:
        new_json = await prompt_session.prompt_async("> ")
    except (EOFError, KeyboardInterrupt):
        return None
    if new_json.strip():
        try:
            return json.loads(new_json)
        except json.JSONDecodeError:
            console.print("[red]JSON 解析失败，取消编辑[/red]")
    return None


def _suffix_to_lexer(suffix: str) -> str:
    """文件后缀 → Pygments lexer 名称."""
    mapping = {
        "py": "python",
        "js": "javascript",
        "ts": "typescript",
        "jsx": "jsx",
        "tsx": "tsx",
        "java": "java",
        "go": "go",
        "rs": "rust",
        "c": "c",
        "cpp": "cpp",
        "h": "c",
        "sh": "bash",
        "bash": "bash",
        "zsh": "bash",
        "json": "json",
        "yaml": "yaml",
        "yml": "yaml",
        "toml": "toml",
        "md": "markdown",
        "html": "html",
        "css": "css",
        "sql": "sql",
        "xml": "xml",
    }
    return mapping.get(suffix, "text")
