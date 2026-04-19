"""文件操作工具：ReadFileTool, WriteFileTool."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel, Field

from .base import PermissionLevel, Tool, ToolResult


# ---------------------------------------------------------------------------
# ReadFileTool
# ---------------------------------------------------------------------------

# 行号格式化：右对齐宽度内加 tab 分隔
def _format_line(line_num: int, line: str, width: int) -> str:
    """格式化带行号前缀的行，如 '  42\tdef foo():'."""
    return f"{line_num:>{width}}\t{line}"


class ReadFileInput(BaseModel):
    path: str = Field(description="文件路径")
    start_line: int | None = Field(
        default=None, description="起始行号（从 1 开始），不传则从第 1 行开始"
    )
    end_line: int | None = Field(
        default=None, description="结束行号（包含），不传则到文件末尾"
    )


@dataclass
class ReadFileTool(Tool):
    """读取文件内容，支持行号范围."""

    InputModel: ClassVar[type[BaseModel]] = ReadFileInput

    name: str = "ReadFile"
    description: str = (
        "读取指定文件的内容。可通过 start_line / end_line 读取特定行范围。"
        "大文件（>500 行）未指定范围时会自动截断中间部分，"
        "保留首 200 行和末 50 行。"
    )
    permission_level: PermissionLevel = PermissionLevel.AUTO

    # 截断阈值
    MAX_FULL_LINES: int = 500
    HEAD_LINES: int = 200
    TAIL_LINES: int = 50

    async def execute(self, **kwargs: Any) -> ToolResult:
        path_str: str = kwargs["path"]
        start_line: int | None = kwargs.get("start_line")
        end_line: int | None = kwargs.get("end_line")

        path = Path(path_str).expanduser()

        if not path.exists():
            return ToolResult(output="", error=f"文件不存在: {path}")
        if not path.is_file():
            return ToolResult(output="", error=f"不是文件: {path}")

        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return ToolResult(output="", error=f"无法以 UTF-8 编码读取文件: {path}")
        except PermissionError:
            return ToolResult(output="", error=f"没有读取权限: {path}")

        lines = content.splitlines()
        total = len(lines)

        # 行号宽度用于对齐
        width = len(str(total)) if total > 0 else 1

        has_range = start_line is not None or end_line is not None

        if has_range:
            # 处理默认值
            sl = start_line if start_line is not None else 1
            el = end_line if end_line is not None else total

            # start_line 超出文件行数
            if sl > total:
                return ToolResult(
                    output="",
                    error=f"start_line({sl}) 超出文件总行数({total})",
                )

            # 规范化范围
            sl = max(1, sl)
            el = min(el, total)

            selected = lines[sl - 1 : el]
            formatted = [
                _format_line(sl + i, line, width)
                for i, line in enumerate(selected)
            ]
            return ToolResult(output="\n".join(formatted))

        # 无行号范围：小文件完整返回
        if total <= self.MAX_FULL_LINES:
            formatted = [
                _format_line(i + 1, line, width)
                for i, line in enumerate(lines)
            ]
            return ToolResult(output="\n".join(formatted))

        # 大文件：截断中间
        head = lines[: self.HEAD_LINES]
        tail = lines[-self.TAIL_LINES :]

        truncated_start = self.HEAD_LINES + 1
        truncated_end = total - self.TAIL_LINES
        truncated_count = truncated_end - truncated_start + 1

        head_formatted = [
            _format_line(i + 1, line, width)
            for i, line in enumerate(head)
        ]
        tail_formatted = [
            _format_line(total - self.TAIL_LINES + i + 1, line, width)
            for i, line in enumerate(tail)
        ]

        truncation_msg = (
            f"\n... [此处省略第 {truncated_start}-{truncated_end} 行，"
            f"共 {truncated_count} 行被截断] ...\n"
            f"（文件共 {total} 行。可使用 start_line/end_line 参数读取被截断的部分，\n"
            f" 或使用 GrepTool 搜索关键词定位具体行号）\n"
        )

        return ToolResult(
            output="\n".join(head_formatted) + truncation_msg + "\n".join(tail_formatted)
        )


# ---------------------------------------------------------------------------
# WriteFileTool
# ---------------------------------------------------------------------------


class WriteFileInput(BaseModel):
    path: str = Field(description="文件路径")
    content: str = Field(description="要写入的内容")


@dataclass
class WriteFileTool(Tool):
    """写入文件内容，写之前记录原始内容用于回滚."""

    InputModel: ClassVar[type[BaseModel]] = WriteFileInput

    name: str = "WriteFile"
    description: str = "将内容写入指定文件。如果文件已存在会覆盖，会自动创建不存在的父目录。"
    permission_level: PermissionLevel = PermissionLevel.CONFIRM

    # 记录写入前的原始内容，用于回滚
    _original_contents: dict[str, str | None] = field(
        default_factory=dict, repr=False
    )

    async def execute(self, **kwargs: Any) -> ToolResult:
        path_str: str = kwargs["path"]
        content: str = kwargs["content"]

        path = Path(path_str).expanduser()

        # 记录原始内容
        if path.exists() and path.is_file():
            try:
                self._original_contents[str(path)] = path.read_text(
                    encoding="utf-8"
                )
            except (UnicodeDecodeError, PermissionError):
                self._original_contents[str(path)] = None
        else:
            self._original_contents[str(path)] = None

        # 确保父目录存在
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
        except PermissionError:
            return ToolResult(output="", error=f"没有权限创建目录: {path.parent}")

        try:
            path.write_text(content, encoding="utf-8")
        except PermissionError:
            return ToolResult(output="", error=f"没有写入权限: {path}")

        return ToolResult(output=f"已写入 {path}（{len(content)} 字符）")
