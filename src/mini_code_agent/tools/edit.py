"""文件局部编辑工具：EditFileTool."""

from __future__ import annotations

import difflib
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel, Field

from .base import PermissionLevel, Tool, ToolResult


def _strip_trailing_whitespace(s: str) -> str:
    """去除每行末尾空白."""
    return "\n".join(line.rstrip() for line in s.splitlines())


def _normalize_indent(s: str) -> str:
    """去除公共缩进，用于模糊匹配."""
    return textwrap.dedent(s)


def _make_diff(old: str, new: str, path: str) -> str:
    """生成 unified diff 字符串."""
    old_lines = old.splitlines(keepends=True)
    new_lines = new.splitlines(keepends=True)
    diff = difflib.unified_diff(old_lines, new_lines, fromfile=path, tofile=path)
    return "".join(diff)


def _preview(text: str, max_lines: int = 3) -> str:
    """取文本前几行用于提示."""
    lines = text.splitlines()
    preview = "\n".join(lines[:max_lines])
    if len(lines) > max_lines:
        preview += f"\n... (共 {len(lines)} 行)"
    return preview


class EditFileInput(BaseModel):
    path: str = Field(description="文件路径")
    old_content: str = Field(description="要替换的原始文本（必须在文件中唯一匹配）")
    new_content: str = Field(description="替换后的文本（空字符串表示删除）")


@dataclass
class EditFileTool(Tool):
    """局部编辑文件：查找并替换唯一匹配的文本片段."""

    InputModel: ClassVar[type[BaseModel]] = EditFileInput

    name: str = "EditFile"
    description: str = (
        "局部编辑文件内容。提供要替换的原始文本(old_content)和替换后的文本(new_content)。"
        "old_content 必须在文件中唯一匹配。new_content 为空字符串表示删除该片段。"
        "修改已有文件时优先使用此工具而不是 WriteFile。"
    )
    permission_level: PermissionLevel = PermissionLevel.CONFIRM

    async def execute(self, **kwargs: Any) -> ToolResult:
        path_str: str = kwargs["path"]
        old_content: str = kwargs["old_content"]
        new_content: str = kwargs["new_content"]

        path = Path(path_str).expanduser()

        # --- 基础校验 ---
        if not path.exists():
            return ToolResult(output="", error=f"文件不存在: {path}")
        if not path.is_file():
            return ToolResult(output="", error=f"不是文件: {path}")

        try:
            file_content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return ToolResult(output="", error=f"无法以 UTF-8 编码读取文件: {path}")
        except PermissionError:
            return ToolResult(output="", error=f"没有读取权限: {path}")

        # --- 精确匹配 ---
        count = file_content.count(old_content)
        if count == 1:
            new_file_content = file_content.replace(old_content, new_content, 1)
            diff = _make_diff(file_content, new_file_content, path_str)
            path.write_text(new_file_content, encoding="utf-8")
            return ToolResult(output=f"已编辑 {path}\n\n{diff}")

        if count > 1:
            return ToolResult(
                output="",
                error=f"old_content 在文件中匹配了 {count} 处，请提供更多上下文使其唯一匹配",
            )

        # --- 容错：忽略首尾空白差异 ---
        stripped_old = _strip_trailing_whitespace(old_content)
        stripped_file = _strip_trailing_whitespace(file_content)
        if stripped_file.count(stripped_old) == 1:
            # 在原始文件中找到对应位置并替换
            idx = stripped_file.index(stripped_old)
            # 将 stripped 位置映射回原始文件
            # 逐行对齐：找到 stripped_old 对应的原始行范围
            result = self._replace_by_stripped_match(
                file_content, old_content, new_content, path, path_str
            )
            if result is not None:
                return result

        # --- 容错：忽略缩进差异 ---
        dedented_old = _normalize_indent(old_content)
        result = self._try_dedent_match(
            file_content, dedented_old, new_content, path, path_str
        )
        if result is not None:
            return result

        # --- 容错：相似片段建议 ---
        suggestion = self._find_similar(file_content, old_content)
        if suggestion:
            return ToolResult(
                output="",
                error=f"old_content not found in file\n\nDid you mean this?\n{suggestion}",
            )

        return ToolResult(output="", error="old_content not found in file")

    def _replace_by_stripped_match(
        self,
        file_content: str,
        old_content: str,
        new_content: str,
        path: Path,
        path_str: str,
    ) -> ToolResult | None:
        """忽略行尾空白后进行匹配替换."""
        old_lines = old_content.splitlines()
        file_lines = file_content.splitlines()

        # 滑动窗口查找
        old_stripped = [line.rstrip() for line in old_lines]
        matches = []
        for i in range(len(file_lines) - len(old_lines) + 1):
            window = [file_lines[i + j].rstrip() for j in range(len(old_lines))]
            if window == old_stripped:
                matches.append(i)

        if len(matches) == 1:
            start = matches[0]
            end = start + len(old_lines)
            new_lines = file_lines[:start] + new_content.splitlines() + file_lines[end:]
            new_file_content = "\n".join(new_lines)
            # 保留原文件的末尾换行符
            if file_content.endswith("\n"):
                new_file_content += "\n"
            diff = _make_diff(file_content, new_file_content, path_str)
            path.write_text(new_file_content, encoding="utf-8")
            return ToolResult(output=f"已编辑 {path}（忽略行尾空白匹配）\n\n{diff}")

        return None

    def _try_dedent_match(
        self,
        file_content: str,
        dedented_old: str,
        new_content: str,
        path: Path,
        path_str: str,
    ) -> ToolResult | None:
        """忽略缩进差异进行匹配."""
        file_lines = file_content.splitlines()
        old_lines = dedented_old.splitlines()

        # 对每个起始位置，dedent 对应的窗口并比较
        matches = []
        for i in range(len(file_lines) - len(old_lines) + 1):
            window = file_lines[i : i + len(old_lines)]
            dedented_window = textwrap.dedent("\n".join(window)).splitlines()
            if dedented_window == old_lines:
                matches.append(i)

        if len(matches) == 1:
            start = matches[0]
            end = start + len(old_lines)
            new_lines = file_lines[:start] + new_content.splitlines() + file_lines[end:]
            new_file_content = "\n".join(new_lines)
            if file_content.endswith("\n"):
                new_file_content += "\n"
            diff = _make_diff(file_content, new_file_content, path_str)
            path.write_text(new_file_content, encoding="utf-8")
            return ToolResult(output=f"已编辑 {path}（忽略缩进差异匹配）\n\n{diff}")

        return None

    def _find_similar(self, file_content: str, old_content: str) -> str | None:
        """用 difflib 找最相似的片段."""
        file_lines = file_content.splitlines()
        old_lines = old_content.splitlines()
        n = len(old_lines)

        if n == 0 or len(file_lines) == 0:
            return None

        # 收集所有同长度窗口
        candidates: list[str] = []
        for i in range(max(1, len(file_lines) - n + 1)):
            end = min(i + n, len(file_lines))
            chunk = "\n".join(file_lines[i:end])
            candidates.append(chunk)

        matches = difflib.get_close_matches(old_content, candidates, n=1, cutoff=0.5)
        if matches:
            return _preview(matches[0])
        return None
