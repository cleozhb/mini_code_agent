"""代码搜索工具：GrepTool, ListDirTool."""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel, Field

from .base import PermissionLevel, Tool, ToolResult


# ---------------------------------------------------------------------------
# GrepTool
# ---------------------------------------------------------------------------


class GrepInput(BaseModel):
    pattern: str = Field(description="搜索模式（正则表达式）")
    path: str = Field(default=".", description="搜索路径，默认当前目录")


@dataclass
class GrepTool(Tool):
    """用 grep 搜索代码."""

    InputModel: ClassVar[type[BaseModel]] = GrepInput

    name: str = "Grep"
    description: str = (
        "在指定路径下递归搜索匹配 pattern 的内容。"
        "返回匹配的文件名、行号和内容。结果最多 50 条。"
    )
    permission_level: PermissionLevel = PermissionLevel.AUTO

    max_results: int = 50

    async def execute(self, **kwargs: Any) -> ToolResult:
        pattern: str = kwargs["pattern"]
        path: str = kwargs.get("path", ".")

        search_path = Path(path).expanduser()
        if not search_path.exists():
            return ToolResult(output="", error=f"路径不存在: {search_path}")

        cmd = [
            "grep", "-rn",
            "--include=*.py", "--include=*.js", "--include=*.ts",
            "--include=*.jsx", "--include=*.tsx", "--include=*.java",
            "--include=*.go", "--include=*.rs", "--include=*.c",
            "--include=*.cpp", "--include=*.h", "--include=*.hpp",
            "--include=*.rb", "--include=*.php", "--include=*.sh",
            "--include=*.yaml", "--include=*.yml", "--include=*.json",
            "--include=*.toml", "--include=*.cfg", "--include=*.ini",
            "--include=*.md", "--include=*.txt", "--include=*.html",
            "--include=*.css", "--include=*.sql",
            pattern,
            str(search_path),
        ]

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=15
            )
        except asyncio.TimeoutError:
            try:
                proc.kill()  # type: ignore[union-attr]
                await proc.wait()  # type: ignore[union-attr]
            except ProcessLookupError:
                pass
            return ToolResult(output="", error="搜索超时（15 秒）")

        output = stdout.decode("utf-8", errors="replace") if stdout else ""

        if proc.returncode == 1:
            # grep 未找到匹配
            return ToolResult(output="未找到匹配内容")

        if proc.returncode not in (0, 1):
            err_msg = stderr.decode("utf-8", errors="replace") if stderr else ""
            return ToolResult(output="", error=f"grep 执行出错: {err_msg}")

        # 限制结果数量
        lines = output.splitlines()
        if len(lines) > self.max_results:
            lines = lines[: self.max_results]
            lines.append(f"\n... 结果过多，仅显示前 {self.max_results} 条匹配")

        return ToolResult(output="\n".join(lines))


# ---------------------------------------------------------------------------
# ListDirTool
# ---------------------------------------------------------------------------

# 默认忽略的目录
_IGNORE_DIRS = {
    ".git", "node_modules", "__pycache__", ".venv", "venv",
    ".mypy_cache", ".pytest_cache", ".ruff_cache", ".tox",
    "dist", "build", ".eggs", "*.egg-info",
}


class ListDirInput(BaseModel):
    path: str = Field(default=".", description="目录路径，默认当前目录")
    max_depth: int = Field(default=2, description="最大递归深度，默认 2")


@dataclass
class ListDirTool(Tool):
    """列出目录结构."""

    InputModel: ClassVar[type[BaseModel]] = ListDirInput

    name: str = "ListDir"
    description: str = (
        "列出指定目录的树形结构。"
        "自动忽略 .git, node_modules, __pycache__ 等目录。"
    )
    permission_level: PermissionLevel = PermissionLevel.AUTO

    async def execute(self, **kwargs: Any) -> ToolResult:
        path: str = kwargs.get("path", ".")
        max_depth: int = kwargs.get("max_depth", 2)

        root = Path(path).expanduser()
        if not root.exists():
            return ToolResult(output="", error=f"路径不存在: {root}")
        if not root.is_dir():
            return ToolResult(output="", error=f"不是目录: {root}")

        lines: list[str] = [f"{root.name}/"]
        self._walk(root, lines, prefix="", depth=0, max_depth=max_depth)

        return ToolResult(output="\n".join(lines))

    def _walk(
        self,
        directory: Path,
        lines: list[str],
        prefix: str,
        depth: int,
        max_depth: int,
    ) -> None:
        if depth >= max_depth:
            return

        try:
            entries = sorted(directory.iterdir(), key=lambda e: (not e.is_dir(), e.name))
        except PermissionError:
            lines.append(f"{prefix}[权限不足]")
            return

        # 过滤掉忽略的目录
        entries = [
            e for e in entries
            if not self._should_ignore(e)
        ]

        for i, entry in enumerate(entries):
            is_last = i == len(entries) - 1
            connector = "└── " if is_last else "├── "
            child_prefix = prefix + ("    " if is_last else "│   ")

            if entry.is_dir():
                lines.append(f"{prefix}{connector}{entry.name}/")
                self._walk(entry, lines, child_prefix, depth + 1, max_depth)
            else:
                lines.append(f"{prefix}{connector}{entry.name}")

    def _should_ignore(self, entry: Path) -> bool:
        """检查是否应该忽略该条目."""
        name = entry.name
        if name in _IGNORE_DIRS:
            return True
        # 匹配 *.egg-info 模式
        if name.endswith(".egg-info"):
            return True
        # 隐藏文件也忽略（除了特定配置文件）
        if name.startswith(".") and name not in {".env", ".env.example", ".gitignore"}:
            return True
        return False
