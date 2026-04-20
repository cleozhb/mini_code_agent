"""Git 工具集：查看状态、diff、提交、日志."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, ClassVar

from pydantic import BaseModel, Field

from .base import PermissionLevel, Tool, ToolResult


# ---------------------------------------------------------------------------
# 辅助：运行 git 命令
# ---------------------------------------------------------------------------

async def _run_git(
    *args: str,
    cwd: str | None = None,
    timeout: int = 15,
) -> tuple[int, str]:
    """运行 git 子命令，返回 (exit_code, stdout+stderr)."""
    cmd = ["git", *args]
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        cwd=cwd,
    )
    try:
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        try:
            proc.kill()
            await proc.wait()
        except ProcessLookupError:
            pass
        return -1, f"git 命令超时（{timeout}s）: git {' '.join(args)}"
    output = stdout.decode("utf-8", errors="replace") if stdout else ""
    return proc.returncode or 0, output


# ---------------------------------------------------------------------------
# GitStatusTool
# ---------------------------------------------------------------------------


class GitStatusInput(BaseModel):
    path: str = Field(default=".", description="项目目录路径（默认当前目录）")


@dataclass
class GitStatusTool(Tool):
    """查看 git 仓库状态：分支、staged/unstaged/untracked 文件."""

    InputModel: ClassVar[type[BaseModel]] = GitStatusInput

    name: str = "GitStatus"
    description: str = (
        "查看 git 仓库状态，返回当前分支名、staged/unstaged/untracked 文件列表。"
        "用于了解工作区状态。"
    )
    permission_level: PermissionLevel = PermissionLevel.AUTO

    async def execute(self, **kwargs: Any) -> ToolResult:
        path: str = kwargs.get("path", ".")

        # 检查是否在 git 仓库中
        code, out = await _run_git("rev-parse", "--is-inside-work-tree", cwd=path)
        if code != 0:
            return ToolResult(output="", error=f"不是 git 仓库: {path}")

        # 获取当前分支
        code, branch = await _run_git("branch", "--show-current", cwd=path)
        branch = branch.strip() or "(detached HEAD)"

        # 获取 porcelain 状态
        code, status = await _run_git("status", "--porcelain", cwd=path)

        staged: list[str] = []
        unstaged: list[str] = []
        untracked: list[str] = []

        for line in status.splitlines():
            if len(line) < 3:
                continue
            x, y = line[0], line[1]
            filename = line[3:]
            if x == "?":
                untracked.append(filename)
            else:
                if x not in (" ", "?"):
                    staged.append(f"{x} {filename}")
                if y not in (" ", "?"):
                    unstaged.append(f"{y} {filename}")

        parts = [f"分支: {branch}"]
        if staged:
            parts.append(f"\nStaged ({len(staged)}):")
            parts.extend(f"  {f}" for f in staged)
        if unstaged:
            parts.append(f"\nUnstaged ({len(unstaged)}):")
            parts.extend(f"  {f}" for f in unstaged)
        if untracked:
            parts.append(f"\nUntracked ({len(untracked)}):")
            parts.extend(f"  {f}" for f in untracked)
        if not staged and not unstaged and not untracked:
            parts.append("\n工作区干净，无改动。")

        return ToolResult(output="\n".join(parts))


# ---------------------------------------------------------------------------
# GitDiffTool
# ---------------------------------------------------------------------------


class GitDiffInput(BaseModel):
    path: str = Field(default=".", description="项目目录路径（默认当前目录）")
    staged: bool = Field(default=False, description="是否查看 staged 的 diff（默认查看 unstaged）")


@dataclass
class GitDiffTool(Tool):
    """查看 git diff 输出."""

    InputModel: ClassVar[type[BaseModel]] = GitDiffInput

    name: str = "GitDiff"
    description: str = (
        "查看 git diff 输出。默认查看 unstaged 改动，设 staged=true 查看已暂存的改动。"
        "超过 200 行会截断。"
    )
    permission_level: PermissionLevel = PermissionLevel.AUTO

    max_lines: int = 200

    async def execute(self, **kwargs: Any) -> ToolResult:
        path: str = kwargs.get("path", ".")
        staged: bool = kwargs.get("staged", False)

        args = ["diff"]
        if staged:
            args.append("--cached")

        code, diff_output = await _run_git(*args, cwd=path)
        if code != 0:
            return ToolResult(output="", error=f"git diff 失败: {diff_output}")

        if not diff_output.strip():
            label = "staged" if staged else "unstaged"
            return ToolResult(output=f"无 {label} 改动。")

        # 截断
        lines = diff_output.splitlines()
        if len(lines) > self.max_lines:
            truncated = lines[: self.max_lines]
            truncated.append(f"\n... [截断：共 {len(lines)} 行，仅显示前 {self.max_lines} 行]")
            diff_output = "\n".join(truncated)

        return ToolResult(output=diff_output)


# ---------------------------------------------------------------------------
# GitCommitTool
# ---------------------------------------------------------------------------


class GitCommitInput(BaseModel):
    message: str = Field(description="commit message")


@dataclass
class GitCommitTool(Tool):
    """提交代码：自动 stage 所有已跟踪的修改文件，然后 commit."""

    InputModel: ClassVar[type[BaseModel]] = GitCommitInput

    name: str = "GitCommit"
    description: str = (
        "提交代码。自动 stage 所有已跟踪但被修改的文件（不含 untracked），"
        "然后创建 commit。需要用户确认。"
    )
    permission_level: PermissionLevel = PermissionLevel.CONFIRM

    async def execute(self, **kwargs: Any) -> ToolResult:
        message: str = kwargs["message"]

        # 先 stage 所有已跟踪的改动（-u 不包含 untracked）
        code, out = await _run_git("add", "-u")
        if code != 0:
            return ToolResult(output="", error=f"git add -u 失败: {out}")

        # 检查是否有 staged 内容
        code, status = await _run_git("diff", "--cached", "--stat")
        if not status.strip():
            return ToolResult(output="", error="没有已暂存的改动可提交。")

        # 执行 commit
        code, out = await _run_git("commit", "-m", message)
        if code != 0:
            return ToolResult(output="", error=f"git commit 失败: {out}")

        # 获取简要信息
        _, log_line = await _run_git("log", "--oneline", "-1")
        return ToolResult(output=f"提交成功: {log_line.strip()}\n\n{status.strip()}")


# ---------------------------------------------------------------------------
# GitLogTool
# ---------------------------------------------------------------------------


class GitLogInput(BaseModel):
    count: int = Field(default=10, description="显示最近 N 条 commit（默认 10）")
    oneline: bool = Field(default=True, description="简洁模式（默认 true）")


@dataclass
class GitLogTool(Tool):
    """查看 git 提交历史."""

    InputModel: ClassVar[type[BaseModel]] = GitLogInput

    name: str = "GitLog"
    description: str = "查看最近的 git 提交历史。"
    permission_level: PermissionLevel = PermissionLevel.AUTO

    async def execute(self, **kwargs: Any) -> ToolResult:
        count: int = kwargs.get("count", 10)
        oneline: bool = kwargs.get("oneline", True)

        args = ["log", f"-{count}"]
        if oneline:
            args.append("--oneline")

        code, out = await _run_git(*args)
        if code != 0:
            return ToolResult(output="", error=f"git log 失败: {out}")

        return ToolResult(output=out.strip() if out.strip() else "暂无提交历史。")
