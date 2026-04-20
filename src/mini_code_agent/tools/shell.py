"""Shell 执行工具：BashTool."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, ClassVar

from pydantic import BaseModel, Field

from .base import PermissionLevel, Tool, ToolResult


class BashInput(BaseModel):
    command: str = Field(description="要执行的 shell 命令")


@dataclass
class BashTool(Tool):
    """执行 Shell 命令."""

    InputModel: ClassVar[type[BaseModel]] = BashInput

    name: str = "Bash"
    description: str = (
        "在 shell 中执行命令，返回 stdout+stderr 合并输出和 exit_code。"
        "非零退出码不算工具错误（pytest / grep 等命令的 exit!=0 是业务信号），"
        "退出码会附加在输出末尾 `[exit code: N]`。只有超时 / 子进程启动失败算工具错误。"
        "超时 30 秒。输出超过 200 行会截断中间部分。"
    )
    permission_level: PermissionLevel = PermissionLevel.CONFIRM

    timeout: int = 30
    max_output_lines: int = 200
    head_lines: int = 20
    tail_lines: int = 50
    # 限制子进程的工作目录；None 表示继承当前进程 cwd（保持原行为）
    cwd: str | None = None

    async def execute(self, **kwargs: Any) -> ToolResult:
        command: str = kwargs["command"]

        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=self.cwd,
            )
            stdout, _ = await asyncio.wait_for(
                proc.communicate(), timeout=self.timeout
            )
        except asyncio.TimeoutError:
            # 超时后尝试终止进程
            try:
                proc.kill()  # type: ignore[union-attr]
                await proc.wait()  # type: ignore[union-attr]
            except ProcessLookupError:
                pass
            return ToolResult(
                output="",
                error=f"命令执行超时（{self.timeout} 秒）: {command}",
                exit_code=-1,
            )

        output = stdout.decode("utf-8", errors="replace") if stdout else ""
        exit_code = proc.returncode

        # 截断过长输出
        output = self._truncate_output(output)

        # 非零 exit code 不打 is_error：pytest / grep / diff 等命令的非零退出
        # 是有意义的业务信号（"测试挂了" / "没匹配到"），不是工具故障。把 exit
        # code 附在 output 末尾，让 Agent 自己据此判断。超时 / 启动失败仍然走 error。
        if exit_code != 0:
            sep = "" if not output or output.endswith("\n") else "\n"
            output = f"{output}{sep}[exit code: {exit_code}]"

        return ToolResult(output=output, exit_code=exit_code)

    def _truncate_output(self, output: str) -> str:
        """如果输出超过 max_output_lines 行，截断中间部分."""
        lines = output.splitlines()
        if len(lines) <= self.max_output_lines:
            return output

        head = lines[: self.head_lines]
        tail = lines[-self.tail_lines :]
        omitted = len(lines) - self.head_lines - self.tail_lines

        return "\n".join(
            head
            + [f"\n... [省略 {omitted} 行] ...\n"]
            + tail
        )
