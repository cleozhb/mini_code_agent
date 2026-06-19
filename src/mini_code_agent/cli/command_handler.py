"""CommandHandler — 命令解析与路由层，REPL 和 TUI 共用."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..core.agent import Agent


class CommandResult(Enum):
    OK = "ok"
    QUIT = "quit"
    NEEDS_RUNNER = "needs_runner"
    UNSUPPORTED = "unsupported"
    UNKNOWN = "unknown"


@dataclass
class RunnerCommand:
    """需要独立 runner 的命令，TUI 侧决定如何执行."""
    kind: str  # "goal", "plan", "exec"
    arg: str = ""
    config: dict[str, Any] = field(default_factory=dict)


class CommandHandler:
    """处理不需要独立 runner 的简单命令."""

    def __init__(self, agent: "Agent", renderer: Any) -> None:
        self._agent = agent
        self._renderer = renderer

    async def handle(self, cmd: str) -> CommandResult | RunnerCommand:
        parts = cmd.split(maxsplit=1)
        command = parts[0].lower()
        arg = parts[1].strip() if len(parts) > 1 else ""

        handler = self._COMMANDS.get(command)
        if handler is None:
            if command in self._RUNNER_COMMANDS:
                return RunnerCommand(kind=command.lstrip("/"), arg=arg)
            return CommandResult.UNKNOWN
        return await handler(self, arg)

    async def _cmd_quit(self, arg: str) -> CommandResult:
        self._renderer.render_system("再见！")
        return CommandResult.QUIT

    async def _cmd_clear(self, arg: str) -> CommandResult:
        self._agent.reset()
        self._renderer.render_system("对话历史已清空")
        return CommandResult.OK

    async def _cmd_cost(self, arg: str) -> CommandResult:
        from ..llm.pricing import estimate_cost
        from rich.panel import Panel

        usage = self._agent.total_usage
        model = self._agent.llm_client.model
        cost = estimate_cost(model, usage)

        lines = [
            f"模型: {model}",
            f"输入 tokens: {usage.input_tokens:,}",
            f"输出 tokens: {usage.output_tokens:,}",
            f"总计 tokens: {usage.total_tokens:,}",
        ]
        if usage.cached_input_tokens:
            lines.append(f"OpenAI cached input tokens: {usage.cached_input_tokens:,}")
        if usage.reasoning_tokens:
            lines.append(f"OpenAI reasoning tokens: {usage.reasoning_tokens:,}")
        if usage.cache_creation_input_tokens:
            lines.append(f"Anthropic cache creation tokens: {usage.cache_creation_input_tokens:,}")
        if usage.cache_read_input_tokens:
            lines.append(f"Anthropic cache read tokens: {usage.cache_read_input_tokens:,}")
        if cost is not None:
            suffix = "" if cost.is_complete else "（部分估算）"
            lines.append(f"估算费用: ${cost.total_cost:.4f}{suffix}")
            if cost.missing_price_items:
                lines.append("缺失价格项: " + ", ".join(cost.missing_price_items))
        else:
            lines.append("估算费用: (该模型暂无价格数据)")

        self._renderer.console.print(Panel(
            "\n".join(lines),
            title="[bold]会话消耗[/bold]",
            border_style="cyan",
        ))
        return CommandResult.OK

    async def _cmd_model(self, arg: str) -> CommandResult:
        if not arg:
            self._renderer.render_system(f"当前模型: {self._agent.llm_client.model}")
            self._renderer.render_system("用法: /model <模型名称>")
            return CommandResult.OK
        old = self._agent.llm_client.model
        self._agent.llm_client.model = arg
        self._renderer.render_system(f"模型已切换: {old} → {arg}")
        return CommandResult.OK

    async def _cmd_help(self, arg: str) -> CommandResult:
        self._renderer.render_system(
            "可用命令: /quit  /clear  /cost  /model  /memory  /save"
            "  /undo  /checkpoints  /diff  /goal  /plan  /exec"
        )
        return CommandResult.OK

    async def _cmd_memory(self, arg: str) -> CommandResult:
        from rich.panel import Panel

        lines: list[str] = []
        conv = self._agent.conversation
        lines.append(f"对话 token 数: {conv.token_count:,}")
        lines.append(f"对话消息数: {len(conv.messages)}")
        threshold = int(conv.max_tokens * conv.compress_ratio)
        lines.append(f"压缩阈值: {threshold:,} tokens")

        pm = self._agent.project_memory
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

        self._renderer.console.print(Panel(
            "\n".join(lines),
            title="[bold]记忆状态[/bold]",
            border_style="magenta",
        ))
        return CommandResult.OK

    async def _cmd_save(self, arg: str) -> CommandResult:
        pm = self._agent.project_memory
        if not pm:
            self._renderer.render_error("项目记忆未启用")
            return CommandResult.OK
        if not arg:
            self._renderer.render_system("用法: /save <要记住的信息>")
            return CommandResult.OK
        pm.add_convention(arg)
        self._renderer.render_system(f"已保存到项目记忆: {arg}")
        return CommandResult.OK

    async def _cmd_undo(self, arg: str) -> CommandResult:
        cp = self._agent.git_checkpoint
        if cp is None:
            self._renderer.render_error("Git checkpoint 未启用")
            return CommandResult.OK
        if not await cp.is_git_repo():
            self._renderer.render_error("当前目录不是 git 仓库，无法回滚")
            return CommandResult.OK
        checkpoints = await cp.list_checkpoints()
        if not checkpoints:
            self._renderer.render_system("没有找到 checkpoint，无法回滚")
            return CommandResult.OK
        latest = checkpoints[0]
        self._renderer.render_system(
            f"将回滚到 checkpoint 之前的状态: "
            f"{latest.message} ({latest.commit_hash[:8]})"
        )
        success = await cp.rollback_last()
        if success:
            self._renderer.render_system("回滚成功")
        else:
            self._renderer.render_error("回滚失败")
        return CommandResult.OK

    async def _cmd_checkpoints(self, arg: str) -> CommandResult:
        from rich.panel import Panel

        cp = self._agent.git_checkpoint
        if cp is None:
            self._renderer.render_error("Git checkpoint 未启用")
            return CommandResult.OK
        checkpoints = await cp.list_checkpoints()
        if not checkpoints:
            self._renderer.render_system("暂无 git checkpoint")
        else:
            lines = []
            for i, c in enumerate(checkpoints):
                lines.append(
                    f"  {i + 1}. [{c.commit_hash[:8]}] {c.message}  "
                    f"[dim]({c.timestamp})[/dim]"
                )
            self._renderer.console.print(Panel(
                "\n".join(lines),
                title=f"[bold]Git Checkpoints ({len(checkpoints)})[/bold]",
                border_style="cyan",
            ))
        return CommandResult.OK

    async def _cmd_diff(self, arg: str) -> CommandResult:
        cp = self._agent.git_checkpoint
        if cp is None:
            self._renderer.render_error("Git checkpoint 未启用")
            return CommandResult.OK
        checkpoints = await cp.list_checkpoints()
        if not checkpoints:
            self._renderer.render_system("暂无 checkpoint，无法显示 diff")
            return CommandResult.OK

        base_hash = checkpoints[-1].commit_hash
        from ..tools.git import _run_git

        code, diff_output = await _run_git("diff", f"{base_hash}~1", "HEAD", cwd=cp.cwd)
        if code != 0:
            code, diff_output = await _run_git("diff", base_hash, "HEAD", cwd=cp.cwd)

        if not diff_output.strip():
            self._renderer.render_system("无改动")
            return CommandResult.OK

        lines = diff_output.splitlines()
        if len(lines) > 200:
            display = "\n".join(lines[:200])
            display += f"\n\n... [截断：共 {len(lines)} 行，仅显示前 200 行]"
        else:
            display = diff_output

        from rich.syntax import Syntax
        self._renderer.console.print(Syntax(display, "diff", theme="monokai"))
        return CommandResult.OK

    _COMMANDS: dict[str, object] = {
        "/quit": _cmd_quit,
        "/exit": _cmd_quit,
        "/q": _cmd_quit,
        "/clear": _cmd_clear,
        "/cost": _cmd_cost,
        "/model": _cmd_model,
        "/help": _cmd_help,
        "/memory": _cmd_memory,
        "/save": _cmd_save,
        "/undo": _cmd_undo,
        "/checkpoints": _cmd_checkpoints,
        "/diff": _cmd_diff,
    }

    _RUNNER_COMMANDS = {"/goal", "/plan", "/exec"}
