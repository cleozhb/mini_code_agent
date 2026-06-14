"""SubAgent 工具：在当前 Agent 内派发子 Agent 执行具体任务."""

from __future__ import annotations

from dataclasses import dataclass, field
from inspect import isawaitable
from typing import Any, ClassVar

from pydantic import BaseModel, Field

from .base import PermissionLevel, Tool, ToolResult


class SubAgentInput(BaseModel):
    goal: str = Field(description="子 Agent 需要完成的具体任务描述")
    context: str = Field(default="", description="当前项目状态摘要，帮助子 Agent 理解上下文")
    max_rounds: int = Field(default=25, description="子 Agent 最大工具调用轮数")


@dataclass
class SubAgentTool(Tool):
    """派发一个子 Agent 执行具体任务，完成后返回结果."""

    InputModel: ClassVar[type[BaseModel]] = SubAgentInput

    name: str = "SubAgent"
    description: str = (
        "派发一个子 Agent 去完成一个具体的编码任务。"
        "子 Agent 拥有完整的文件读写、Shell、Git 等工具。"
        "适合 1-2 个文件级别的改动粒度。"
        "返回子 Agent 的执行结果摘要和停止原因。"
    )
    permission_level: PermissionLevel = PermissionLevel.AUTO

    # 构造时注入
    llm_client: Any = None
    project_path: str = "."
    system_prompt: str = ""
    confirm_callback: Any = None
    event_callback: Any = None

    async def execute(self, **kwargs: Any) -> ToolResult:
        goal: str = kwargs["goal"]
        context: str = kwargs.get("context", "")
        max_rounds: int = kwargs.get("max_rounds", 25)

        from ..core.agent import Agent, AgentEvent, AgentEventType, AgentResult
        from ..llm.base import TokenUsage
        from ..safety.command_filter import CommandFilter
        from ..safety.file_guard import FileGuard
        from ..safety.loop_guard import LoopGuard
        from .base import ToolRegistry
        from .edit import EditFileTool
        from .file_ops import ReadFileTool, WriteFileTool
        from .git import GitDiffTool, GitLogTool, GitStatusTool
        from .search import GrepTool, ListDirTool
        from .shell import BashTool

        registry = ToolRegistry()
        registry.register(ReadFileTool())
        registry.register(WriteFileTool())
        registry.register(EditFileTool())
        registry.register(BashTool(cwd=self.project_path))
        registry.register(GrepTool())
        registry.register(ListDirTool())
        registry.register(GitStatusTool())
        registry.register(GitDiffTool())
        registry.register(GitLogTool())

        command_filter = CommandFilter()
        file_guard = FileGuard(work_dir=self.project_path)
        loop_guard = LoopGuard(max_rounds=max_rounds)

        sub_agent = Agent(
            llm_client=self.llm_client,
            tool_registry=registry,
            system_prompt=self.system_prompt or "你是一个编程助手，按指令完成任务。",
            confirm_callback=self.confirm_callback,
            command_filter=command_filter,
            file_guard=file_guard,
            loop_guard=loop_guard,
            project_path=self.project_path,
        )

        path_context = (
            f"当前项目根目录：{self.project_path}\n"
            "路径规则：优先使用相对路径；如果必须使用绝对路径，只能使用上面的项目根目录。"
            "不要编造 /home/user/repo、/workspace、/Users/boxiao 等未验证路径。"
        )
        prompt_parts = [path_context]
        if context:
            prompt_parts.append(context)
        prompt_parts.append(f"你的任务：{goal}")
        prompt = "\n\n".join(prompt_parts)

        try:
            if self.event_callback is not None:
                result = await self._run_streaming_subagent(sub_agent, prompt)
            else:
                result = await sub_agent.run(prompt)
        except Exception as e:
            return ToolResult(
                output=f"[SubAgent 异常] {type(e).__name__}: {e}",
                error=str(e),
            )

        total_tokens = result.usage.input_tokens + result.usage.output_tokens
        return ToolResult(
            output=(
                f"[stop_reason: {result.stop_reason}]\n"
                f"[usage: input_tokens={result.usage.input_tokens} "
                f"output_tokens={result.usage.output_tokens} "
                f"total_tokens={total_tokens}]\n"
                f"{result.content}"
            ),
        )

    async def _run_streaming_subagent(
        self,
        sub_agent: Any,
        prompt: str,
    ) -> AgentResult:
        from ..core.agent import AgentEvent, AgentEventType, AgentResult
        from ..llm.base import TokenUsage

        content = ""
        usage = TokenUsage()
        tool_calls_count = 0
        tool_calls_errors = 0
        stop_reason = "ok"

        async for event in sub_agent.run_stream(prompt):
            await self._emit_event(event)
            if event.type == AgentEventType.TEXT_DELTA:
                content += event.content
            elif event.type == AgentEventType.TOOL_RESULT:
                tool_calls_count += 1
                if event.tool_result is not None and event.tool_result.is_error:
                    tool_calls_errors += 1
            elif event.type == AgentEventType.FINISH:
                stop_reason = event.content or "ok"
                if event.usage is not None:
                    usage = event.usage

        return AgentResult(
            content=content,
            usage=usage,
            tool_calls_count=tool_calls_count,
            tool_calls_errors=tool_calls_errors,
            stop_reason=stop_reason,
        )

    async def _emit_event(self, event: Any) -> None:
        if self.event_callback is None:
            return
        maybe_awaitable = self.event_callback(event)
        if isawaitable(maybe_awaitable):
            await maybe_awaitable
