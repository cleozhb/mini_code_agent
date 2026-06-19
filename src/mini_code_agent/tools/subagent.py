"""SubAgent 工具：在当前 Agent 内派发子 Agent 执行具体任务."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from inspect import isawaitable
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field

from .base import PermissionLevel, Tool, ToolResult

SubAgentType = Literal["coder", "explore", "plan"]


class SubAgentInput(BaseModel):
    goal: str = Field(description="子 Agent 需要完成的具体任务描述")
    context: str = Field(default="", description="当前项目状态摘要，帮助子 Agent 理解上下文")
    type: SubAgentType = Field(
        default="coder",
        description=(
            "子 Agent 类型：coder（默认，通用编码，拥有全工具集）| "
            "explore（代码库探索，只读）| plan（规划与架构设计，只读+可写计划文件）"
        ),
    )
    max_rounds: int = Field(default=25, description="子 Agent 最大工具调用轮数")


EXPLORE_SUBAGENT_PROMPT = """\
你是一个代码库探索助手。你的任务是分析、搜索和理解代码，然后给出清晰的总结。
你只有只读工具（ReadFile, Grep, ListDir, 只读 Bash, Git 查询）。
不要尝试修改任何文件或执行写操作。
专注于：理解代码结构、追踪调用链、分析依赖关系、总结发现。
"""

PLAN_SUBAGENT_PROMPT = """\
你是一个编程规划助手。严格按照用户的具体需求输出对应结果。

## 核心原则（必须遵守）

1. **严格范围控制**：只做用户明确要求的事。用户说"分析代码结构"就只输出结构分析，不要自作主张输出重构方案、目标架构、实施计划。
2. **迭代而非重写**：如果提供了已有计划文件，在其基础上做增量修改，保留未变更的部分原文不动。
3. **输出格式匹配请求**：
   - 分析类请求 → 输出分析结果（函数清单、调用关系、依赖图等）
   - 规划类请求（"怎么做"/"设计方案"/"实施计划"）→ 输出结构化方案
   - 修改计划请求 → 只修改/新增相关章节

将结果写入指定的计划文件。
"""


@dataclass
class SubAgentTool(Tool):
    """派发一个子 Agent 执行具体任务，完成后返回结果."""

    InputModel: ClassVar[type[BaseModel]] = SubAgentInput

    name: str = "SubAgent"
    description: str = (
        "派发一个子 Agent 执行任务。type 参数决定子 Agent 能力："
        "coder（默认）拥有全工具集，适合编码修改；"
        "explore 只读，适合代码库探索和分析；"
        "plan 只读+可写计划文件，适合架构规划。"
    )
    permission_level: PermissionLevel = PermissionLevel.AUTO

    llm_client: Any = None
    project_path: str = "."
    system_prompt: str = ""
    confirm_callback: Any = None
    event_callback: Any = None
    lsp_manager: Any = None
    input_channel: Any = None
    _plan_file_override: str | None = None

    async def execute(self, **kwargs: Any) -> ToolResult:
        goal: str = kwargs["goal"]
        context: str = kwargs.get("context", "")
        agent_type: SubAgentType = kwargs.get("type", "coder")
        max_rounds: int = kwargs.get("max_rounds", 25)

        from ..core.agent import Agent, AgentEvent, AgentEventType, AgentResult
        from ..llm.base import TokenUsage
        from ..safety.command_filter import CommandFilter, create_readonly_filter
        from ..safety.file_guard import FileGuard
        from ..safety.loop_guard import LoopGuard
        from .base import ToolRegistry

        registry = ToolRegistry()
        plan_file_path: str | None = None

        if agent_type == "coder":
            self._register_coder_tools(registry)
            command_filter = CommandFilter()
            file_guard = FileGuard(work_dir=self.project_path)
            prompt_prefix = self.system_prompt or "你是一个编程助手，按指令完成任务。"
        elif agent_type == "explore":
            self._register_readonly_tools(registry)
            command_filter = create_readonly_filter()
            file_guard = FileGuard(work_dir=self.project_path)
            prompt_prefix = EXPLORE_SUBAGENT_PROMPT
        else:  # plan
            plan_file_path = self._plan_file_override or self._generate_plan_file_path()
            self._register_plan_tools(registry, plan_file_path)
            command_filter = create_readonly_filter()
            file_guard = FileGuard(work_dir=self.project_path)
            if context and "已有的计划文件" in context:
                prompt_prefix = PLAN_SUBAGENT_PROMPT + f"\n\n将修改后的完整方案写入（覆盖原文件）: {plan_file_path}"
            else:
                prompt_prefix = PLAN_SUBAGENT_PROMPT + f"\n\n将方案写入: {plan_file_path}"

        loop_guard = LoopGuard(max_rounds=max_rounds)

        sub_agent = Agent(
            llm_client=self.llm_client,
            tool_registry=registry,
            system_prompt=prompt_prefix,
            confirm_callback=self.confirm_callback,
            command_filter=command_filter,
            file_guard=file_guard,
            loop_guard=loop_guard,
            project_path=self.project_path,
        )

        path_context = (
            f"当前项目根目录：{self.project_path}\n"
            "路径规则：优先使用相对路径；如果必须使用绝对路径，只能使用上面的项目根目录。"
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
        except (KeyboardInterrupt, asyncio.CancelledError):
            raise
        except Exception as e:
            return ToolResult(
                output=f"[SubAgent 异常] {type(e).__name__}: {e}",
                error=str(e),
            )

        total_tokens = result.usage.input_tokens + result.usage.output_tokens
        output = (
            f"[type: {agent_type}] [stop_reason: {result.stop_reason}]\n"
            f"[usage: input={result.usage.input_tokens} output={result.usage.output_tokens} total={total_tokens}]\n"
            f"{result.content}"
        )
        if plan_file_path:
            output += f"\n[plan_file: {plan_file_path}]"
        return ToolResult(output=output)

    def _register_coder_tools(self, registry: Any) -> None:
        from .edit import EditFileTool
        from .file_ops import ReadFileTool, WriteFileTool
        from .git import GitDiffTool, GitLogTool, GitStatusTool
        from .lsp import (
            FindReferencesTool,
            GetDiagnosticsTool,
            GetHoverInfoTool,
            GotoDefinitionTool,
        )
        from .search import GrepTool, ListDirTool
        from .shell import BashTool

        registry.register(ReadFileTool())
        registry.register(WriteFileTool())
        registry.register(EditFileTool())
        registry.register(BashTool(cwd=self.project_path))
        registry.register(GrepTool())
        registry.register(ListDirTool())
        registry.register(GitStatusTool())
        registry.register(GitDiffTool())
        registry.register(GitLogTool())
        if self.lsp_manager is not None:
            goto_def_tool = GotoDefinitionTool()
            goto_def_tool._lsp_manager = self.lsp_manager
            find_refs_tool = FindReferencesTool()
            find_refs_tool._lsp_manager = self.lsp_manager
            hover_tool = GetHoverInfoTool()
            hover_tool._lsp_manager = self.lsp_manager
            diagnostics_tool = GetDiagnosticsTool()
            diagnostics_tool._lsp_manager = self.lsp_manager
            registry.register(goto_def_tool)
            registry.register(find_refs_tool)
            registry.register(hover_tool)
            registry.register(diagnostics_tool)

    def _register_readonly_tools(self, registry: Any) -> None:
        from .file_ops import ReadFileTool
        from .git import GitDiffTool, GitLogTool, GitStatusTool
        from .search import GrepTool, ListDirTool
        from .shell import BashTool

        registry.register(ReadFileTool())
        registry.register(BashTool(cwd=self.project_path))
        registry.register(GrepTool())
        registry.register(ListDirTool())
        registry.register(GitStatusTool())
        registry.register(GitDiffTool())
        registry.register(GitLogTool())

    def _register_plan_tools(self, registry: Any, plan_file_path: str) -> None:
        from .edit import EditFileTool
        from .file_ops import ReadFileTool, WriteFileTool
        from .git import GitDiffTool, GitLogTool, GitStatusTool
        from .search import GrepTool, ListDirTool
        from .shell import BashTool

        registry.register(ReadFileTool())
        registry.register(WriteFileTool())
        registry.register(EditFileTool())
        registry.register(BashTool(cwd=self.project_path))
        registry.register(GrepTool())
        registry.register(ListDirTool())
        registry.register(GitStatusTool())
        registry.register(GitDiffTool())
        registry.register(GitLogTool())

    def _generate_plan_file_path(self) -> str:
        import time
        from pathlib import Path

        plans_dir = Path(self.project_path) / ".agent" / "plans"
        plans_dir.mkdir(parents=True, exist_ok=True)
        plan_id = f"plan-{int(time.time())}"
        return str(plans_dir / f"{plan_id}.md")

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

        async for event in sub_agent.run_stream(prompt, input_channel=self.input_channel):
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
            elif event.type == AgentEventType.INTERRUPTED:
                stop_reason = "interrupted"
                if self.input_channel is not None:
                    from ..core.runtime_input import InputKind, RuntimeInput
                    self.input_channel._queue.put_nowait(
                        RuntimeInput(kind=InputKind.PAUSE_REQUEST)
                    )
                break

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
