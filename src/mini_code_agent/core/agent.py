"""Agent 核心循环 — 整个系统的心脏."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator, Callable, Awaitable

from ..llm.base import (
    LLMClient,
    LLMResponse,
    Message,
    StreamDelta,
    StreamDeltaType,
    TokenUsage,
    ToolCall,
)
from ..llm.base import ToolResult as LLMToolResult
from ..tools.base import PermissionLevel, ToolRegistry
from ..tools.base import ToolResult as ExecToolResult

logger = logging.getLogger(__name__)

MAX_TOOL_ROUNDS = 25


class AgentError(Exception):
    """Agent 运行时错误."""


@dataclass
class AgentResult:
    """单次 run() 的返回结果."""

    content: str
    usage: TokenUsage = field(default_factory=TokenUsage)
    tool_calls_count: int = 0


# ---------------------------------------------------------------------------
# 流式事件类型
# ---------------------------------------------------------------------------


class AgentEventType(str, Enum):
    """Agent 流式循环中产出的事件类型."""

    TEXT_DELTA = "text_delta"          # LLM 生成文本片段
    TOOL_CALL_START = "tool_call_start"  # 开始一个工具调用
    TOOL_CALL_DELTA = "tool_call_delta"  # 工具调用参数增量
    TOOL_CALL_END = "tool_call_end"    # 工具调用参数接收完毕
    TOOL_RESULT = "tool_result"        # 工具执行结果
    FINISH = "finish"                  # 本轮结束


@dataclass
class AgentEvent:
    """Agent 流式循环产出的单个事件."""

    type: AgentEventType
    content: str = ""
    tool_call: ToolCall | None = None
    tool_result: ExecToolResult | None = None
    usage: TokenUsage | None = None


# 确认回调类型：传入 ToolCall，返回 (approved, edited_args_or_none)
ConfirmCallback = Callable[[str, ToolCall], Awaitable[tuple[bool, dict[str, Any] | None]]]


class Agent:
    """编程 Agent，通过 ReAct 循环协调 LLM 与工具调用."""

    def __init__(
        self,
        llm_client: LLMClient,
        tool_registry: ToolRegistry,
        system_prompt: str,
        confirm_callback: ConfirmCallback | None = None,
    ) -> None:
        self.llm_client = llm_client
        self.tool_registry = tool_registry
        self.system_prompt = system_prompt
        self.messages: list[Message] = [Message.system(system_prompt)]
        self.total_usage = TokenUsage()
        self.confirm_callback = confirm_callback

    async def run(self, user_message: str) -> AgentResult:
        """核心循环：发送消息 → 处理工具调用 → 返回最终文本.

        流程：
        1. 把 user_message 加入 messages
        2. 调用 LLM（带 tools schema）
        3. 如果 LLM 返回 tool_use：逐个执行并把结果回传，回到步骤 2
        4. 如果 LLM 返回纯文本：返回给用户
        5. 最多 MAX_TOOL_ROUNDS 轮 tool calling
        """
        self.messages.append(Message.user(user_message))

        tool_params = self.tool_registry.to_tool_params()
        round_usage = TokenUsage()
        total_tool_calls = 0

        for _round in range(MAX_TOOL_ROUNDS):
            response: LLMResponse = await self.llm_client.chat(
                messages=self.messages,
                tools=tool_params if tool_params else None,
            )

            # 累计 token 用量
            round_usage.input_tokens += response.usage.input_tokens
            round_usage.output_tokens += response.usage.output_tokens

            # 没有 tool_calls → 纯文本回复，结束循环
            if not response.tool_calls:
                self.messages.append(Message.assistant(response.content))
                self._accumulate_usage(round_usage)
                return AgentResult(
                    content=response.content,
                    usage=round_usage,
                    tool_calls_count=total_tool_calls,
                )

            # 有 tool_calls → 先把 assistant 消息（含 tool_calls）加入历史
            self.messages.append(
                Message.assistant(response.content, tool_calls=response.tool_calls)
            )

            # 逐个执行工具
            for tool_call in response.tool_calls:
                tool_result_msg = await self._execute_tool_call(tool_call)
                self.messages.append(tool_result_msg)
                total_tool_calls += 1

        # 超过最大轮数，做一次不带 tools 的收尾调用
        self._accumulate_usage(round_usage)
        logger.warning("达到最大工具调用轮数 (%d)，强制收尾", MAX_TOOL_ROUNDS)
        return await self._force_final_response(round_usage, total_tool_calls)

    async def run_stream(self, user_message: str) -> AsyncIterator[AgentEvent]:
        """流式核心循环：与 run() 逻辑一致，但以事件流形式产出中间过程.

        yields:
            AgentEvent 事件流：TEXT_DELTA / TOOL_CALL_* / TOOL_RESULT / FINISH
        """
        self.messages.append(Message.user(user_message))

        tool_params = self.tool_registry.to_tool_params()
        round_usage = TokenUsage()
        total_tool_calls = 0

        for _round in range(MAX_TOOL_ROUNDS):
            # 通过流式 API 获取响应
            full_content = ""
            tool_calls: list[ToolCall] = []
            # 按顺序存储正在构建的工具调用（列表，按 START 顺序）
            building_tools: list[ToolCall] = []
            # 用于按 id 快速查找
            building_tools_by_id: dict[str, ToolCall] = {}

            async for delta in self.llm_client.chat_stream(
                messages=self.messages,
                tools=tool_params if tool_params else None,
            ):
                if delta.type == StreamDeltaType.TEXT:
                    full_content += delta.content
                    yield AgentEvent(
                        type=AgentEventType.TEXT_DELTA,
                        content=delta.content,
                    )

                elif delta.type == StreamDeltaType.TOOL_CALL_START:
                    tc = ToolCall(
                        id=delta.tool_call_id,
                        name=delta.tool_name,
                        arguments={},
                    )
                    building_tools.append(tc)
                    if delta.tool_call_id:
                        building_tools_by_id[delta.tool_call_id] = tc
                    yield AgentEvent(
                        type=AgentEventType.TOOL_CALL_START,
                        tool_call=tc,
                    )

                elif delta.type == StreamDeltaType.TOOL_CALL_DELTA:
                    yield AgentEvent(
                        type=AgentEventType.TOOL_CALL_DELTA,
                        content=delta.content,
                    )

                elif delta.type == StreamDeltaType.TOOL_CALL_END:
                    # 按 id 查找，或回退到按 name 查找，或取最后一个
                    tc = None
                    if delta.tool_call_id and delta.tool_call_id in building_tools_by_id:
                        tc = building_tools_by_id[delta.tool_call_id]
                    elif delta.tool_name:
                        for candidate in building_tools:
                            if candidate.name == delta.tool_name and candidate not in tool_calls:
                                tc = candidate
                                break
                    if tc is None and building_tools:
                        # 兜底：取还没完成的第一个
                        for candidate in building_tools:
                            if candidate not in tool_calls:
                                tc = candidate
                                break

                    if tc is not None:
                        import json
                        try:
                            tc.arguments = json.loads(delta.content) if delta.content else {}
                        except json.JSONDecodeError:
                            tc.arguments = {}
                        tool_calls.append(tc)
                        yield AgentEvent(
                            type=AgentEventType.TOOL_CALL_END,
                            tool_call=tc,
                        )

                elif delta.type == StreamDeltaType.FINISH:
                    if delta.usage:
                        round_usage.input_tokens += delta.usage.input_tokens
                        round_usage.output_tokens += delta.usage.output_tokens

            # 没有工具调用 → 纯文本回复，结束
            if not tool_calls:
                self.messages.append(Message.assistant(full_content))
                self._accumulate_usage(round_usage)
                yield AgentEvent(
                    type=AgentEventType.FINISH,
                    usage=round_usage,
                )
                return

            # 有工具调用 → 先记录 assistant 消息
            self.messages.append(
                Message.assistant(full_content, tool_calls=tool_calls)
            )

            # 逐个执行工具
            for tool_call in tool_calls:
                tool_result_msg, exec_result = await self._execute_tool_call_with_result(tool_call)
                self.messages.append(tool_result_msg)
                total_tool_calls += 1
                yield AgentEvent(
                    type=AgentEventType.TOOL_RESULT,
                    tool_call=tool_call,
                    tool_result=exec_result,
                )

        # 超出最大轮数 → 收尾
        self._accumulate_usage(round_usage)
        logger.warning("达到最大工具调用轮数 (%d)，强制收尾", MAX_TOOL_ROUNDS)

        # 强制收尾也用流式
        self.messages.append(
            Message.user(
                "你已经进行了很多轮工具调用。请根据目前获得的信息，"
                "直接给出最终回答，不要再调用工具。"
            )
        )
        async for delta in self.llm_client.chat_stream(
            messages=self.messages, tools=None
        ):
            if delta.type == StreamDeltaType.TEXT:
                full_content += delta.content
                yield AgentEvent(
                    type=AgentEventType.TEXT_DELTA,
                    content=delta.content,
                )
            elif delta.type == StreamDeltaType.FINISH and delta.usage:
                round_usage.input_tokens += delta.usage.input_tokens
                round_usage.output_tokens += delta.usage.output_tokens

        self.messages.append(Message.assistant(full_content))
        self._accumulate_usage(round_usage)
        yield AgentEvent(type=AgentEventType.FINISH, usage=round_usage)

    async def _execute_tool_call(self, tool_call: ToolCall) -> Message:
        """执行单个工具调用，返回 tool result 消息."""
        tool = self.tool_registry.get(tool_call.name)

        if tool is None:
            return Message.tool(
                LLMToolResult(
                    tool_call_id=tool_call.id,
                    content=f"错误：未找到工具 '{tool_call.name}'",
                    is_error=True,
                )
            )

        # 检查权限
        if tool.permission_level == PermissionLevel.DENY:
            return Message.tool(
                LLMToolResult(
                    tool_call_id=tool_call.id,
                    content=f"错误：工具 '{tool_call.name}' 被禁止执行",
                    is_error=True,
                )
            )

        # CONFIRM 级别的工具暂时先自动执行，后续加确认逻辑
        if tool.permission_level == PermissionLevel.CONFIRM:
            logger.info(
                "工具 '%s' 需要确认（当前自动放行）: %s",
                tool_call.name,
                tool_call.arguments,
            )

        # 执行工具
        try:
            result: ExecToolResult = await tool.execute(**tool_call.arguments)
        except Exception as e:
            logger.exception("工具 '%s' 执行异常", tool_call.name)
            return Message.tool(
                LLMToolResult(
                    tool_call_id=tool_call.id,
                    content=f"工具执行异常: {type(e).__name__}: {e}",
                    is_error=True,
                )
            )

        # 将 tools.base.ToolResult 转为 llm.base.ToolResult
        if result.is_error:
            content = result.error or "未知错误"
            is_error = True
        else:
            content = result.output
            is_error = False

        return Message.tool(
            LLMToolResult(
                tool_call_id=tool_call.id,
                content=content,
                is_error=is_error,
            )
        )

    async def _execute_tool_call_with_result(
        self, tool_call: ToolCall
    ) -> tuple[Message, ExecToolResult]:
        """执行工具调用，同时返回 Message 和原始 ExecToolResult（供 CLI 展示）."""
        tool = self.tool_registry.get(tool_call.name)

        if tool is None:
            dummy = ExecToolResult(output="", error=f"未找到工具 '{tool_call.name}'")
            msg = Message.tool(
                LLMToolResult(
                    tool_call_id=tool_call.id,
                    content=f"错误：未找到工具 '{tool_call.name}'",
                    is_error=True,
                )
            )
            return msg, dummy

        # 权限检查
        if tool.permission_level == PermissionLevel.DENY:
            dummy = ExecToolResult(output="", error=f"工具 '{tool_call.name}' 被禁止执行")
            msg = Message.tool(
                LLMToolResult(
                    tool_call_id=tool_call.id,
                    content=f"错误：工具 '{tool_call.name}' 被禁止执行",
                    is_error=True,
                )
            )
            return msg, dummy

        # CONFIRM 级别 — 通过回调让 CLI 确认
        if tool.permission_level == PermissionLevel.CONFIRM and self.confirm_callback:
            approved, edited_args = await self.confirm_callback(tool.name, tool_call)
            if not approved:
                dummy = ExecToolResult(output="", error="用户拒绝了此操作")
                msg = Message.tool(
                    LLMToolResult(
                        tool_call_id=tool_call.id,
                        content="用户拒绝了此操作",
                        is_error=True,
                    )
                )
                return msg, dummy
            if edited_args is not None:
                tool_call.arguments = edited_args

        # 执行工具
        try:
            result: ExecToolResult = await tool.execute(**tool_call.arguments)
        except Exception as e:
            logger.exception("工具 '%s' 执行异常", tool_call.name)
            err_result = ExecToolResult(
                output="", error=f"{type(e).__name__}: {e}"
            )
            msg = Message.tool(
                LLMToolResult(
                    tool_call_id=tool_call.id,
                    content=f"工具执行异常: {type(e).__name__}: {e}",
                    is_error=True,
                )
            )
            return msg, err_result

        # 转换
        if result.is_error:
            content = result.error or "未知错误"
            is_error = True
        else:
            content = result.output
            is_error = False

        msg = Message.tool(
            LLMToolResult(
                tool_call_id=tool_call.id,
                content=content,
                is_error=is_error,
            )
        )
        return msg, result

    async def _force_final_response(
        self,
        accumulated_usage: TokenUsage,
        total_tool_calls: int,
    ) -> AgentResult:
        """超出轮数限制时，不带 tools 做一次收尾调用."""
        self.messages.append(
            Message.user(
                "你已经进行了很多轮工具调用。请根据目前获得的信息，"
                "直接给出最终回答，不要再调用工具。"
            )
        )
        response = await self.llm_client.chat(messages=self.messages, tools=None)
        accumulated_usage.input_tokens += response.usage.input_tokens
        accumulated_usage.output_tokens += response.usage.output_tokens
        self.messages.append(Message.assistant(response.content))
        self._accumulate_usage(accumulated_usage)
        return AgentResult(
            content=response.content,
            usage=accumulated_usage,
            tool_calls_count=total_tool_calls,
        )

    def _accumulate_usage(self, usage: TokenUsage) -> None:
        """累计本轮 token 用量到全局."""
        self.total_usage.input_tokens += usage.input_tokens
        self.total_usage.output_tokens += usage.output_tokens

    def reset(self) -> None:
        """重置对话历史（保留 system prompt）."""
        self.messages = [Message.system(self.system_prompt)]
