"""Agent 核心循环 — 整个系统的心脏."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator, Callable, Awaitable, Literal

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
from ..memory.conversation import ConversationManager
from ..memory.project_memory import ProjectMemory
from ..safety.command_filter import CommandFilter, SafetyLevel
from ..safety.file_guard import FileGuard
from ..safety.loop_guard import LoopGuard
from ..tools.base import PermissionLevel, ToolRegistry
from ..tools.base import ToolResult as ExecToolResult
from .planner import Plan, Planner, PlannerError
from .retry import RetryController
from .verifier import VerificationResult, Verifier

logger = logging.getLogger(__name__)

MAX_TOOL_ROUNDS = 25

# 会改动文件的工具名；用于追踪本轮任务修改过哪些文件
WRITE_TOOL_NAMES = {"WriteFile", "EditFile", "write_file", "edit_file"}


class AgentError(Exception):
    """Agent 运行时错误."""


StopReason = Literal["ok", "max_rounds", "max_tokens", "timeout", "error"]


@dataclass
class AgentResult:
    """单次 run() 的返回结果."""

    content: str
    usage: TokenUsage = field(default_factory=TokenUsage)
    tool_calls_count: int = 0
    # 结束原因：
    # - "ok"          正常结束（LLM 给出纯文本回复）
    # - "max_rounds"  达到最大工具调用轮数，被强制收尾
    # - "max_tokens"  达到 token 预算硬上限
    # - "timeout"     超过 max_wall_time_seconds 墙钟上限
    # - "error"       发生异常
    stop_reason: StopReason = "ok"
    # 工具调用里 is_error=True 的次数（含安全拦截、权限拒绝、工具异常、Bash 超时等，
    # 但不含 Bash 非零 exit —— 那是业务信号，附在 output 的 `[exit code: N]` 里）
    tool_calls_errors: int = 0
    # Verifier 触发的统计；从未触发时为 None / 0
    #   verifier_attempts      = 跑过几次 verifier（>=1 即触发过）
    #   verifier_first_passed  = 首次 verifier 是否 pass
    #   verifier_final_passed  = 最后一次 verifier 是否 pass
    verifier_attempts: int = 0
    verifier_first_passed: bool | None = None
    verifier_final_passed: bool | None = None


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


# 确认回调类型：传入 (tool_name, ToolCall, SafetyLevel)，返回 (approved, edited_args_or_none)
ConfirmCallback = Callable[
    [str, ToolCall, SafetyLevel],
    Awaitable[tuple[bool, dict[str, Any] | None]],
]


# Plan mode 回调类型
# - plan_confirm_callback: 传入生成好的 Plan，返回 (approved, final_plan_or_none)
# - plan_progress_callback: 在每一步开始/结束时回调 (index_1based, total, step, phase, success)
# - plan_replan_callback: 某步失败时调用，返回 "replan" | "continue" | "abort"
PlanConfirmCallback = Callable[
    [Plan],
    Awaitable[tuple[bool, Plan | None]],
]
PlanProgressCallback = Callable[
    [int, int, Any, str, bool],  # (idx, total, step, phase: "start"|"end", success)
    Awaitable[None],
]
PlanReplanCallback = Callable[
    [Plan, int, str],  # (plan, failed_step_1based, last_content)
    Awaitable[str],  # "replan" | "continue" | "abort"
]


class Agent:
    """编程 Agent，通过 ReAct 循环协调 LLM 与工具调用."""

    def __init__(
        self,
        llm_client: LLMClient,
        tool_registry: ToolRegistry,
        system_prompt: str,
        confirm_callback: ConfirmCallback | None = None,
        command_filter: CommandFilter | None = None,
        file_guard: FileGuard | None = None,
        loop_guard: LoopGuard | None = None,
        project_memory: ProjectMemory | None = None,
        verifier: Verifier | None = None,
        retry_controller: RetryController | None = None,
        project_path: str | None = None,
        plan_mode: bool = False,
        planner: Planner | None = None,
        plan_confirm_callback: PlanConfirmCallback | None = None,
        plan_progress_callback: PlanProgressCallback | None = None,
        plan_replan_callback: PlanReplanCallback | None = None,
        max_wall_time_seconds: float | None = None,
    ) -> None:
        self.llm_client = llm_client
        self.tool_registry = tool_registry
        self.system_prompt = system_prompt
        self.total_usage = TokenUsage()
        self.confirm_callback = confirm_callback
        self.command_filter = command_filter
        self.file_guard = file_guard
        self.loop_guard = loop_guard
        self.project_memory = project_memory
        self.verifier = verifier
        # 如果外部提供了 verifier 但没给 retry_controller，默认创建一个
        self.retry_controller = retry_controller or (
            RetryController() if verifier is not None else None
        )
        self.project_path = project_path
        self.max_wall_time_seconds = max_wall_time_seconds

        # Plan mode 相关
        self.plan_mode = plan_mode
        self.planner = planner
        self.plan_confirm_callback = plan_confirm_callback
        self.plan_progress_callback = plan_progress_callback
        self.plan_replan_callback = plan_replan_callback

        # 当前任务期间追踪改动过哪些文件（供 Verifier 使用）
        self._files_changed: list[str] = []

        # 使用 ConversationManager 管理消息
        self.conversation = ConversationManager(llm_client=llm_client)
        self.conversation.init_system(self._build_full_system_prompt())

    def _build_full_system_prompt(self) -> str:
        """拼接 system prompt + 项目记忆."""
        prompt = self.system_prompt
        if self.project_memory:
            memory_text = self.project_memory.format_for_prompt()
            if memory_text:
                prompt += f"\n\n<project-memory>\n{memory_text}\n</project-memory>"
        return prompt

    @property
    def messages(self) -> list[Message]:
        """消息列表（委托给 ConversationManager）."""
        return self.conversation.messages

    async def run(self, user_message: str) -> AgentResult:
        """核心循环入口：如果设置了 max_wall_time_seconds，用 wait_for 包裹整个执行.

        超时后返回 stop_reason="timeout" 的 AgentResult，
        注意：正在运行的 Bash 子进程可能残留（尽力而为）。
        """
        if self.max_wall_time_seconds is None:
            return await self._run_impl(user_message)

        try:
            return await asyncio.wait_for(
                self._run_impl(user_message),
                timeout=self.max_wall_time_seconds,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "Agent 运行超时（%.1fs），返回部分结果",
                self.max_wall_time_seconds,
            )
            return AgentResult(
                content=(
                    f"[超时] Agent 超过 {self.max_wall_time_seconds:.0f}s "
                    f"墙钟上限，已中止。"
                ),
                usage=TokenUsage(),
                tool_calls_count=0,
                stop_reason="timeout",
            )

    async def _run_impl(self, user_message: str) -> AgentResult:
        """核心循环：发送消息 → 处理工具调用 → 返回最终文本.

        如果配置了 Verifier，纯文本回复之后会自动对改动的文件跑一次验证，
        失败时在 RetryController 允许的范围内自动把错误回传给 LLM 继续修复。
        """
        # Plan mode 分支：先生成计划，让用户确认，再按步骤执行
        if self.plan_mode and self.planner is not None:
            return await self._run_with_plan(user_message)

        # 每次顶层 run() 开始时重置文件追踪与重试计数
        self._files_changed = []
        if self.retry_controller is not None:
            self.retry_controller.reset()

        total_usage = TokenUsage()
        total_tool_calls = 0
        total_tool_errors = 0
        last_content = ""
        last_stop_reason: StopReason = "ok"
        current_user_msg = user_message

        # verifier 统计
        verifier_attempts = 0
        verifier_first_passed: bool | None = None
        verifier_final_passed: bool | None = None

        while True:
            single = await self._run_once(current_user_msg)
            total_usage.input_tokens += single.usage.input_tokens
            total_usage.output_tokens += single.usage.output_tokens
            total_tool_calls += single.tool_calls_count
            total_tool_errors += single.tool_calls_errors
            last_content = single.content
            last_stop_reason = single.stop_reason

            # 异常终止（max_tokens / max_rounds）— 不再走 verifier 重试
            if single.stop_reason != "ok":
                break

            # 没有 verifier，或者本轮没改过文件 → 不触发验证
            if (
                self.verifier is None
                or self.project_path is None
                or not self._files_changed
            ):
                break

            vr: VerificationResult = await self.verifier.verify_code_change(
                self._files_changed, self.project_path
            )
            verifier_attempts += 1
            if verifier_first_passed is None:
                verifier_first_passed = vr.passed
            verifier_final_passed = vr.passed

            if vr.passed:
                logger.info("验证通过（%d 个文件）", len(self._files_changed))
                break

            logger.info(
                "验证失败: %d 个错误；尝试次数: %d/%d",
                len(vr.errors),
                self.retry_controller.attempts_count + 1 if self.retry_controller else 0,
                self.retry_controller.max_retries if self.retry_controller else 0,
            )

            # 没有重试控制器或已达上限 → 交给用户
            if self.retry_controller is None:
                break

            self.retry_controller.record_attempt(vr.errors, last_content)
            if not self.retry_controller.can_retry():
                giveup = self.retry_controller.build_giveup_summary()
                # 把最终说明作为 assistant 消息写进对话历史
                self.conversation.append(Message.assistant(giveup))
                last_content = giveup
                break

            # 还能重试 — 构造回传提示并清空文件追踪
            retry_prompt = self.retry_controller.build_retry_prompt(vr.errors)
            self._files_changed = []
            current_user_msg = retry_prompt

        return AgentResult(
            content=last_content,
            usage=total_usage,
            tool_calls_count=total_tool_calls,
            stop_reason=last_stop_reason,
            tool_calls_errors=total_tool_errors,
            verifier_attempts=verifier_attempts,
            verifier_first_passed=verifier_first_passed,
            verifier_final_passed=verifier_final_passed,
        )

    async def _run_with_plan(self, user_message: str) -> AgentResult:
        """Plan mode：先生成计划 → 用户确认 → 按步骤执行.

        流程：
        1. 调用 Planner 生成 Plan
        2. 通过 plan_confirm_callback 让用户确认 / 编辑 / 放弃
        3. 逐步执行每一步（每步一次 _run_once）
        4. 每一步用 plan_progress_callback 通知进度
        5. 如果某步失败（tool 报错），通过 plan_replan_callback 询问是否重规划
        """
        assert self.planner is not None

        # 生成计划
        try:
            plan = await self.planner.plan(user_message)
        except PlannerError as e:
            logger.warning("计划生成失败，回退到普通模式: %s", e)
            self.plan_mode = False
            try:
                return await self.run(user_message)
            finally:
                self.plan_mode = True

        # 请用户确认
        if self.plan_confirm_callback:
            approved, edited = await self.plan_confirm_callback(plan)
            if not approved:
                return AgentResult(
                    content="[已放弃] 用户取消了计划。",
                    usage=TokenUsage(),
                    tool_calls_count=0,
                )
            if edited is not None:
                plan = edited

        # 把计划整体推给对话：后续每一步都基于这个上下文
        plan_intro = (
            "现在进入分步执行模式。以下是已确认的执行计划：\n\n"
            f"{plan.format_for_prompt()}\n\n"
            "接下来我会按顺序告诉你每一步要做什么，"
            "你只需针对当前步骤行动，必要时调用工具，完成后简要说明即可。"
        )
        self.conversation.append(Message.user(plan_intro))
        self.conversation.append(Message.assistant("收到计划，我会按步骤执行。"))

        total_usage = TokenUsage()
        total_tool_calls = 0
        last_content = ""

        # 重置重试控制器（整次 plan run 共用一个，顶层）
        if self.retry_controller is not None:
            self.retry_controller.reset()
        self._files_changed = []

        i = 0
        while i < len(plan.steps):
            step = plan.steps[i]
            step_num = i + 1
            total = len(plan.steps)

            if self.plan_progress_callback:
                await self.plan_progress_callback(step_num, total, step, "start", True)

            step_prompt_lines = [
                f"【第 {step_num}/{total} 步】{step.description}",
            ]
            if step.files_involved:
                step_prompt_lines.append(
                    f"涉及文件：{', '.join(step.files_involved)}"
                )
            if step.tools_needed:
                step_prompt_lines.append(
                    f"可能用到的工具：{', '.join(step.tools_needed)}"
                )
            if step.verification:
                step_prompt_lines.append(f"完成后请验证：{step.verification}")
            step_prompt_lines.append("请只完成这一步，不要跨步执行。")
            step_prompt = "\n".join(step_prompt_lines)

            single = await self._run_once(step_prompt)
            total_usage.input_tokens += single.usage.input_tokens
            total_usage.output_tokens += single.usage.output_tokens
            total_tool_calls += single.tool_calls_count
            last_content = single.content

            success = not self._last_step_had_tool_error()
            if self.plan_progress_callback:
                await self.plan_progress_callback(
                    step_num, total, step, "end", success
                )

            if not success and self.plan_replan_callback is not None:
                choice = await self.plan_replan_callback(plan, step_num, last_content)
                if choice == "replan":
                    # 基于剩余目标和上一步的错误重新规划
                    remaining_goal = (
                        f"原始目标：{plan.goal}\n"
                        f"已完成步骤：{step_num - 1}/{total}\n"
                        f"当前失败步骤：{step.description}\n"
                        f"请对剩余工作重新规划。"
                    )
                    try:
                        new_plan = await self.planner.plan(remaining_goal)
                    except PlannerError as e:
                        logger.warning("重规划失败: %s", e)
                        break

                    if self.plan_confirm_callback:
                        approved, edited = await self.plan_confirm_callback(new_plan)
                        if not approved:
                            last_content = "[已放弃] 用户在重规划环节取消。"
                            break
                        if edited is not None:
                            new_plan = edited

                    # 把新计划接到当前对话
                    self.conversation.append(
                        Message.user(
                            "基于当前进展，重新规划后的剩余步骤：\n\n"
                            f"{new_plan.format_for_prompt()}"
                        )
                    )
                    self.conversation.append(
                        Message.assistant("收到新计划，继续按步骤执行。")
                    )
                    plan = new_plan
                    i = 0
                    continue
                if choice == "abort":
                    last_content = last_content or "[已放弃] 用户在某步失败后放弃。"
                    break
                # "continue" → 跳过失败步骤，继续下一步

            i += 1

        return AgentResult(
            content=last_content or "计划执行完毕。",
            usage=total_usage,
            tool_calls_count=total_tool_calls,
        )

    def _last_step_had_tool_error(self) -> bool:
        """粗略判断最近一步是否遇到 tool 错误：扫最近几条 tool 消息."""
        for msg in reversed(self.conversation.messages):
            if msg.role.value == "tool" and msg.tool_result is not None:
                return bool(msg.tool_result.is_error)
            if msg.role.value == "user":
                # 已经回到步骤边界
                break
        return False

    async def _run_once(self, user_message: str) -> AgentResult:
        """单轮对话：发送消息 → 处理工具调用 → 返回最终文本.

        流程：
        1. 把 user_message 加入 messages
        2. 调用 LLM（带 tools schema）
        3. 如果 LLM 返回 tool_use：逐个执行并把结果回传，回到步骤 2
        4. 如果 LLM 返回纯文本：返回给用户
        5. 最多 MAX_TOOL_ROUNDS 轮 tool calling
        """
        self.conversation.append(Message.user(user_message))

        tool_params = self.tool_registry.to_tool_params()
        round_usage = TokenUsage()
        total_tool_calls = 0
        total_tool_errors = 0

        max_rounds = self.loop_guard.max_rounds if self.loop_guard else MAX_TOOL_ROUNDS

        for _round in range(max_rounds):
            # LoopGuard 轮数检查
            if self.loop_guard:
                limit_msg = self.loop_guard.next_round()
                if limit_msg:
                    self.conversation.append(Message.user(limit_msg))
                    break

            response: LLMResponse = await self.llm_client.chat(
                messages=self.messages,
                tools=tool_params if tool_params else None,
            )

            # 累计 token 用量
            round_usage.input_tokens += response.usage.input_tokens
            round_usage.output_tokens += response.usage.output_tokens

            # LoopGuard token 预算检查
            if self.loop_guard:
                token_msg = self.loop_guard.add_tokens(
                    response.usage.input_tokens + response.usage.output_tokens
                )
                if token_msg:
                    logger.warning(token_msg)
                # 硬超限：push 提示后直接停掉本轮，且标记 stop_reason
                if self.loop_guard.total_tokens >= self.loop_guard.max_tokens:
                    if response.content:
                        self.conversation.append(
                            Message.assistant(response.content)
                        )
                    self._accumulate_usage(round_usage)
                    await self._maybe_compress()
                    return AgentResult(
                        content=response.content or (token_msg or "[预算耗尽]"),
                        usage=round_usage,
                        tool_calls_count=total_tool_calls,
                        stop_reason="max_tokens",
                        tool_calls_errors=total_tool_errors,
                    )

            # 没有 tool_calls → 纯文本回复，结束循环
            if not response.tool_calls:
                self.conversation.append(Message.assistant(response.content))
                self._accumulate_usage(round_usage)
                await self._maybe_compress()
                return AgentResult(
                    content=response.content,
                    usage=round_usage,
                    tool_calls_count=total_tool_calls,
                    stop_reason="ok",
                    tool_calls_errors=total_tool_errors,
                )

            # 有 tool_calls → 先把 assistant 消息（含 tool_calls）加入历史
            self.conversation.append(
                Message.assistant(response.content, tool_calls=response.tool_calls)
            )

            # 逐个执行工具
            for tool_call in response.tool_calls:
                tool_result_msg = await self._execute_tool_call(tool_call)
                self.conversation.append(tool_result_msg)
                total_tool_calls += 1
                if (
                    tool_result_msg.tool_result is not None
                    and tool_result_msg.tool_result.is_error
                ):
                    total_tool_errors += 1

        # 超过最大轮数，做一次不带 tools 的收尾调用
        self._accumulate_usage(round_usage)
        logger.warning("达到最大工具调用轮数，强制收尾")
        result = await self._force_final_response(
            round_usage, total_tool_calls, total_tool_errors
        )
        await self._maybe_compress()
        return result

    async def run_stream(self, user_message: str) -> AsyncIterator[AgentEvent]:
        """流式核心循环：与 run() 逻辑一致，但以事件流形式产出中间过程.

        yields:
            AgentEvent 事件流：TEXT_DELTA / TOOL_CALL_* / TOOL_RESULT / FINISH
        """
        self.conversation.append(Message.user(user_message))

        tool_params = self.tool_registry.to_tool_params()
        round_usage = TokenUsage()
        total_tool_calls = 0

        max_rounds = self.loop_guard.max_rounds if self.loop_guard else MAX_TOOL_ROUNDS

        for _round in range(max_rounds):
            # LoopGuard 轮数检查
            if self.loop_guard:
                limit_msg = self.loop_guard.next_round()
                if limit_msg:
                    self.conversation.append(Message.user(limit_msg))
                    break

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
                        # LoopGuard token 预算检查
                        if self.loop_guard:
                            token_msg = self.loop_guard.add_tokens(
                                delta.usage.input_tokens + delta.usage.output_tokens
                            )
                            if token_msg:
                                logger.warning(token_msg)

            # 没有工具调用 → 纯文本回复，结束
            if not tool_calls:
                self.conversation.append(Message.assistant(full_content))
                self._accumulate_usage(round_usage)
                await self._maybe_compress()
                yield AgentEvent(
                    type=AgentEventType.FINISH,
                    usage=round_usage,
                )
                return

            # 有工具调用 → 先记录 assistant 消息
            self.conversation.append(
                Message.assistant(full_content, tool_calls=tool_calls)
            )

            # 逐个执行工具
            for tool_call in tool_calls:
                tool_result_msg, exec_result = await self._execute_tool_call_with_result(tool_call)
                self.conversation.append(tool_result_msg)
                total_tool_calls += 1
                yield AgentEvent(
                    type=AgentEventType.TOOL_RESULT,
                    tool_call=tool_call,
                    tool_result=exec_result,
                )

        # 超出最大轮数 → 收尾
        self._accumulate_usage(round_usage)
        logger.warning("达到最大工具调用轮数，强制收尾")

        # 强制收尾也用流式
        self.conversation.append(
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

        self.conversation.append(Message.assistant(full_content))
        self._accumulate_usage(round_usage)
        await self._maybe_compress()
        yield AgentEvent(type=AgentEventType.FINISH, usage=round_usage)

    async def _maybe_compress(self) -> None:
        """检查是否需要压缩对话历史，需要则执行."""
        if self.conversation.needs_compression():
            compressed = await self.conversation.compress()
            if compressed:
                logger.info(
                    "对话已压缩，当前 token 数: %d", self.conversation.token_count
                )

    def _check_safety(
        self, tool_name: str, tool_call: ToolCall
    ) -> tuple[SafetyLevel, str | None]:
        """对工具调用进行安全检查.

        Returns:
            (safety_level, block_reason) — SAFE/NEEDS_CONFIRM 时 reason 为 None
        """
        args = tool_call.arguments

        # 1) Bash 命令过滤
        if tool_name == "Bash" and self.command_filter:
            command = args.get("command", "")
            level = self.command_filter.is_safe(command)
            if level == SafetyLevel.BLOCKED:
                reason = self.command_filter.get_block_reason(command)
                return SafetyLevel.BLOCKED, reason or "危险命令被拦截"
            if level == SafetyLevel.NEEDS_CONFIRM:
                return SafetyLevel.NEEDS_CONFIRM, None
            # SAFE → 降低权限需求（跳过确认）
            return SafetyLevel.SAFE, None

        # 2) 文件操作保护
        if self.file_guard and tool_name in ("WriteFile", "EditFile"):
            path = args.get("path", "")
            if path:
                verdict, reason = self.file_guard.check_write(path)
                if verdict == "blocked":
                    return SafetyLevel.BLOCKED, reason
                if verdict == "needs_confirm":
                    return SafetyLevel.NEEDS_CONFIRM, None

        if self.file_guard and tool_name == "ReadFile":
            path = args.get("path", "")
            if path:
                allowed, reason = self.file_guard.check_read(path)
                if not allowed:
                    return SafetyLevel.BLOCKED, reason

        # 3) 重复调用检测
        if self.loop_guard:
            warning = self.loop_guard.record_tool_call(tool_name, args)
            if warning:
                logger.warning(warning)
                # 重复检测只警告，不拦截

        # 默认保持工具原有权限级别
        return SafetyLevel.NEEDS_CONFIRM, None

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

        # 安全检查
        safety_level, block_reason = self._check_safety(tool.name, tool_call)
        if safety_level == SafetyLevel.BLOCKED:
            return Message.tool(
                LLMToolResult(
                    tool_call_id=tool_call.id,
                    content=f"[安全拦截] {block_reason}",
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
        if tool.permission_level == PermissionLevel.CONFIRM and safety_level != SafetyLevel.SAFE:
            logger.info(
                "工具 '%s' 需要确认（当前自动放行）: %s",
                tool_call.name,
                tool_call.arguments,
            )

        # 写操作前备份
        if self.file_guard and tool.name in ("WriteFile", "EditFile"):
            path = tool_call.arguments.get("path", "")
            if path:
                self.file_guard.pre_write(path)

        # 执行工具
        try:
            result: ExecToolResult = await tool.run(tool_call.arguments)
        except Exception as e:
            logger.exception("工具 '%s' 执行异常", tool_call.name)
            return Message.tool(
                LLMToolResult(
                    tool_call_id=tool_call.id,
                    content=f"工具执行异常: {type(e).__name__}: {e}",
                    is_error=True,
                )
            )

        # 追踪文件改动
        self._track_file_change(tool.name, tool_call.arguments, result)

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

        # 安全检查
        safety_level, block_reason = self._check_safety(tool.name, tool_call)
        if safety_level == SafetyLevel.BLOCKED:
            dummy = ExecToolResult(output="", error=block_reason or "危险操作被拦截")
            msg = Message.tool(
                LLMToolResult(
                    tool_call_id=tool_call.id,
                    content=f"[安全拦截] {block_reason}",
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

        # CONFIRM 级别 — 白名单命令跳过确认，其他通过回调让 CLI 确认
        needs_confirm = (
            tool.permission_level == PermissionLevel.CONFIRM
            and safety_level != SafetyLevel.SAFE
        )
        if needs_confirm and self.confirm_callback:
            approved, edited_args = await self.confirm_callback(
                tool.name, tool_call, safety_level,
            )
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

        # 写操作前备份
        if self.file_guard and tool.name in ("WriteFile", "EditFile"):
            path = tool_call.arguments.get("path", "")
            if path:
                self.file_guard.pre_write(path)

        # 执行工具
        try:
            result: ExecToolResult = await tool.run(tool_call.arguments)
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

        # 追踪文件改动
        self._track_file_change(tool.name, tool_call.arguments, result)

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

    def _track_file_change(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        result: ExecToolResult,
    ) -> None:
        """如果是写/编辑类工具且执行成功，记录改动的文件路径."""
        if tool_name not in WRITE_TOOL_NAMES:
            return
        if result.is_error:
            return
        path = arguments.get("path") or arguments.get("file_path")
        if not path or not isinstance(path, str):
            return
        if path not in self._files_changed:
            self._files_changed.append(path)

    async def _force_final_response(
        self,
        accumulated_usage: TokenUsage,
        total_tool_calls: int,
        total_tool_errors: int = 0,
    ) -> AgentResult:
        """超出轮数限制时，不带 tools 做一次收尾调用."""
        self.conversation.append(
            Message.user(
                "你已经进行了很多轮工具调用。请根据目前获得的信息，"
                "直接给出最终回答，不要再调用工具。"
            )
        )
        response = await self.llm_client.chat(messages=self.messages, tools=None)
        accumulated_usage.input_tokens += response.usage.input_tokens
        accumulated_usage.output_tokens += response.usage.output_tokens
        self.conversation.append(Message.assistant(response.content))
        self._accumulate_usage(accumulated_usage)
        return AgentResult(
            content=response.content,
            usage=accumulated_usage,
            tool_calls_count=total_tool_calls,
            stop_reason="max_rounds",
            tool_calls_errors=total_tool_errors,
        )

    def _accumulate_usage(self, usage: TokenUsage) -> None:
        """累计本轮 token 用量到全局."""
        self.total_usage.input_tokens += usage.input_tokens
        self.total_usage.output_tokens += usage.output_tokens

    def reset(self) -> None:
        """重置对话历史（保留 system prompt）."""
        self.conversation.reset(self._build_full_system_prompt())
        if self.loop_guard:
            self.loop_guard.reset()
