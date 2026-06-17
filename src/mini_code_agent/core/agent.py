"""Agent 核心循环 — 整个系统的心脏."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, AsyncIterator, Callable, Awaitable, Literal

from pydantic import ValidationError

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
from ..safety.git_checkpoint import GitCheckpoint
from ..safety.loop_guard import LoopGuard
from ..tools.base import PermissionLevel, Tool, ToolRegistry
from ..tools.base import ToolResult as ExecToolResult
from .retry import RetryController
from .verifier import VerificationResult, Verifier

if TYPE_CHECKING:
    from ..longrun.checkpoint_manager import CheckpointManager
    from ..longrun.config import LongRunConfig
    from ..longrun.ledger_manager import TaskLedgerManager
    from ..longrun.session_state import SessionState
    from ..longrun.task_ledger import TaskLedger
    from ..trace import TraceRecorder
    from ..verify.verifier import IncrementalVerifier
    from .task_graph import TaskGraph

logger = logging.getLogger(__name__)

MAX_TOOL_ROUNDS = 25

# 会改动文件的工具名；用于追踪本轮任务修改过哪些文件
WRITE_TOOL_NAMES = {"WriteFile", "EditFile", "write_file", "edit_file"}


def _find_shared_lsp_manager(tool_registry: ToolRegistry) -> Any | None:
    for tool in tool_registry.list_tools():
        manager = getattr(tool, "_lsp_manager", None)
        if manager is not None:
            return manager
    return None


class AgentError(Exception):
    """Agent 运行时错误."""


class AgentStuckError(Exception):
    """Agent 主动宣告卡住，需要上层介入（例如 STUCK confidence）."""

    def __init__(self, reason: str, question: str = "") -> None:
        super().__init__(reason)
        self.reason = reason
        self.question = question


class AgentObserver:
    """观察者接口：Agent 在 tool / LLM 调用时通知 observer.

    SubtaskRunner 的 ArtifactObserver 是主要使用方；
    没人挂 observer 时 Agent 行为不变。
    """

    def on_tool_call(self, name: str, args: dict[str, Any], result: Any) -> None:
        ...

    def on_llm_call(self, tokens_in: int, tokens_out: int, model: str) -> None:
        ...


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
        git_checkpoint: GitCheckpoint | None = None,
        max_wall_time_seconds: float | None = None,
        ledger: TaskLedger | None = None,
        ledger_manager: TaskLedgerManager | None = None,
        checkpoint_manager: CheckpointManager | None = None,
        longrun_config: LongRunConfig | None = None,
        task_graph: TaskGraph | None = None,
        incremental_verifier: IncrementalVerifier | None = None,
        trace_recorder: TraceRecorder | None = None,
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
        self.git_checkpoint = git_checkpoint
        self.max_wall_time_seconds = max_wall_time_seconds

        # Ledger（外部记忆）
        self.ledger = ledger
        self.ledger_manager = ledger_manager

        # Checkpoint / Resume 系统
        self.checkpoint_manager = checkpoint_manager
        self.longrun_config = longrun_config
        self.task_graph = task_graph
        self.last_checkpoint: SessionState | None = None
        self.current_task_id: str | None = None

        # 当前任务期间追踪改动过哪些文件（供 Verifier 使用）
        self._files_changed: list[str] = []

        # Incremental Verifier（可选）—— 每次 Edit 后做层级 1 快速验证
        self.incremental_verifier = incremental_verifier
        self.lsp_manager = _find_shared_lsp_manager(tool_registry)
        self.trace_recorder = trace_recorder or getattr(
            llm_client, "trace_recorder", None
        )

        # Observer 列表（可选）— SubtaskRunner 用来追踪 tool / LLM 调用
        self.observers: list[AgentObserver] = []

        # 使用 ConversationManager 管理消息
        self.conversation = ConversationManager(llm_client=llm_client)
        self.conversation.init_system(self._build_full_system_prompt())

    def _build_full_system_prompt(self) -> str:
        """拼接 system prompt + 项目记忆 + Ledger 上下文."""
        prompt = self.system_prompt
        if self.project_memory:
            memory_text = self.project_memory.format_for_prompt()
            if memory_text:
                prompt += f"\n\n<project-memory>\n{memory_text}\n</project-memory>"
        if self.ledger and self.ledger_manager:
            ledger_context = self.ledger_manager.build_context_summary(self.ledger)
            if ledger_context:
                prompt += f"\n\n<task-ledger>\n{ledger_context}\n</task-ledger>"
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
        trace_run_id = self._trace_agent_run_start(user_message)
        if self.max_wall_time_seconds is None:
            try:
                result = await self._run_impl(user_message)
            except Exception as e:
                self._trace_agent_run_finish(trace_run_id, error=e)
                raise
            self._trace_agent_run_finish(trace_run_id, result=result)
            return result

        try:
            result = await asyncio.wait_for(
                self._run_impl(user_message),
                timeout=self.max_wall_time_seconds,
            )
            self._trace_agent_run_finish(trace_run_id, result=result)
            return result
        except asyncio.TimeoutError:
            logger.warning(
                "Agent 运行超时（%.1fs），返回部分结果",
                self.max_wall_time_seconds,
            )
            result = AgentResult(
                content=(
                    f"[超时] Agent 超过 {self.max_wall_time_seconds:.0f}s "
                    f"墙钟上限，已中止。"
                ),
                usage=TokenUsage(),
                tool_calls_count=0,
                stop_reason="timeout",
            )
            self._trace_agent_run_finish(
                trace_run_id,
                result=result,
                stop_reason="timeout",
            )
            return result

    async def _run_impl(self, user_message: str) -> AgentResult:
        """核心循环：发送消息 → 处理工具调用 → 返回最终文本.

        如果配置了 Verifier，纯文本回复之后会自动对改动的文件跑一次验证，
        失败时在 RetryController 允许的范围内自动把错误回传给 LLM 继续修复。
        """
        # 每次顶层 run() 开始时重置文件追踪与重试计数
        self._files_changed = []
        if self.retry_controller is not None:
            self.retry_controller.reset()

        # 自动 checkpoint：记录任务开始前的 HEAD（不创建 commit）
        task_desc = user_message[:80]
        if self.git_checkpoint is not None:
            await self.git_checkpoint.save_head()

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
            total_usage.add(single.usage)
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

        # 自动 checkpoint：任务完成后（create_checkpoint 内部会检测无改动则跳过）
        if self.git_checkpoint is not None:
            await self.git_checkpoint.create_checkpoint(f"after: {task_desc}")

        # Ledger 资源更新 — 只记 +1 步；
        # token 由 SubtaskRunner 经 Artifact.resource_usage 写入，
        # 在这里再加一次会导致双重记账。
        if self.ledger and self.ledger_manager:
            self.ledger_manager.update_resources(
                self.ledger, 0, 1, 0.0,
            )

        # 自动 checkpoint 检查
        await self._maybe_auto_checkpoint()

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

    async def _run_once(self, user_message: str) -> AgentResult:
        """单轮对话：发送消息 → 处理工具调用 → 返回最终文本.

        流程：
        1. 把 user_message 加入 messages
        2. 调用 LLM（带 tools schema）
        3. 如果 LLM 返回 tool_use：逐个执行并把结果回传，回到步骤 2
        4. 如果 LLM 返回纯文本：返回给用户
        5. 最多 MAX_TOOL_ROUNDS 轮 tool calling
        """
        # Ledger 上下文刷新：每轮替换 system prompt（防止累积）
        if self.ledger and self.ledger_manager:
            self.conversation.update_system(self._build_full_system_prompt())

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
            round_usage.add(response.usage)
            self._notify_llm_call(response.usage)

            # LoopGuard token 预算检查
            if self.loop_guard:
                token_msg = self.loop_guard.add_tokens(
                    response.usage.input_tokens + response.usage.output_tokens
                )
                if token_msg:
                    logger.warning(token_msg)
                    # 软警告（80%）：注入 conversation 让 LLM 感知到预算压力
                    if self.loop_guard.total_tokens < self.loop_guard.max_tokens:
                        self.conversation.append(Message.user(token_msg))
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
        trace_run_id = self._trace_agent_run_start(user_message)
        # 自动 checkpoint：记录任务开始前的 HEAD（不创建 commit）
        task_desc = user_message[:80]
        self._files_changed = []
        if self.git_checkpoint is not None:
            await self.git_checkpoint.save_head()

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
                        tc.raw_arguments = delta.content
                        try:
                            parsed_args = json.loads(delta.content) if delta.content else {}
                        except json.JSONDecodeError:
                            tc.arguments = {}
                            tc.parse_error = "工具参数不是合法 JSON"
                        else:
                            if isinstance(parsed_args, dict):
                                tc.arguments = parsed_args
                                tc.parse_error = None
                            else:
                                tc.arguments = {}
                                tc.parse_error = (
                                    "工具参数必须是 JSON object，"
                                    f"实际是 {type(parsed_args).__name__}"
                                )
                        tool_calls.append(tc)
                        yield AgentEvent(
                            type=AgentEventType.TOOL_CALL_END,
                            tool_call=tc,
                        )

                elif delta.type == StreamDeltaType.FINISH:
                    if delta.usage:
                        round_usage.add(delta.usage)
                        self._notify_llm_call(delta.usage)
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
                # 自动 checkpoint：任务完成后
                if self.git_checkpoint is not None:
                    await self.git_checkpoint.create_checkpoint(f"after: {task_desc}")
                self._trace_agent_run_finish(
                    trace_run_id,
                    result=AgentResult(
                        content=full_content,
                        usage=round_usage,
                        tool_calls_count=total_tool_calls,
                    ),
                )
                yield AgentEvent(
                    type=AgentEventType.FINISH,
                    content="ok",
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
                round_usage.add(delta.usage)
                self._notify_llm_call(delta.usage)

        self.conversation.append(Message.assistant(full_content))
        self._accumulate_usage(round_usage)
        await self._maybe_compress()
        # 自动 checkpoint：超轮数收尾后
        if self.git_checkpoint is not None:
            await self.git_checkpoint.create_checkpoint(f"after: {task_desc}")
        self._trace_agent_run_finish(
            trace_run_id,
            result=AgentResult(
                content=full_content,
                usage=round_usage,
                tool_calls_count=total_tool_calls,
                stop_reason="max_rounds",
            ),
        )
        yield AgentEvent(
            type=AgentEventType.FINISH,
            content="max_rounds",
            usage=round_usage,
        )

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

    def _validate_tool_call_arguments(
        self,
        tool: Tool,
        tool_call: ToolCall,
    ) -> ExecToolResult | None:
        """先校验工具参数，再进入安全检查 / 审批 / 执行.

        安全检查会读取 Bash command、文件 path 等字段；如果模型给了
        bool/list 之类的坏类型，必须先作为 tool error 返给模型自修正，
        不能让安全层或工具实现直接抛 Python 异常。
        """
        if not isinstance(tool_call.arguments, dict):
            return ExecToolResult(
                output="",
                error=(
                    "参数校验失败:\n"
                    "工具参数必须是 JSON object，"
                    f"实际是 {type(tool_call.arguments).__name__}: "
                    f"{tool_call.arguments!r}"
                ),
            )

        if tool.InputModel is None:
            return None

        try:
            validated = tool.InputModel.model_validate(tool_call.arguments)
        except ValidationError as e:
            return ExecToolResult(output="", error=f"参数校验失败:\n{e}")

        tool_call.arguments = validated.model_dump()
        return None

    async def _execute_tool_call(self, tool_call: ToolCall) -> Message:
        """执行单个工具调用，返回 tool result 消息."""
        trace_llm_call_id = self._trace_tool_call_start(tool_call)
        if tool_call.parse_error:
            content = (
                "错误：工具参数 JSON 解析失败，请重新调用工具并传入合法参数。"
                f"\nparse_error: {tool_call.parse_error}"
                f"\nraw_arguments: {tool_call.raw_arguments}"
            )
            self._trace_tool_call_result(
                tool_call,
                ExecToolResult(output="", error=content),
                llm_call_id=trace_llm_call_id,
                metadata={"parse_error": True},
            )
            return Message.tool(
                LLMToolResult(
                    tool_call_id=tool_call.id,
                    content=content,
                    is_error=True,
                )
            )

        tool = self.tool_registry.get(tool_call.name)

        if tool is None:
            result = ExecToolResult(output="", error=f"未找到工具 '{tool_call.name}'")
            self._trace_tool_call_result(
                tool_call,
                result,
                llm_call_id=trace_llm_call_id,
                metadata={"tool_missing": True},
            )
            return Message.tool(
                LLMToolResult(
                    tool_call_id=tool_call.id,
                    content=f"错误：未找到工具 '{tool_call.name}'",
                    is_error=True,
                )
            )

        validation_error = self._validate_tool_call_arguments(tool, tool_call)
        if validation_error is not None:
            self._trace_tool_call_result(
                tool_call,
                validation_error,
                llm_call_id=trace_llm_call_id,
                metadata={"validation_error": True},
            )
            return Message.tool(
                LLMToolResult(
                    tool_call_id=tool_call.id,
                    content=validation_error.error or "参数校验失败",
                    is_error=True,
                )
            )

        # 安全检查
        safety_level, block_reason = self._check_safety(tool.name, tool_call)
        if safety_level == SafetyLevel.BLOCKED:
            result = ExecToolResult(output="", error=f"[安全拦截] {block_reason}")
            self._trace_tool_call_result(
                tool_call,
                result,
                llm_call_id=trace_llm_call_id,
                metadata={
                    "safety_level": safety_level,
                    "block_reason": block_reason,
                },
            )
            return Message.tool(
                LLMToolResult(
                    tool_call_id=tool_call.id,
                    content=f"[安全拦截] {block_reason}",
                    is_error=True,
                )
            )

        # 检查权限
        if tool.permission_level == PermissionLevel.DENY:
            result = ExecToolResult(
                output="",
                error=f"工具 '{tool_call.name}' 被禁止执行",
            )
            self._trace_tool_call_result(
                tool_call,
                result,
                llm_call_id=trace_llm_call_id,
                metadata={"permission_level": tool.permission_level},
            )
            return Message.tool(
                LLMToolResult(
                    tool_call_id=tool_call.id,
                    content=f"错误：工具 '{tool_call.name}' 被禁止执行",
                    is_error=True,
                )
            )

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
                result = ExecToolResult(output="", error="用户拒绝了此操作")
                self._trace_tool_call_result(
                    tool_call,
                    result,
                    llm_call_id=trace_llm_call_id,
                    metadata={
                        "safety_level": safety_level,
                        "user_approved": False,
                    },
                )
                return Message.tool(
                    LLMToolResult(
                        tool_call_id=tool_call.id,
                        content="用户拒绝了此操作",
                        is_error=True,
                    )
                )
            if edited_args is not None:
                tool_call.arguments = edited_args
        elif needs_confirm:
            logger.info(
                "工具 '%s' 需要确认但没有 confirm_callback，当前自动放行: %s",
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
        except AgentStuckError as e:
            self._trace_tool_call_result(
                tool_call,
                ExecToolResult(output="", error=str(e)),
                llm_call_id=trace_llm_call_id,
                metadata={"agent_stuck": True},
            )
            # Agent 主动宣告卡住 — 让上层捕获，不要包成 tool error
            raise
        except Exception as e:
            logger.exception("工具 '%s' 执行异常", tool_call.name)
            result = ExecToolResult(output="", error=f"{type(e).__name__}: {e}")
            self._trace_tool_call_result(
                tool_call,
                result,
                llm_call_id=trace_llm_call_id,
                metadata={"exception": True},
            )
            return Message.tool(
                LLMToolResult(
                    tool_call_id=tool_call.id,
                    content=f"工具执行异常: {type(e).__name__}: {e}",
                    is_error=True,
                )
            )

        # 追踪文件改动
        self._track_file_change(tool.name, tool_call.arguments, result)
        self._notify_tool_call(tool.name, tool_call.arguments, result)

        # 将 tools.base.ToolResult 转为 llm.base.ToolResult
        if result.is_error:
            content = result.error or "未知错误"
            is_error = True
        else:
            content = result.output
            is_error = False

        # Incremental Verifier — 层级 1 快速反馈
        warning = await self._maybe_quick_verify(tool.name, tool_call.arguments, result)
        if warning:
            content = (content or "") + warning
            result = ExecToolResult(
                output=(result.output or "") + warning,
                error=result.error,
                exit_code=result.exit_code,
            )

        self._trace_tool_call_result(
            tool_call,
            result,
            llm_call_id=trace_llm_call_id,
            metadata={"safety_level": safety_level},
        )

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
        trace_llm_call_id = self._trace_tool_call_start(tool_call)
        if tool_call.parse_error:
            content = (
                "错误：工具参数 JSON 解析失败，请重新调用工具并传入合法参数。"
                f"\nparse_error: {tool_call.parse_error}"
                f"\nraw_arguments: {tool_call.raw_arguments}"
            )
            dummy = ExecToolResult(output="", error=content)
            self._trace_tool_call_result(
                tool_call,
                dummy,
                llm_call_id=trace_llm_call_id,
                metadata={"parse_error": True},
            )
            msg = Message.tool(
                LLMToolResult(
                    tool_call_id=tool_call.id,
                    content=content,
                    is_error=True,
                )
            )
            return msg, dummy

        tool = self.tool_registry.get(tool_call.name)

        if tool is None:
            dummy = ExecToolResult(output="", error=f"未找到工具 '{tool_call.name}'")
            self._trace_tool_call_result(
                tool_call,
                dummy,
                llm_call_id=trace_llm_call_id,
                metadata={"tool_missing": True},
            )
            msg = Message.tool(
                LLMToolResult(
                    tool_call_id=tool_call.id,
                    content=f"错误：未找到工具 '{tool_call.name}'",
                    is_error=True,
                )
            )
            return msg, dummy

        validation_error = self._validate_tool_call_arguments(tool, tool_call)
        if validation_error is not None:
            self._trace_tool_call_result(
                tool_call,
                validation_error,
                llm_call_id=trace_llm_call_id,
                metadata={"validation_error": True},
            )
            msg = Message.tool(
                LLMToolResult(
                    tool_call_id=tool_call.id,
                    content=validation_error.error or "参数校验失败",
                    is_error=True,
                )
            )
            return msg, validation_error

        # 安全检查
        safety_level, block_reason = self._check_safety(tool.name, tool_call)
        if safety_level == SafetyLevel.BLOCKED:
            dummy = ExecToolResult(output="", error=block_reason or "危险操作被拦截")
            self._trace_tool_call_result(
                tool_call,
                dummy,
                llm_call_id=trace_llm_call_id,
                metadata={
                    "safety_level": safety_level,
                    "block_reason": block_reason,
                },
            )
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
            self._trace_tool_call_result(
                tool_call,
                dummy,
                llm_call_id=trace_llm_call_id,
                metadata={"permission_level": tool.permission_level},
            )
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
                self._trace_tool_call_result(
                    tool_call,
                    dummy,
                    llm_call_id=trace_llm_call_id,
                    metadata={
                        "safety_level": safety_level,
                        "user_approved": False,
                    },
                )
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
        except AgentStuckError as e:
            self._trace_tool_call_result(
                tool_call,
                ExecToolResult(output="", error=str(e)),
                llm_call_id=trace_llm_call_id,
                metadata={"agent_stuck": True},
            )
            raise
        except Exception as e:
            logger.exception("工具 '%s' 执行异常", tool_call.name)
            err_result = ExecToolResult(
                output="", error=f"{type(e).__name__}: {e}"
            )
            self._trace_tool_call_result(
                tool_call,
                err_result,
                llm_call_id=trace_llm_call_id,
                metadata={"exception": True},
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
        self._notify_tool_call(tool.name, tool_call.arguments, result)

        # 转换
        if result.is_error:
            content = result.error or "未知错误"
            is_error = True
        else:
            content = result.output
            is_error = False

        # Incremental Verifier — 层级 1 快速反馈
        warning = await self._maybe_quick_verify(tool.name, tool_call.arguments, result)
        if warning:
            content = (content or "") + warning
            result.output = (result.output or "") + warning

        self._trace_tool_call_result(
            tool_call,
            result,
            llm_call_id=trace_llm_call_id,
            metadata={"safety_level": safety_level},
        )

        msg = Message.tool(
            LLMToolResult(
                tool_call_id=tool_call.id,
                content=content,
                is_error=is_error,
            )
        )
        return msg, result

    async def _maybe_quick_verify(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        result: ExecToolResult,
    ) -> str:
        """如果配置了 incremental_verifier 且这是写文件类工具，跑一次层级 1 快检.

        Returns:
            一段附加到 tool 输出末尾的警告字符串（无问题或未配置时返回 ""）
        """
        if self.incremental_verifier is None or self.project_path is None:
            return ""
        if tool_name not in WRITE_TOOL_NAMES:
            return ""
        if result.is_error:
            return ""
        path = arguments.get("path") or arguments.get("file_path")
        if not path or not isinstance(path, str):
            return ""

        try:
            quick = await self.incremental_verifier.verify_after_edit(
                files_changed=[path],
                project_path=self.project_path,
            )
        except Exception as e:  # noqa: BLE001 — 验证失败不能挂掉 Agent
            logger.debug("incremental quick verify error: %s", e)
            return ""

        if quick.overall_passed:
            return ""

        return f"\n\n[VERIFICATION WARNING]\n{quick.summary()}"

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
        accumulated_usage.add(response.usage)
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
        self.total_usage.add(usage)

    def _current_llm_call_id(self) -> str | None:
        """最近一次 LLM 调用 id，用于关联 tool 事件."""
        return getattr(self.llm_client, "last_llm_call_id", None)

    def _trace_agent_run_start(self, user_message: str) -> str | None:
        if self.trace_recorder is None:
            return None
        try:
            return self.trace_recorder.start_agent_run(user_message)
        except Exception as e:  # noqa: BLE001
            logger.debug("trace agent_run_start failed: %s", e)
            return None

    def _trace_agent_run_finish(
        self,
        run_id: str | None,
        *,
        result: AgentResult | None = None,
        error: BaseException | None = None,
        stop_reason: str | None = None,
    ) -> None:
        if self.trace_recorder is None:
            return
        try:
            self.trace_recorder.finish_agent_run(
                run_id,
                result=result,
                error=error,
                stop_reason=stop_reason,
            )
        except Exception as e:  # noqa: BLE001
            logger.debug("trace agent_run_finish failed: %s", e)

    def _trace_tool_call_start(self, tool_call: ToolCall) -> str | None:
        llm_call_id = self._current_llm_call_id()
        if self.trace_recorder is None:
            return llm_call_id
        try:
            self.trace_recorder.record_tool_call_start(
                llm_call_id=llm_call_id,
                tool_call=tool_call,
            )
        except Exception as e:  # noqa: BLE001
            logger.debug("trace tool_call_start failed: %s", e)
        return llm_call_id

    def _trace_tool_call_result(
        self,
        tool_call: ToolCall,
        result: ExecToolResult,
        *,
        llm_call_id: str | None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if self.trace_recorder is None:
            return
        try:
            self.trace_recorder.record_tool_call_result(
                llm_call_id=llm_call_id,
                tool_call=tool_call,
                result=result,
                metadata=metadata,
            )
        except Exception as e:  # noqa: BLE001
            logger.debug("trace tool_call_result failed: %s", e)

    def add_observer(self, obs: AgentObserver) -> None:
        """挂载一个 observer."""
        self.observers.append(obs)

    def remove_observer(self, obs: AgentObserver | None = None) -> None:
        """移除 observer：传入具体对象只移除该对象，否则清空所有."""
        if obs is None:
            self.observers.clear()
        else:
            try:
                self.observers.remove(obs)
            except ValueError:
                pass

    def _notify_tool_call(self, name: str, args: dict[str, Any], result: Any) -> None:
        for obs in self.observers:
            try:
                obs.on_tool_call(name, args, result)
            except Exception as e:  # noqa: BLE001 — observer 异常不能挂掉 Agent
                logger.debug("observer.on_tool_call error: %s", e)

    def _notify_llm_call(self, usage: TokenUsage) -> None:
        if not self.observers:
            return
        model = getattr(self.llm_client, "model", "")
        for obs in self.observers:
            try:
                obs.on_llm_call(usage.input_tokens, usage.output_tokens, model)
            except Exception as e:  # noqa: BLE001
                logger.debug("observer.on_llm_call error: %s", e)

    def reset(self) -> None:
        """重置对话历史（保留 system prompt）."""
        self.conversation.reset(self._build_full_system_prompt())
        if self.loop_guard:
            self.loop_guard.reset()

    def inject_initial_message(self, text: str) -> None:
        """把一条 user message 注入 messages 列表.

        这条消息不会触发 LLM 调用，而是在下一次 run() 时作为初始上下文。
        """
        self.conversation.append(Message.user(text))

    async def _maybe_auto_checkpoint(self) -> None:
        """在 Agent 主循环每轮结束后检查是否需要自动 checkpoint."""
        if (
            self.checkpoint_manager is None
            or self.ledger is None
            or self.longrun_config is None
        ):
            return

        trigger = self.checkpoint_manager.auto_checkpoint_policy(
            self.ledger, self.last_checkpoint, self.longrun_config,
        )
        if trigger is not None:
            try:
                # 将当前 token 数记入 config snapshot 以便下次比较
                config = self.longrun_config
                config_dict = config.to_dict()
                config_dict["_tokens_at_checkpoint"] = self.ledger.total_tokens_used

                # 临时用一个带 _tokens_at_checkpoint 的 config
                from ..longrun.config import LongRunConfig
                snapshot_config = LongRunConfig.from_dict(config_dict)
                snapshot_config_dict = snapshot_config.to_dict()
                snapshot_config_dict["_tokens_at_checkpoint"] = self.ledger.total_tokens_used

                # 需要传 messages 作为 list[dict]
                msg_dicts: list[dict] = []
                for m in self.messages:
                    entry: dict = {"role": m.role.value if hasattr(m.role, "value") else str(m.role)}
                    if isinstance(m.content, str):
                        entry["content"] = m.content[:500]
                    else:
                        entry["content"] = str(m.content)[:500]
                    msg_dicts.append(entry)

                self.last_checkpoint = await self.checkpoint_manager.save_checkpoint(
                    ledger=self.ledger,
                    trigger=trigger,
                    config=self.longrun_config,
                    current_task_id=self.current_task_id,
                    recent_messages=msg_dicts,
                )
                logger.info("自动 checkpoint 已创建: trigger=%s", trigger.value)
            except Exception as e:
                logger.warning("自动 checkpoint 失败: %s", e)
