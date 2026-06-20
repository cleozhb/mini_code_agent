"""LLM 客户端抽象基类与统一数据结构."""

from __future__ import annotations

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, AsyncIterator, Callable, TypeVar

T = TypeVar("T")

if TYPE_CHECKING:
    from ..trace import TraceRecorder


# ---------------------------------------------------------------------------
# 统一消息格式
# ---------------------------------------------------------------------------


class Role(str, Enum):
    """消息角色."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class ToolCall:
    """模型发起的工具调用."""

    id: str
    name: str
    arguments: dict  # 已解析的 JSON
    raw_arguments: str = ""
    parse_error: str | None = None

    def arguments_json(self) -> str:
        if self.raw_arguments and self.parse_error:
            return self.raw_arguments
        return json.dumps(self.arguments, ensure_ascii=False)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "arguments": self.arguments,
            "raw_arguments": self.raw_arguments,
            "parse_error": self.parse_error,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ToolCall:
        required = {"id", "name", "arguments"}
        if not isinstance(data, dict) or not required.issubset(data):
            raise ValueError("invalid checkpoint tool_call")
        arguments = data["arguments"]
        if not isinstance(arguments, dict):
            raise ValueError("invalid checkpoint tool_call.arguments")
        return cls(
            id=str(data["id"]),
            name=str(data["name"]),
            arguments=arguments,
            raw_arguments=str(data.get("raw_arguments") or ""),
            parse_error=data.get("parse_error"),
        )


@dataclass
class ToolResult:
    """工具执行结果，回传给模型."""

    tool_call_id: str
    content: str
    is_error: bool = False

    def to_dict(self) -> dict:
        content = self.content if isinstance(self.content, str) else str(self.content)
        return {
            "tool_call_id": self.tool_call_id,
            "content": content[:1500],
            "is_error": self.is_error,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ToolResult:
        required = {"tool_call_id", "content", "is_error"}
        if not isinstance(data, dict) or not required.issubset(data):
            raise ValueError("invalid checkpoint tool_result")
        return cls(
            tool_call_id=str(data["tool_call_id"]),
            content=str(data["content"]),
            is_error=bool(data["is_error"]),
        )


@dataclass
class Message:
    """统一消息结构，覆盖所有角色."""

    role: Role
    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_result: ToolResult | None = None

    # ---------- 便捷构造 ----------

    @staticmethod
    def system(content: str) -> Message:
        return Message(role=Role.SYSTEM, content=content)

    @staticmethod
    def user(content: str) -> Message:
        return Message(role=Role.USER, content=content)

    @staticmethod
    def assistant(content: str, tool_calls: list[ToolCall] | None = None) -> Message:
        return Message(
            role=Role.ASSISTANT,
            content=content,
            tool_calls=tool_calls or [],
        )

    @staticmethod
    def tool(tool_result: ToolResult) -> Message:
        return Message(role=Role.TOOL, tool_result=tool_result)

    def to_dict(self) -> dict:
        content = self.content if isinstance(self.content, str) else str(self.content or "")
        return {
            "role": self.role.value,
            "content": content[:1500],
            "tool_calls": [tool_call.to_dict() for tool_call in self.tool_calls],
            "tool_result": self.tool_result.to_dict() if self.tool_result else None,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Message:
        required = {"role", "content", "tool_calls", "tool_result"}
        if not isinstance(data, dict) or not required.issubset(data):
            raise ValueError("invalid checkpoint message")
        try:
            role = Role(data["role"])
        except ValueError as e:
            raise ValueError("invalid checkpoint message role") from e
        raw_tool_calls = data["tool_calls"]
        if not isinstance(raw_tool_calls, list):
            raise ValueError("invalid checkpoint message tool_calls")
        raw_tool_result = data["tool_result"]
        return cls(
            role=role,
            content=str(data["content"]),
            tool_calls=[ToolCall.from_dict(item) for item in raw_tool_calls],
            tool_result=(
                ToolResult.from_dict(raw_tool_result)
                if raw_tool_result is not None
                else None
            ),
        )


# ---------------------------------------------------------------------------
# 工具定义（传给 API 的 schema）
# ---------------------------------------------------------------------------


@dataclass
class ToolParam:
    """一个工具的定义，传递给 LLM API."""

    name: str
    description: str
    parameters: dict  # JSON Schema
    strict: bool = False


# ---------------------------------------------------------------------------
# 响应数据结构
# ---------------------------------------------------------------------------


@dataclass
class TokenUsage:
    """Token 计数."""

    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    reasoning_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return (
            self.input_tokens
            + self.output_tokens
            + self.cache_creation_input_tokens
            + self.cache_read_input_tokens
        )

    def add(self, other: TokenUsage) -> None:
        """把另一份用量累加到当前对象."""
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cached_input_tokens += other.cached_input_tokens
        self.reasoning_tokens += other.reasoning_tokens
        self.cache_creation_input_tokens += other.cache_creation_input_tokens
        self.cache_read_input_tokens += other.cache_read_input_tokens


@dataclass(frozen=True)
class LLMCapabilities:
    """LLM backend 能力声明."""

    structured_outputs: bool = False
    strict_tools: bool = False
    streaming_tools: bool = False
    prompt_cache: bool = False
    reasoning_usage: bool = False


@dataclass
class LLMResponse:
    """完整的 LLM 响应."""

    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: TokenUsage = field(default_factory=TokenUsage)
    raw: object = None  # 保留原始响应以便调试


class StreamDeltaType(str, Enum):
    TEXT = "text"
    TOOL_CALL_START = "tool_call_start"
    TOOL_CALL_DELTA = "tool_call_delta"
    TOOL_CALL_END = "tool_call_end"
    FINISH = "finish"


@dataclass
class StreamDelta:
    """流式输出的单个增量."""

    type: StreamDeltaType
    content: str = ""
    # 工具调用相关
    tool_call_id: str = ""
    tool_name: str = ""
    # 结束时携带用量
    usage: TokenUsage | None = None


# ---------------------------------------------------------------------------
# 自定义异常
# ---------------------------------------------------------------------------


class LLMError(Exception):
    """LLM 调用相关错误的基类."""


class LLMAuthError(LLMError):
    """API Key 无效或缺失."""


class LLMRateLimitError(LLMError):
    """触发速率限制."""


# ---------------------------------------------------------------------------
# 抽象基类
# ---------------------------------------------------------------------------


class LLMClient(ABC):
    """LLM 客户端抽象基类.

    子类负责将统一格式转换为各自提供商的 API 格式。
    """

    def __init__(self, model: str, api_key: str | None = None, base_url: str | None = None) -> None:
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.total_usage = TokenUsage()
        self.capabilities = LLMCapabilities()
        self.trace_recorder: TraceRecorder | None = None
        self.last_llm_call_id: str | None = None

    def _accumulate_usage(self, usage: TokenUsage) -> None:
        """累计 token 用量."""
        self.total_usage.add(usage)

    async def _call_with_retry(self, coro_factory: Callable[[], T]) -> T:
        """对 LLMRateLimitError 做指数退避重试."""
        _logger = logging.getLogger(__name__)
        delay = 5.0
        for attempt in range(5):
            try:
                return await coro_factory()
            except LLMRateLimitError:
                if attempt == 4:
                    raise
                _logger.warning(
                    "Rate limited, retrying in %.1fs (attempt %d/5)",
                    delay, attempt + 1,
                )
                await asyncio.sleep(delay)
                delay = min(delay * 2, 60.0)
        raise RuntimeError("unreachable")

    def set_trace_recorder(self, recorder: TraceRecorder | None) -> None:
        """挂载 trace recorder."""
        self.trace_recorder = recorder

    def _trace_start_llm_call(
        self,
        *,
        mode: str,
        messages: list[Message],
        tools: list[ToolParam] | None,
        response_format: dict | None,
    ) -> str | None:
        """记录一次 LLM 调用开始."""
        if self.trace_recorder is None:
            return None
        call_id = self.trace_recorder.start_llm_call(
            model=self.model,
            backend=type(self).__name__,
            mode=mode,
            messages=messages,
            tools=tools,
            response_format=response_format,
        )
        self.last_llm_call_id = call_id
        return call_id

    def _trace_provider_request(self, call_id: str | None, payload: object) -> None:
        if self.trace_recorder is not None:
            self.trace_recorder.record_llm_provider_request(call_id, payload)

    def _trace_raw_response(
        self,
        call_id: str | None,
        response: object,
        *,
        label: str = "response.raw",
    ) -> None:
        if self.trace_recorder is not None:
            self.trace_recorder.record_llm_raw_response(call_id, response, label=label)

    def _trace_stream_event(self, call_id: str | None, event: object) -> None:
        if self.trace_recorder is not None:
            self.trace_recorder.record_llm_stream_event(call_id, event)

    def _trace_parsed_response(
        self,
        call_id: str | None,
        response: LLMResponse,
    ) -> None:
        if self.trace_recorder is not None:
            self.trace_recorder.record_llm_parsed_response(call_id, response)

    def _trace_finish_llm_call(
        self,
        call_id: str | None,
        *,
        response: LLMResponse | None = None,
        error: BaseException | None = None,
    ) -> None:
        if self.trace_recorder is not None:
            self.trace_recorder.finish_llm_call(
                call_id,
                response=response,
                error=error,
            )

    @abstractmethod
    async def chat(
        self,
        messages: list[Message],
        tools: list[ToolParam] | None = None,
        response_format: dict | None = None,  # 新增：支持结构化输出
    ) -> LLMResponse:
        """发送消息并获取完整响应.

        Args:
            messages: 消息列表
            tools: 可选的工具定义
            response_format: 可选的响应格式，用于结构化输出
                例如: {"type": "json_object"} 或 {"type": "json_schema", "json_schema": {...}}
        """
        ...

    @abstractmethod
    def chat_stream(
        self,
        messages: list[Message],
        tools: list[ToolParam] | None = None,
        response_format: dict | None = None,  # 新增：支持结构化输出
    ) -> AsyncIterator[StreamDelta]:
        """发送消息并获取流式响应.

        Args:
            messages: 消息列表
            tools: 可选的工具定义
            response_format: 可选的响应格式，用于结构化输出
        """
        ...
