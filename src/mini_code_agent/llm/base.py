"""LLM 客户端抽象基类与统一数据结构."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import AsyncIterator


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


@dataclass
class ToolResult:
    """工具执行结果，回传给模型."""

    tool_call_id: str
    content: str
    is_error: bool = False


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

    def _accumulate_usage(self, usage: TokenUsage) -> None:
        """累计 token 用量."""
        self.total_usage.add(usage)

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
