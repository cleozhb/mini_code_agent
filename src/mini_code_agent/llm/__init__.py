"""LLM 客户端抽象层 — 统一多模型接入."""

from .base import (
    LLMClient,
    LLMError,
    LLMAuthError,
    LLMRateLimitError,
    LLMResponse,
    Message,
    Role,
    StreamDelta,
    StreamDeltaType,
    TokenUsage,
    ToolCall,
    ToolParam,
    ToolResult,
)
from .factory import create_client

__all__ = [
    "LLMClient",
    "LLMError",
    "LLMAuthError",
    "LLMRateLimitError",
    "LLMResponse",
    "Message",
    "Role",
    "StreamDelta",
    "StreamDeltaType",
    "TokenUsage",
    "ToolCall",
    "ToolParam",
    "ToolResult",
    "create_client",
]
