"""core 模块 — Agent 核心循环与 System Prompt 构建."""

from .agent import Agent, AgentError, AgentEvent, AgentEventType, AgentResult, ConfirmCallback
from .system_prompt import DEFAULT_SYSTEM_PROMPT, build_system_prompt, build_system_prompt_with_context

__all__ = [
    "Agent",
    "AgentError",
    "AgentEvent",
    "AgentEventType",
    "AgentResult",
    "ConfirmCallback",
    "DEFAULT_SYSTEM_PROMPT",
    "build_system_prompt",
    "build_system_prompt_with_context",
]
