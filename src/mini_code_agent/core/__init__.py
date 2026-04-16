"""core 模块 — Agent 核心循环、System Prompt、任务验证与重试控制."""

from .agent import Agent, AgentError, AgentEvent, AgentEventType, AgentResult, ConfirmCallback
from .retry import AttemptRecord, RetryController
from .system_prompt import DEFAULT_SYSTEM_PROMPT, build_system_prompt, build_system_prompt_with_context
from .verifier import VerificationResult, Verifier

__all__ = [
    "Agent",
    "AgentError",
    "AgentEvent",
    "AgentEventType",
    "AgentResult",
    "AttemptRecord",
    "ConfirmCallback",
    "DEFAULT_SYSTEM_PROMPT",
    "RetryController",
    "VerificationResult",
    "Verifier",
    "build_system_prompt",
    "build_system_prompt_with_context",
]
