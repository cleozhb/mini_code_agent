"""core 模块 — Agent 核心循环、System Prompt、任务验证与重试控制."""

from .agent import (
    Agent,
    AgentError,
    AgentEvent,
    AgentEventType,
    AgentObserver,
    AgentResult,
    AgentStuckError,
    ConfirmCallback,
    PlanConfirmCallback,
    PlanProgressCallback,
    PlanReplanCallback,
)
from .subtask_runner import GraphContext, SubtaskRunner
from .graph_executor import GraphExecutor, GraphResult, run_verification
from .graph_planner import GRAPH_PLANNER_SYSTEM_PROMPT, GraphPlanner, GraphPlannerError
from .planner import (
    PLANNER_SYSTEM_PROMPT,
    Plan,
    Planner,
    PlannerError,
    PlanStep,
)
from .retry import AttemptRecord, RetryController
from .system_prompt import DEFAULT_SYSTEM_PROMPT, build_system_prompt, build_system_prompt_with_context
from .task_graph import CyclicDependencyError, TaskGraph, TaskNode, TaskStatus
from .verifier import VerificationResult, Verifier

__all__ = [
    "Agent",
    "AgentError",
    "AgentEvent",
    "AgentEventType",
    "AgentObserver",
    "AgentResult",
    "AgentStuckError",
    "AttemptRecord",
    "GraphContext",
    "SubtaskRunner",
    "ConfirmCallback",
    "CyclicDependencyError",
    "DEFAULT_SYSTEM_PROMPT",
    "GRAPH_PLANNER_SYSTEM_PROMPT",
    "GraphExecutor",
    "GraphPlanner",
    "GraphPlannerError",
    "GraphResult",
    "PLANNER_SYSTEM_PROMPT",
    "Plan",
    "PlanConfirmCallback",
    "PlanProgressCallback",
    "PlanReplanCallback",
    "PlanStep",
    "Planner",
    "PlannerError",
    "RetryController",
    "TaskGraph",
    "TaskNode",
    "TaskStatus",
    "VerificationResult",
    "Verifier",
    "build_system_prompt",
    "build_system_prompt_with_context",
    "run_verification",
]
