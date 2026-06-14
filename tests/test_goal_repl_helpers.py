"""Goal mode REPL helper tests."""

from __future__ import annotations

from mini_code_agent.cli.repl import _changed_path_from_subagent_event
from mini_code_agent.core.agent import AgentEvent, AgentEventType
from mini_code_agent.llm.base import ToolCall
from mini_code_agent.tools.base import ToolResult


def test_changed_path_from_successful_write_event() -> None:
    event = AgentEvent(
        type=AgentEventType.TOOL_RESULT,
        tool_call=ToolCall(
            id="call-1",
            name="WriteFile",
            arguments={"path": "expr_goal_case/expr_eval.py"},
        ),
        tool_result=ToolResult(output="已写入"),
    )

    assert _changed_path_from_subagent_event(event) == "expr_goal_case/expr_eval.py"


def test_changed_path_ignores_non_write_or_failed_events() -> None:
    read_event = AgentEvent(
        type=AgentEventType.TOOL_RESULT,
        tool_call=ToolCall(
            id="call-1",
            name="ReadFile",
            arguments={"path": "expr_goal_case/expr_eval.py"},
        ),
        tool_result=ToolResult(output="content"),
    )
    failed_write = AgentEvent(
        type=AgentEventType.TOOL_RESULT,
        tool_call=ToolCall(
            id="call-2",
            name="WriteFile",
            arguments={"path": "expr_goal_case/expr_eval.py"},
        ),
        tool_result=ToolResult(output="", error="nope"),
    )
    end_event = AgentEvent(type=AgentEventType.TOOL_CALL_END)

    assert _changed_path_from_subagent_event(read_event) is None
    assert _changed_path_from_subagent_event(failed_write) is None
    assert _changed_path_from_subagent_event(end_event) is None
