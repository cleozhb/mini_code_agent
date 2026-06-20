"""Goal mode REPL helper tests."""

from __future__ import annotations

from mini_code_agent.cli.repl import _changed_path_from_subagent_event
from mini_code_agent.core.agent import AgentEvent, AgentEventType
from mini_code_agent.llm.base import Message, ToolCall
from mini_code_agent.llm.base import ToolResult as LLMToolResult
from mini_code_agent.longrun.goal_observer import (
    parse_subagent_stop_reason,
    parse_subagent_total_tokens,
)
from mini_code_agent.longrun.message_checkpoint import (
    restore_checkpoint_messages,
    serialize_checkpoint_messages,
)
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


def test_message_checkpoint_round_trips_tool_call_and_result() -> None:
    messages = [
        Message.system("sys"),
        Message.user("inspect"),
        Message.assistant(
            "calling",
            tool_calls=[ToolCall(
                id="call-1",
                name="ReadFile",
                arguments={"path": "a.py"},
                raw_arguments='{"path":"a.py"}',
                parse_error=None,
            )],
        ),
        Message.tool(LLMToolResult(tool_call_id="call-1", content="content", is_error=False)),
    ]

    raw = serialize_checkpoint_messages(messages)
    restored = restore_checkpoint_messages(raw)

    assert [m.role.value for m in restored] == ["user", "assistant", "tool"]
    assert restored[1].tool_calls[0].id == "call-1"
    assert restored[1].tool_calls[0].arguments == {"path": "a.py"}
    assert restored[2].tool_result is not None
    assert restored[2].tool_result.tool_call_id == "call-1"
    assert restored[2].tool_result.content == "content"


def test_message_checkpoint_rejects_legacy_role_content_format() -> None:
    try:
        restore_checkpoint_messages([{"role": "user", "content": "hello"}])
    except ValueError as exc:
        assert "checkpoint message" in str(exc)
    else:
        raise AssertionError("legacy checkpoint message should fail")


def test_message_checkpoint_drops_unpaired_trailing_tool_call() -> None:
    raw = serialize_checkpoint_messages([
        Message.user("before"),
        Message.assistant(
            "calling",
            tool_calls=[ToolCall(id="call-1", name="ReadFile", arguments={})],
        ),
    ])

    restored = restore_checkpoint_messages(raw)

    assert [m.role.value for m in restored] == ["user"]


def test_goal_observer_parses_current_subagent_metadata() -> None:
    output = "[type: coder] [stop_reason: ok]\n[usage: input=10 output=20 total=30]\nDone"

    assert parse_subagent_stop_reason(output) == "ok"
    assert parse_subagent_total_tokens(output) == 30


def test_goal_observer_parses_legacy_total_tokens_metadata() -> None:
    output = "[stop_reason: max_rounds]\n[usage: total_tokens=42]\nDone"

    assert parse_subagent_stop_reason(output) == "max_rounds"
    assert parse_subagent_total_tokens(output) == 42


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
