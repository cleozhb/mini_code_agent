"""LLM provider 协议转换单元测试（不调用真实 API）."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from mini_code_agent.llm.base import (
    LLMError,
    Message,
    TokenUsage,
    ToolCall,
    ToolParam,
    ToolResult,
)
from mini_code_agent.llm.base import StreamDeltaType
from mini_code_agent.llm.claude_client import ClaudeClient
from mini_code_agent.llm.openai_client import OpenAIClient
from mini_code_agent.llm.openai_responses_client import OpenAIResponsesClient


def _ns(**kwargs):
    return SimpleNamespace(**kwargs)


class _AsyncStream:
    def __init__(self, events: list[object]) -> None:
        self._events = events
        self._index = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._index >= len(self._events):
            raise StopAsyncIteration
        event = self._events[self._index]
        self._index += 1
        return event


def _sample_response_format() -> dict:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "Plan",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {"goal": {"type": "string"}},
                "required": ["goal"],
            },
        },
    }


def test_openai_chat_converts_messages_and_tools():
    client = OpenAIClient(model="gpt-test")
    messages = [
        Message.system("sys"),
        Message.user("hello"),
        Message.assistant(
            "thinking",
            tool_calls=[
                ToolCall(
                    id="call_1",
                    name="ReadFile",
                    arguments={"path": "a.py"},
                )
            ],
        ),
        Message.tool(ToolResult(tool_call_id="call_1", content="content")),
    ]

    converted = client._convert_messages(messages)
    assert converted[0] == {"role": "system", "content": "sys"}
    assert converted[1] == {"role": "user", "content": "hello"}
    assert converted[2]["tool_calls"][0]["function"]["arguments"] == '{"path": "a.py"}'
    assert converted[3] == {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": "content",
    }

    tools = client._convert_tools(
        [
            ToolParam("Loose", "", {"type": "object"}),
            ToolParam("Strict", "", {"type": "object"}, strict=True),
        ]
    )
    assert "strict" not in tools[0]["function"]
    assert tools[1]["function"]["strict"] is True


def test_openai_chat_parse_usage_and_bad_tool_arguments():
    client = OpenAIClient(model="gpt-test")
    resp = _ns(
        choices=[
            _ns(
                message=_ns(
                    content="",
                    tool_calls=[
                        _ns(
                            id="call_1",
                            function=_ns(name="ReadFile", arguments='{"path": '),
                        )
                    ],
                )
            )
        ],
        usage=_ns(
            prompt_tokens=100,
            completion_tokens=30,
            prompt_tokens_details=_ns(cached_tokens=40),
            completion_tokens_details=_ns(reasoning_tokens=7),
        ),
    )

    parsed = client._parse_response(resp)
    assert parsed.usage == TokenUsage(
        input_tokens=100,
        output_tokens=30,
        cached_input_tokens=40,
        reasoning_tokens=7,
    )
    assert parsed.tool_calls[0].arguments == {}
    assert parsed.tool_calls[0].raw_arguments == '{"path": '
    assert parsed.tool_calls[0].parse_error is not None


def test_openai_chat_empty_choices_raises_llm_error():
    client = OpenAIClient(model="gpt-test")
    resp = _ns(
        choices=[],
        usage=_ns(
            prompt_tokens=10,
            completion_tokens=2,
            prompt_tokens_details=None,
            completion_tokens_details=None,
        ),
    )

    with pytest.raises(LLMError, match="缺少 choices"):
        client._parse_response(resp)

    assert client.total_usage == TokenUsage(input_tokens=10, output_tokens=2)


@pytest.mark.asyncio
async def test_openai_chat_streaming_parallel_tool_calls():
    class _Completions:
        async def create(self, **_kwargs):
            return _AsyncStream(
                [
                    _ns(
                        usage=None,
                        choices=[
                            _ns(
                                delta=_ns(
                                    content=None,
                                    tool_calls=[
                                        _ns(
                                            index=0,
                                            id="call_1",
                                            function=_ns(name="A", arguments='{"x"'),
                                        ),
                                        _ns(
                                            index=1,
                                            id="call_2",
                                            function=_ns(name="B", arguments='{"y"'),
                                        ),
                                    ],
                                ),
                                finish_reason=None,
                            )
                        ],
                    ),
                    _ns(
                        usage=None,
                        choices=[
                            _ns(
                                delta=_ns(
                                    content=None,
                                    tool_calls=[
                                        _ns(
                                            index=0,
                                            id=None,
                                            function=_ns(name=None, arguments=": 1}"),
                                        ),
                                        _ns(
                                            index=1,
                                            id=None,
                                            function=_ns(name=None, arguments=": 2}"),
                                        ),
                                    ],
                                ),
                                finish_reason="tool_calls",
                            )
                        ],
                    ),
                    _ns(
                        usage=_ns(
                            prompt_tokens=10,
                            completion_tokens=5,
                            prompt_tokens_details=None,
                            completion_tokens_details=None,
                        ),
                        choices=[],
                    ),
                ]
            )

    client = OpenAIClient(model="gpt-test")
    client._OpenAIClient__client = _ns(  # type: ignore[attr-defined]
        chat=_ns(completions=_Completions())
    )

    deltas = [delta async for delta in client.chat_stream([Message.user("hi")])]
    ends = [d for d in deltas if d.type == StreamDeltaType.TOOL_CALL_END]

    assert [(d.tool_call_id, d.tool_name, d.content) for d in ends] == [
        ("call_1", "A", '{"x": 1}'),
        ("call_2", "B", '{"y": 2}'),
    ]
    assert deltas[-1].type == StreamDeltaType.FINISH
    assert deltas[-1].usage == TokenUsage(input_tokens=10, output_tokens=5)


def test_openai_responses_converts_messages_tools_and_response_format():
    client = OpenAIResponsesClient(model="gpt-test")
    messages = [
        Message.system("sys"),
        Message.user("hello"),
        Message.assistant(
            "ok",
            tool_calls=[
                ToolCall("call_1", "ReadFile", {"path": "a.py"})
            ],
        ),
        Message.tool(ToolResult("call_1", "file content")),
    ]

    converted = client._convert_messages(messages)
    assert converted[0] == {"type": "message", "role": "system", "content": "sys"}
    assert converted[2] == {"type": "message", "role": "assistant", "content": "ok"}
    assert converted[3] == {
        "type": "function_call",
        "call_id": "call_1",
        "name": "ReadFile",
        "arguments": '{"path": "a.py"}',
        "status": "completed",
    }
    assert converted[4] == {
        "type": "function_call_output",
        "call_id": "call_1",
        "output": "file content",
        "status": "completed",
    }

    tools = client._convert_tools([ToolParam("ReadFile", "read", {"type": "object"})])
    assert tools == [
        {
            "type": "function",
            "name": "ReadFile",
            "description": "read",
            "parameters": {"type": "object"},
            "strict": False,
        }
    ]
    assert client._convert_response_format(_sample_response_format()) == {
        "format": {
            "type": "json_schema",
            "name": "Plan",
            "schema": _sample_response_format()["json_schema"]["schema"],
            "strict": True,
        }
    }


def test_openai_responses_parse_response_and_usage():
    client = OpenAIResponsesClient(model="gpt-test")
    resp = {
        "output": [
            {
                "type": "message",
                "content": [{"type": "output_text", "text": "hello"}],
            },
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "ReadFile",
                "arguments": '{"path": "a.py"}',
            },
        ],
        "usage": {
            "input_tokens": 80,
            "output_tokens": 20,
            "input_tokens_details": {"cached_tokens": 50},
            "output_tokens_details": {"reasoning_tokens": 5},
        },
    }

    parsed = client._parse_response(resp)
    assert parsed.content == "hello"
    assert parsed.tool_calls[0] == ToolCall(
        id="call_1",
        name="ReadFile",
        arguments={"path": "a.py"},
        raw_arguments='{"path": "a.py"}',
    )
    assert parsed.usage.cached_input_tokens == 50
    assert parsed.usage.reasoning_tokens == 5


@pytest.mark.asyncio
async def test_openai_responses_streaming_function_arguments():
    class _Responses:
        async def create(self, **kwargs):
            assert kwargs["text"] == {
                "format": {
                    "type": "json_schema",
                    "name": "Plan",
                    "schema": _sample_response_format()["json_schema"]["schema"],
                    "strict": True,
                }
            }
            return _AsyncStream(
                [
                    _ns(
                        type="response.output_item.added",
                        output_index=0,
                        item=_ns(
                            type="function_call",
                            id="item_1",
                            call_id="call_1",
                            name="ReadFile",
                            arguments="",
                        ),
                    ),
                    _ns(
                        type="response.function_call_arguments.delta",
                        output_index=0,
                        delta='{"path"',
                    ),
                    _ns(
                        type="response.function_call_arguments.done",
                        output_index=0,
                        name="ReadFile",
                        arguments='{"path": "a.py"}',
                    ),
                    _ns(
                        type="response.completed",
                        response={
                            "usage": {
                                "input_tokens": 1,
                                "output_tokens": 2,
                                "input_tokens_details": {"cached_tokens": 0},
                                "output_tokens_details": {"reasoning_tokens": 0},
                            }
                        },
                    ),
                ]
            )

    client = OpenAIResponsesClient(model="gpt-test")
    client._OpenAIResponsesClient__client = _ns(responses=_Responses())  # type: ignore[attr-defined]

    deltas = [
        delta
        async for delta in client.chat_stream(
            [Message.user("hi")],
            response_format=_sample_response_format(),
        )
    ]

    assert [d.type for d in deltas] == [
        StreamDeltaType.TOOL_CALL_START,
        StreamDeltaType.TOOL_CALL_DELTA,
        StreamDeltaType.TOOL_CALL_END,
        StreamDeltaType.FINISH,
    ]
    assert deltas[2].tool_call_id == "call_1"
    assert deltas[2].tool_name == "ReadFile"
    assert deltas[2].content == '{"path": "a.py"}'


def test_claude_converts_system_tools_results_and_response_format():
    client = ClaudeClient(model="claude-test", cache_ttl="5m")
    system, messages = client._convert_messages(
        [
            Message.system("sys"),
            Message.assistant(
                "",
                tool_calls=[ToolCall("call_1", "ReadFile", {"path": "a.py"})],
            ),
            Message.tool(ToolResult("call_1", "file content")),
            Message.user("continue"),
        ]
    )

    assert system == [
        {
            "type": "text",
            "text": "sys",
            "cache_control": {"type": "ephemeral", "ttl": "5m"},
        }
    ]
    assert messages[0]["role"] == "assistant"
    assert messages[0]["content"][0] == {
        "type": "tool_use",
        "id": "call_1",
        "name": "ReadFile",
        "input": {"path": "a.py"},
    }
    assert messages[1]["content"][0]["type"] == "tool_result"
    assert messages[1]["content"][1] == {"type": "text", "text": "continue"}

    tools = client._convert_tools(
        [ToolParam("ReadFile", "read", {"type": "object"}, strict=True)]
    )
    assert tools[0]["strict"] is True
    assert tools[0]["cache_control"] == {"type": "ephemeral", "ttl": "5m"}
    assert client._convert_response_format(_sample_response_format()) == {
        "format": {
            "type": "json_schema",
            "schema": _sample_response_format()["json_schema"]["schema"],
        }
    }


def test_claude_parse_cache_usage():
    client = ClaudeClient(model="claude-test")
    resp = _ns(
        stop_reason="end_turn",
        content=[_ns(type="text", text="done")],
        usage=_ns(
            input_tokens=10,
            output_tokens=5,
            cache_creation_input_tokens=100,
            cache_read_input_tokens=200,
        ),
    )

    parsed = client._parse_response(resp)
    assert parsed.content == "done"
    assert parsed.usage.total_tokens == 315
    assert parsed.usage.cache_creation_input_tokens == 100
    assert parsed.usage.cache_read_input_tokens == 200


@pytest.mark.asyncio
async def test_claude_streaming_input_json_delta():
    class _ClaudeStream:
        def __init__(self) -> None:
            self._stream = _AsyncStream(
                [
                    _ns(
                        type="content_block_start",
                        content_block=_ns(
                            type="tool_use",
                            id="call_1",
                            name="ReadFile",
                        ),
                    ),
                    _ns(
                        type="content_block_delta",
                        delta=_ns(
                            type="input_json_delta",
                            partial_json='{"path": "a.py"}',
                        ),
                    ),
                    _ns(type="content_block_stop"),
                ]
            )

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        def __aiter__(self):
            return self._stream

        async def get_final_message(self):
            return _ns(
                stop_reason="tool_use",
                usage=_ns(
                    input_tokens=3,
                    output_tokens=4,
                    cache_creation_input_tokens=0,
                    cache_read_input_tokens=0,
                ),
            )

    class _Messages:
        def stream(self, **kwargs):
            assert kwargs["output_config"] == {
                "format": {
                    "type": "json_schema",
                    "schema": _sample_response_format()["json_schema"]["schema"],
                }
            }
            return _ClaudeStream()

    client = ClaudeClient(model="claude-test")
    client._ClaudeClient__client = _ns(messages=_Messages())  # type: ignore[attr-defined]

    deltas = [
        delta
        async for delta in client.chat_stream(
            [Message.user("hi")],
            response_format=_sample_response_format(),
        )
    ]

    assert [d.type for d in deltas] == [
        StreamDeltaType.TOOL_CALL_START,
        StreamDeltaType.TOOL_CALL_DELTA,
        StreamDeltaType.TOOL_CALL_END,
        StreamDeltaType.FINISH,
    ]
    assert deltas[2].content == '{"path": "a.py"}'
    assert deltas[-1].usage == TokenUsage(input_tokens=3, output_tokens=4)
