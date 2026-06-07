"""Trace recorder 与 Agent trace 集成测试."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from mini_code_agent.core import Agent
from mini_code_agent.llm import (
    LLMClient,
    LLMResponse,
    Message,
    TokenUsage,
    ToolCall,
    ToolParam,
)
from mini_code_agent.tools import ReadFileTool, ToolRegistry
from mini_code_agent.trace import TraceRecorder


class _Unserializable:
    __slots__ = ()

    def __repr__(self) -> str:
        return "<unserializable>"


def _read_events(recorder: TraceRecorder) -> list[dict]:
    return [
        json.loads(line)
        for line in (recorder.session_dir / "events.jsonl").read_text().splitlines()
    ]


def test_trace_recorder_writes_session_events_and_json(tmp_path: Path) -> None:
    recorder = TraceRecorder(
        project_dir=tmp_path,
        provider="openai",
        model="gpt-test",
        session_id="abc123",
    )

    first = recorder.next_llm_call_id()
    second = recorder.next_llm_call_id()
    recorder.record_event("custom", {"obj": SimpleNamespace(x=_Unserializable())})

    assert first == "0001"
    assert second == "0002"
    assert recorder.session_dir.parent == tmp_path / ".agent" / "traces"
    assert (recorder.session_dir / "session.json").is_file()

    session = json.loads((recorder.session_dir / "session.json").read_text())
    assert session["session_id"] == "abc123"
    assert session["project_dir"] == str(tmp_path)
    assert session["trace_schema_version"] == 1

    events = _read_events(recorder)
    assert events[0]["type"] == "session_start"
    assert events[-1]["type"] == "custom"
    assert events[-1]["payload"]["obj"]["x"] == "<unserializable>"


@pytest.mark.asyncio
async def test_agent_trace_records_llm_tool_sequence(tmp_path: Path) -> None:
    target = tmp_path / "hello.txt"
    target.write_text("Hello")

    class TracedMockLLM(LLMClient):
        def __init__(self) -> None:
            super().__init__(model="mock")
            self._responses = [
                LLMResponse(
                    content="",
                    tool_calls=[
                        ToolCall(
                            id="call_1",
                            name="ReadFile",
                            arguments={"path": str(target)},
                        )
                    ],
                    usage=TokenUsage(10, 5),
                ),
                LLMResponse(content="done", usage=TokenUsage(20, 6)),
            ]
            self._index = 0

        async def chat(
            self,
            messages: list[Message],
            tools: list[ToolParam] | None = None,
            response_format: dict | None = None,
        ) -> LLMResponse:
            trace_id = self._trace_start_llm_call(
                mode="chat",
                messages=messages,
                tools=tools,
                response_format=response_format,
            )
            response = self._responses[self._index]
            self._index += 1
            self._trace_provider_request(trace_id, {"mock": True})
            self._trace_parsed_response(trace_id, response)
            self._trace_finish_llm_call(trace_id, response=response)
            return response

        def chat_stream(self, messages, tools=None, response_format=None):
            raise NotImplementedError

    recorder = TraceRecorder(project_dir=tmp_path, provider="mock", model="mock")
    llm = TracedMockLLM()
    llm.set_trace_recorder(recorder)
    registry = ToolRegistry()
    registry.register(ReadFileTool())

    agent = Agent(llm, registry, "你是助手", trace_recorder=recorder)
    result = await agent.run("读取文件")

    assert result.content == "done"
    events = _read_events(recorder)
    event_types = [e["type"] for e in events]
    assert event_types.index("llm_call_finish") < event_types.index("tool_call_start")
    assert event_types.index("tool_call_start") < event_types.index("tool_call_result")

    tool_result = next(e for e in events if e["type"] == "tool_call_result")
    assert tool_result["payload"]["llm_call_id"] == "0001"
    assert tool_result["payload"]["tool_name"] == "ReadFile"
    assert "Hello" in tool_result["payload"]["result"]["output"]

    second_request = json.loads(
        (recorder.session_dir / "llm" / "0002-request.normalized.json").read_text()
    )
    assert any(
        msg["role"] == "tool" and "Hello" in msg["tool_result"]["content"]
        for msg in second_request["messages"]
    )


@pytest.mark.asyncio
async def test_agent_trace_records_missing_tool(tmp_path: Path) -> None:
    class TracedMockLLM(LLMClient):
        def __init__(self) -> None:
            super().__init__(model="mock")
            self._responses = [
                LLMResponse(
                    content="",
                    tool_calls=[ToolCall(id="call_1", name="Nope", arguments={})],
                    usage=TokenUsage(1, 1),
                ),
                LLMResponse(content="handled", usage=TokenUsage(1, 1)),
            ]
            self._index = 0

        async def chat(self, messages, tools=None, response_format=None):
            trace_id = self._trace_start_llm_call(
                mode="chat",
                messages=messages,
                tools=tools,
                response_format=response_format,
            )
            response = self._responses[self._index]
            self._index += 1
            self._trace_parsed_response(trace_id, response)
            self._trace_finish_llm_call(trace_id, response=response)
            return response

        def chat_stream(self, messages, tools=None, response_format=None):
            raise NotImplementedError

    recorder = TraceRecorder(project_dir=tmp_path, provider="mock", model="mock")
    llm = TracedMockLLM()
    llm.set_trace_recorder(recorder)
    agent = Agent(llm, ToolRegistry(), "你是助手", trace_recorder=recorder)

    await agent.run("调用不存在的工具")

    result_event = next(
        e for e in _read_events(recorder)
        if e["type"] == "tool_call_result"
    )
    assert result_event["payload"]["metadata"]["tool_missing"] is True
    assert "未找到工具" in result_event["payload"]["result"]["error"]
