"""OpenAI GPT 客户端实现."""

from __future__ import annotations

import json
from typing import AsyncIterator

import openai

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
)


class OpenAIClient(LLMClient):
    """基于 OpenAI SDK 的 GPT 客户端."""

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        super().__init__(model, api_key=api_key, base_url=base_url)
        self.__client: openai.AsyncOpenAI | None = None

    @property
    def _client(self) -> openai.AsyncOpenAI:
        """延迟初始化，显式传参避免 SDK 读取系统环境变量."""
        if self.__client is None:
            self.__client = openai.AsyncOpenAI(
                api_key=self.api_key or "missing-key",
                base_url=self.base_url,
            )
        return self.__client

    # ------------------------------------------------------------------
    # 格式转换：统一格式 -> OpenAI 格式
    # ------------------------------------------------------------------

    def _convert_messages(self, messages: list[Message]) -> list[dict]:
        """统一消息 -> OpenAI chat messages 格式."""
        api_messages: list[dict] = []

        for msg in messages:
            if msg.role == Role.SYSTEM:
                api_messages.append({"role": "system", "content": msg.content})

            elif msg.role == Role.USER:
                api_messages.append({"role": "user", "content": msg.content})

            elif msg.role == Role.ASSISTANT:
                m: dict = {"role": "assistant"}
                if msg.content:
                    m["content"] = msg.content
                if msg.tool_calls:
                    m["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": tc.arguments_json(),
                            },
                        }
                        for tc in msg.tool_calls
                    ]
                api_messages.append(m)

            elif msg.role == Role.TOOL:
                assert msg.tool_result is not None
                tr = msg.tool_result
                api_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tr.tool_call_id,
                        "content": tr.content,
                    }
                )

        return api_messages

    def _convert_tools(self, tools: list[ToolParam] | None) -> list[dict] | None:
        """统一 ToolParam -> OpenAI function tools 格式."""
        if not tools:
            return None
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in tools
        ]

    def _parse_response(
        self, resp: openai.types.chat.ChatCompletion
    ) -> LLMResponse:
        """OpenAI 响应 -> 统一 LLMResponse."""
        choice = resp.choices[0]
        message = choice.message

        tool_calls: list[ToolCall] = []
        if message.tool_calls:
            for tc in message.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}
                tool_calls.append(
                    ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=args,
                    )
                )

        usage = TokenUsage()
        if resp.usage:
            usage = TokenUsage(
                input_tokens=resp.usage.prompt_tokens,
                output_tokens=resp.usage.completion_tokens,
            )
        self._accumulate_usage(usage)

        return LLMResponse(
            content=message.content or "",
            tool_calls=tool_calls,
            usage=usage,
            raw=resp,
        )

    # ------------------------------------------------------------------
    # 公共接口
    # ------------------------------------------------------------------

    async def chat(
        self,
        messages: list[Message],
        tools: list[ToolParam] | None = None,
        response_format: dict | None = None,
    ) -> LLMResponse:
        api_messages = self._convert_messages(messages)
        api_tools = self._convert_tools(tools)

        kwargs: dict = {
            "model": self.model,
            "messages": api_messages,
        }
        if api_tools:
            kwargs["tools"] = api_tools
        if response_format:
            kwargs["response_format"] = response_format

        try:
            resp = await self._client.chat.completions.create(**kwargs)
        except openai.AuthenticationError as e:
            raise LLMAuthError(f"OpenAI 认证失败: {e}") from e
        except openai.RateLimitError as e:
            raise LLMRateLimitError(f"OpenAI 速率限制: {e}") from e
        except openai.APIError as e:
            raise LLMError(f"OpenAI API 错误: {e}") from e

        return self._parse_response(resp)

    async def chat_stream(
        self,
        messages: list[Message],
        tools: list[ToolParam] | None = None,
        response_format: dict | None = None,
    ) -> AsyncIterator[StreamDelta]:
        api_messages = self._convert_messages(messages)
        api_tools = self._convert_tools(tools)

        kwargs: dict = {
            "model": self.model,
            "messages": api_messages,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if api_tools:
            kwargs["tools"] = api_tools
        if response_format:
            kwargs["response_format"] = response_format

        try:
            stream = await self._client.chat.completions.create(**kwargs)
        except openai.AuthenticationError as e:
            raise LLMAuthError(f"OpenAI 认证失败: {e}") from e
        except openai.RateLimitError as e:
            raise LLMRateLimitError(f"OpenAI 速率限制: {e}") from e
        except openai.APIError as e:
            raise LLMError(f"OpenAI API 错误: {e}") from e

        # 跟踪进行中的 tool call
        tool_calls_in_progress: dict[int, dict] = {}  # index -> {id, name, args}

        async for chunk in stream:
            # 最终 chunk 携带用量信息（无 choices）
            if chunk.usage:
                usage = TokenUsage(
                    input_tokens=chunk.usage.prompt_tokens,
                    output_tokens=chunk.usage.completion_tokens,
                )
                self._accumulate_usage(usage)
                yield StreamDelta(type=StreamDeltaType.FINISH, usage=usage)
                continue

            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta

            # --- 文本增量 ---
            if delta.content:
                yield StreamDelta(
                    type=StreamDeltaType.TEXT,
                    content=delta.content,
                )

            # --- 工具调用增量 ---
            if delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index

                    # 新工具调用开始
                    if idx not in tool_calls_in_progress:
                        tool_calls_in_progress[idx] = {
                            "id": tc_delta.id or "",
                            "name": tc_delta.function.name if tc_delta.function else "",
                            "args": "",
                        }
                        yield StreamDelta(
                            type=StreamDeltaType.TOOL_CALL_START,
                            tool_call_id=tool_calls_in_progress[idx]["id"],
                            tool_name=tool_calls_in_progress[idx]["name"],
                        )

                    # 累积参数
                    if tc_delta.function and tc_delta.function.arguments:
                        tool_calls_in_progress[idx]["args"] += (
                            tc_delta.function.arguments
                        )
                        yield StreamDelta(
                            type=StreamDeltaType.TOOL_CALL_DELTA,
                            content=tc_delta.function.arguments,
                            tool_call_id=tool_calls_in_progress[idx]["id"],
                            tool_name=tool_calls_in_progress[idx]["name"],
                        )

            # --- finish_reason 处理 ---
            # 兼容 OpenAI ("tool_calls") 和 DeepSeek 等 ("stop") 的差异
            finish_reason = chunk.choices[0].finish_reason
            if finish_reason is not None and tool_calls_in_progress:
                for info in tool_calls_in_progress.values():
                    yield StreamDelta(
                        type=StreamDeltaType.TOOL_CALL_END,
                        content=info["args"],
                        tool_call_id=info["id"],
                        tool_name=info["name"],
                    )
                tool_calls_in_progress.clear()

        # 流结束后，如果还有未关闭的 tool calls（某些 API 不发 finish_reason）
        if tool_calls_in_progress:
            for info in tool_calls_in_progress.values():
                yield StreamDelta(
                    type=StreamDeltaType.TOOL_CALL_END,
                    content=info["args"],
                    tool_call_id=info["id"],
                    tool_name=info["name"],
                )
            tool_calls_in_progress.clear()
