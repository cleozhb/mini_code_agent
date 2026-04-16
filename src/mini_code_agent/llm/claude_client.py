"""Anthropic Claude 客户端实现."""

from __future__ import annotations

import json
from typing import AsyncIterator

import anthropic

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


class ClaudeClient(LLMClient):
    """基于 Anthropic SDK 的 Claude 客户端."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        super().__init__(model, api_key=api_key, base_url=base_url)
        self.__client: anthropic.AsyncAnthropic | None = None

    @property
    def _client(self) -> anthropic.AsyncAnthropic:
        """延迟初始化，显式传参避免 SDK 读取系统环境变量."""
        if self.__client is None:
            self.__client = anthropic.AsyncAnthropic(
                api_key=self.api_key or "missing-key",
                base_url=self.base_url,
            )
        return self.__client

    # ------------------------------------------------------------------
    # 格式转换：统一格式 -> Anthropic 格式
    # ------------------------------------------------------------------

    def _convert_messages(
        self, messages: list[Message]
    ) -> tuple[str | None, list[dict]]:
        """将统一消息列表转为 Anthropic 的 system + messages 格式.

        Returns:
            (system_prompt, messages_list)
        """
        system: str | None = None
        api_messages: list[dict] = []

        for msg in messages:
            if msg.role == Role.SYSTEM:
                system = msg.content
                continue

            if msg.role == Role.USER:
                api_messages.append({"role": "user", "content": msg.content})

            elif msg.role == Role.ASSISTANT:
                content: list[dict] = []
                if msg.content:
                    content.append({"type": "text", "text": msg.content})
                for tc in msg.tool_calls:
                    content.append(
                        {
                            "type": "tool_use",
                            "id": tc.id,
                            "name": tc.name,
                            "input": tc.arguments,
                        }
                    )
                api_messages.append({"role": "assistant", "content": content})

            elif msg.role == Role.TOOL:
                assert msg.tool_result is not None
                tr = msg.tool_result
                api_messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": tr.tool_call_id,
                                "content": tr.content,
                                "is_error": tr.is_error,
                            }
                        ],
                    }
                )

        return system, api_messages

    def _convert_tools(self, tools: list[ToolParam] | None) -> list[dict] | None:
        """统一 ToolParam -> Anthropic tools 格式."""
        if not tools:
            return None
        return [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.parameters,
            }
            for t in tools
        ]

    def _parse_response(self, resp: anthropic.types.Message) -> LLMResponse:
        """Anthropic 响应 -> 统一 LLMResponse."""
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []

        for block in resp.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=block.id,
                        name=block.name,
                        arguments=block.input if isinstance(block.input, dict) else {},
                    )
                )

        usage = TokenUsage(
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
        )
        self._accumulate_usage(usage)

        return LLMResponse(
            content="\n".join(text_parts),
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
    ) -> LLMResponse:
        system, api_messages = self._convert_messages(messages)
        api_tools = self._convert_tools(tools)

        kwargs: dict = {
            "model": self.model,
            "max_tokens": 4096,
            "messages": api_messages,
        }
        if system:
            kwargs["system"] = system
        if api_tools:
            kwargs["tools"] = api_tools

        try:
            resp = await self._client.messages.create(**kwargs)
        except anthropic.AuthenticationError as e:
            raise LLMAuthError(f"Anthropic 认证失败: {e}") from e
        except anthropic.RateLimitError as e:
            raise LLMRateLimitError(f"Anthropic 速率限制: {e}") from e
        except anthropic.APIError as e:
            raise LLMError(f"Anthropic API 错误: {e}") from e

        return self._parse_response(resp)

    async def chat_stream(
        self,
        messages: list[Message],
        tools: list[ToolParam] | None = None,
    ) -> AsyncIterator[StreamDelta]:
        system, api_messages = self._convert_messages(messages)
        api_tools = self._convert_tools(tools)

        kwargs: dict = {
            "model": self.model,
            "max_tokens": 4096,
            "messages": api_messages,
        }
        if system:
            kwargs["system"] = system
        if api_tools:
            kwargs["tools"] = api_tools

        try:
            async with self._client.messages.stream(**kwargs) as stream:
                current_tool_id = ""
                current_tool_name = ""
                tool_args_buffer = ""

                async for event in stream:
                    # --- 文本增量 ---
                    if event.type == "content_block_delta":
                        delta = event.delta
                        if delta.type == "text_delta":
                            yield StreamDelta(
                                type=StreamDeltaType.TEXT,
                                content=delta.text,
                            )
                        elif delta.type == "input_json_delta":
                            tool_args_buffer += delta.partial_json
                            yield StreamDelta(
                                type=StreamDeltaType.TOOL_CALL_DELTA,
                                content=delta.partial_json,
                                tool_call_id=current_tool_id,
                                tool_name=current_tool_name,
                            )

                    # --- 新 content block ---
                    elif event.type == "content_block_start":
                        block = event.content_block
                        if block.type == "tool_use":
                            current_tool_id = block.id
                            current_tool_name = block.name
                            tool_args_buffer = ""
                            yield StreamDelta(
                                type=StreamDeltaType.TOOL_CALL_START,
                                tool_call_id=block.id,
                                tool_name=block.name,
                            )

                    # --- content block 结束 ---
                    elif event.type == "content_block_stop":
                        if current_tool_id:
                            yield StreamDelta(
                                type=StreamDeltaType.TOOL_CALL_END,
                                content=tool_args_buffer,
                                tool_call_id=current_tool_id,
                                tool_name=current_tool_name,
                            )
                            current_tool_id = ""
                            current_tool_name = ""
                            tool_args_buffer = ""

                # 从最终消息中提取用量
                final = await stream.get_final_message()
                usage = TokenUsage(
                    input_tokens=final.usage.input_tokens,
                    output_tokens=final.usage.output_tokens,
                )
                self._accumulate_usage(usage)

                yield StreamDelta(
                    type=StreamDeltaType.FINISH,
                    usage=usage,
                )

        except anthropic.AuthenticationError as e:
            raise LLMAuthError(f"Anthropic 认证失败: {e}") from e
        except anthropic.RateLimitError as e:
            raise LLMRateLimitError(f"Anthropic 速率限制: {e}") from e
        except anthropic.APIError as e:
            raise LLMError(f"Anthropic API 错误: {e}") from e
