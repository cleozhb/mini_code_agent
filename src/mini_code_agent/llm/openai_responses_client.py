"""OpenAI Responses API 客户端实现."""

from __future__ import annotations

import json
from typing import AsyncIterator

import openai

from .base import (
    LLMClient,
    LLMCapabilities,
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


class OpenAIResponsesClient(LLMClient):
    """基于 OpenAI Responses API 的客户端."""

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        super().__init__(model, api_key=api_key, base_url=base_url)
        self.__client: openai.AsyncOpenAI | None = None
        self.capabilities = LLMCapabilities(
            structured_outputs=True,
            strict_tools=True,
            streaming_tools=True,
            prompt_cache=True,
            reasoning_usage=True,
        )

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
    # 格式转换：统一格式 -> Responses API 格式
    # ------------------------------------------------------------------

    def _convert_messages(self, messages: list[Message]) -> list[dict]:
        """统一消息 -> Responses input items."""
        input_items: list[dict] = []

        for msg in messages:
            if msg.role in (Role.SYSTEM, Role.USER):
                input_items.append(
                    {
                        "type": "message",
                        "role": msg.role.value,
                        "content": msg.content or "",
                    }
                )

            elif msg.role == Role.ASSISTANT:
                if msg.content:
                    input_items.append(
                        {
                            "type": "message",
                            "role": "assistant",
                            "content": msg.content,
                        }
                    )
                for tc in msg.tool_calls:
                    input_items.append(
                        {
                            "type": "function_call",
                            "call_id": tc.id,
                            "name": tc.name,
                            "arguments": tc.arguments_json(),
                            "status": "completed",
                        }
                    )

            elif msg.role == Role.TOOL:
                assert msg.tool_result is not None
                tr = msg.tool_result
                input_items.append(
                    {
                        "type": "function_call_output",
                        "call_id": tr.tool_call_id,
                        "output": tr.content,
                        "status": "completed",
                    }
                )

        return input_items

    def _convert_tools(self, tools: list[ToolParam] | None) -> list[dict] | None:
        """统一 ToolParam -> Responses function tools 格式."""
        if not tools:
            return None
        return [
            {
                "type": "function",
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
                "strict": t.strict,
            }
            for t in tools
        ]

    def _convert_response_format(self, response_format: dict | None) -> dict | None:
        """OpenAI Chat 风格 response_format -> Responses text 配置."""
        if not response_format:
            return None

        if (
            response_format.get("type") == "json_schema"
            and isinstance(response_format.get("json_schema"), dict)
        ):
            json_schema = response_format["json_schema"]
            text_format = {
                "type": "json_schema",
                "name": json_schema.get("name", "response"),
                "schema": json_schema["schema"],
            }
            if "strict" in json_schema:
                text_format["strict"] = json_schema["strict"]
            if "description" in json_schema:
                text_format["description"] = json_schema["description"]
            return {"format": text_format}

        return {"format": response_format}

    # ------------------------------------------------------------------
    # 格式转换：Responses API 格式 -> 统一格式
    # ------------------------------------------------------------------

    def _get(self, obj: object, key: str, default: object = None) -> object:
        """兼容 SDK model 和测试里使用的 dict/简单对象."""
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    def _usage_from_response(self, resp: object) -> TokenUsage:
        """从 Responses usage 提取统一用量."""
        raw_usage = self._get(resp, "usage")
        if raw_usage is None:
            return TokenUsage()

        input_details = self._get(raw_usage, "input_tokens_details")
        output_details = self._get(raw_usage, "output_tokens_details")
        return TokenUsage(
            input_tokens=int(self._get(raw_usage, "input_tokens", 0) or 0),
            output_tokens=int(self._get(raw_usage, "output_tokens", 0) or 0),
            cached_input_tokens=int(self._get(input_details, "cached_tokens", 0) or 0),
            reasoning_tokens=int(self._get(output_details, "reasoning_tokens", 0) or 0),
        )

    def _parse_tool_call_arguments(self, raw_arguments: str) -> tuple[dict, str | None]:
        """解析工具参数 JSON，失败时保留错误给上层观察."""
        raw_arguments = raw_arguments or ""
        if not raw_arguments:
            return {}, None
        try:
            parsed = json.loads(raw_arguments)
        except json.JSONDecodeError as e:
            return {}, str(e)
        if not isinstance(parsed, dict):
            return {}, f"tool arguments must be a JSON object, got {type(parsed).__name__}"
        return parsed, None

    def _tool_call_from_parts(
        self,
        *,
        tool_call_id: str,
        name: str,
        raw_arguments: str,
    ) -> ToolCall:
        arguments, parse_error = self._parse_tool_call_arguments(raw_arguments)
        return ToolCall(
            id=tool_call_id,
            name=name,
            arguments=arguments,
            raw_arguments=raw_arguments,
            parse_error=parse_error,
        )

    def _parse_response(self, resp: object) -> LLMResponse:
        """Responses API 响应 -> 统一 LLMResponse."""
        error = self._get(resp, "error")
        if error:
            raise LLMError(f"OpenAI Responses 返回错误: {error}")

        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []

        for item in self._get(resp, "output", []) or []:
            item_type = self._get(item, "type")
            if item_type == "message":
                for block in self._get(item, "content", []) or []:
                    block_type = self._get(block, "type")
                    if block_type == "output_text":
                        text_parts.append(str(self._get(block, "text", "") or ""))
                    elif block_type == "refusal":
                        text_parts.append(str(self._get(block, "refusal", "") or ""))

            elif item_type == "function_call":
                raw_arguments = str(self._get(item, "arguments", "") or "")
                tool_calls.append(
                    self._tool_call_from_parts(
                        tool_call_id=str(
                            self._get(item, "call_id", None)
                            or self._get(item, "id", "")
                        ),
                        name=str(self._get(item, "name", "") or ""),
                        raw_arguments=raw_arguments,
                    )
                )

        usage = self._usage_from_response(resp)
        self._accumulate_usage(usage)

        return LLMResponse(
            content="\n".join(part for part in text_parts if part),
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
        kwargs: dict = {
            "model": self.model,
            "input": self._convert_messages(messages),
            "store": False,
        }
        api_tools = self._convert_tools(tools)
        text_config = self._convert_response_format(response_format)
        if api_tools:
            kwargs["tools"] = api_tools
        if text_config:
            kwargs["text"] = text_config

        try:
            resp = await self._client.responses.create(**kwargs)
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
        kwargs: dict = {
            "model": self.model,
            "input": self._convert_messages(messages),
            "store": False,
            "stream": True,
        }
        api_tools = self._convert_tools(tools)
        text_config = self._convert_response_format(response_format)
        if api_tools:
            kwargs["tools"] = api_tools
        if text_config:
            kwargs["text"] = text_config

        try:
            stream = await self._client.responses.create(**kwargs)
        except openai.AuthenticationError as e:
            raise LLMAuthError(f"OpenAI 认证失败: {e}") from e
        except openai.RateLimitError as e:
            raise LLMRateLimitError(f"OpenAI 速率限制: {e}") from e
        except openai.APIError as e:
            raise LLMError(f"OpenAI API 错误: {e}") from e

        tool_calls_in_progress: dict[int, dict] = {}

        async for event in stream:
            event_type = self._get(event, "type")

            if event_type == "response.output_text.delta":
                yield StreamDelta(
                    type=StreamDeltaType.TEXT,
                    content=str(self._get(event, "delta", "") or ""),
                )

            elif event_type == "response.output_item.added":
                item = self._get(event, "item")
                if self._get(item, "type") != "function_call":
                    continue
                output_index = int(self._get(event, "output_index", 0) or 0)
                if output_index in tool_calls_in_progress:
                    info = tool_calls_in_progress[output_index]
                    info["id"] = str(
                        self._get(item, "call_id", None)
                        or self._get(item, "id", "")
                        or info["id"]
                    )
                    info["item_id"] = str(self._get(item, "id", "") or info["item_id"])
                    info["name"] = str(self._get(item, "name", "") or info["name"])
                    info["args"] = str(self._get(item, "arguments", "") or info["args"])
                else:
                    info = {
                        "id": str(
                            self._get(item, "call_id", None)
                            or self._get(item, "id", "")
                        ),
                        "item_id": str(self._get(item, "id", "") or ""),
                        "name": str(self._get(item, "name", "") or ""),
                        "args": str(self._get(item, "arguments", "") or ""),
                        "ended": False,
                    }
                    tool_calls_in_progress[output_index] = info
                    yield StreamDelta(
                        type=StreamDeltaType.TOOL_CALL_START,
                        tool_call_id=info["id"],
                        tool_name=info["name"],
                    )

            elif event_type == "response.function_call_arguments.delta":
                output_index = int(self._get(event, "output_index", 0) or 0)
                delta = str(self._get(event, "delta", "") or "")
                if output_index not in tool_calls_in_progress:
                    tool_calls_in_progress[output_index] = {
                        "id": "",
                        "item_id": "",
                        "name": "",
                        "args": "",
                        "ended": False,
                    }
                    yield StreamDelta(type=StreamDeltaType.TOOL_CALL_START)
                info = tool_calls_in_progress[output_index]
                info["args"] += delta
                yield StreamDelta(
                    type=StreamDeltaType.TOOL_CALL_DELTA,
                    content=delta,
                    tool_call_id=info["id"],
                    tool_name=info["name"],
                )

            elif event_type == "response.function_call_arguments.done":
                output_index = int(self._get(event, "output_index", 0) or 0)
                if output_index not in tool_calls_in_progress:
                    tool_calls_in_progress[output_index] = {
                        "id": "",
                        "item_id": "",
                        "name": "",
                        "args": "",
                        "ended": False,
                    }
                    yield StreamDelta(type=StreamDeltaType.TOOL_CALL_START)
                info = tool_calls_in_progress[output_index]
                info["args"] = str(self._get(event, "arguments", "") or info["args"])
                info["name"] = str(self._get(event, "name", "") or info["name"])
                if not info["ended"]:
                    yield StreamDelta(
                        type=StreamDeltaType.TOOL_CALL_END,
                        content=info["args"],
                        tool_call_id=info["id"],
                        tool_name=info["name"],
                    )
                    info["ended"] = True

            elif event_type == "response.output_item.done":
                item = self._get(event, "item")
                if self._get(item, "type") != "function_call":
                    continue
                output_index = int(self._get(event, "output_index", 0) or 0)
                info = tool_calls_in_progress.setdefault(
                    output_index,
                    {"id": "", "item_id": "", "name": "", "args": "", "ended": False},
                )
                info["id"] = str(self._get(item, "call_id", None) or info["id"])
                info["name"] = str(self._get(item, "name", "") or info["name"])
                info["args"] = str(self._get(item, "arguments", "") or info["args"])
                if not info["ended"]:
                    yield StreamDelta(
                        type=StreamDeltaType.TOOL_CALL_END,
                        content=info["args"],
                        tool_call_id=info["id"],
                        tool_name=info["name"],
                    )
                    info["ended"] = True

            elif event_type == "response.completed":
                usage = self._usage_from_response(self._get(event, "response"))
                self._accumulate_usage(usage)
                yield StreamDelta(type=StreamDeltaType.FINISH, usage=usage)

            elif event_type in {"response.failed", "response.incomplete"}:
                raise LLMError(f"OpenAI Responses streaming 失败: {event}")
