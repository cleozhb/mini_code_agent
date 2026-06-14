"""Anthropic Claude 客户端实现."""

from __future__ import annotations

import json
from typing import AsyncIterator

import anthropic

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


class ClaudeClient(LLMClient):
    """基于 Anthropic SDK 的 Claude 客户端."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        api_key: str | None = None,
        base_url: str | None = None,
        cache_ttl: str = "off",
    ) -> None:
        super().__init__(model, api_key=api_key, base_url=base_url)
        self.__client: anthropic.AsyncAnthropic | None = None
        if cache_ttl not in {"off", "5m", "1h"}:
            raise LLMError(
                f"不支持的 ANTHROPIC_CACHE_TTL: {cache_ttl!r}，可选: off, 5m, 1h"
            )
        self.cache_ttl = cache_ttl
        self.capabilities = LLMCapabilities(
            structured_outputs=True,
            strict_tools=True,
            streaming_tools=True,
            prompt_cache=cache_ttl != "off",
            reasoning_usage=False,
        )

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

    def _cache_control(self) -> dict | None:
        """构造 Anthropic cache_control 配置."""
        if self.cache_ttl == "off":
            return None
        cache_control = {"type": "ephemeral"}
        if self.cache_ttl in {"5m", "1h"}:
            cache_control["ttl"] = self.cache_ttl
        return cache_control

    def _convert_messages(
        self, messages: list[Message]
    ) -> tuple[str | list[dict] | None, list[dict]]:
        """将统一消息列表转为 Anthropic 的 system + messages 格式.

        Returns:
            (system_prompt, messages_list)
        """
        system_parts: list[str] = []
        api_messages: list[dict] = []
        pending_tool_results: list[dict] = []

        def flush_tool_results() -> None:
            nonlocal pending_tool_results
            if pending_tool_results:
                api_messages.append(
                    {"role": "user", "content": pending_tool_results}
                )
                pending_tool_results = []

        for msg in messages:
            if msg.role == Role.SYSTEM:
                if msg.content:
                    system_parts.append(msg.content)
                continue

            if msg.role == Role.USER:
                if pending_tool_results:
                    content = list(pending_tool_results)
                    pending_tool_results = []
                    if msg.content:
                        content.append({"type": "text", "text": msg.content})
                    api_messages.append({"role": "user", "content": content})
                else:
                    api_messages.append({"role": "user", "content": msg.content or ""})

            elif msg.role == Role.ASSISTANT:
                flush_tool_results()
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
                pending_tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tr.tool_call_id,
                        "content": tr.content,
                        "is_error": tr.is_error,
                    }
                )

        flush_tool_results()

        system: str | list[dict] | None = None
        if system_parts:
            system_text = "\n\n".join(system_parts)
            cache_control = self._cache_control()
            if cache_control:
                system = [
                    {
                        "type": "text",
                        "text": system_text,
                        "cache_control": cache_control,
                    }
                ]
            else:
                system = system_text

        return system, api_messages

    def _convert_tools(self, tools: list[ToolParam] | None) -> list[dict] | None:
        """统一 ToolParam -> Anthropic tools 格式."""
        if not tools:
            return None
        api_tools: list[dict] = []
        for t in tools:
            api_tool = {
                "name": t.name,
                "description": t.description,
                "input_schema": t.parameters,
            }
            if t.strict:
                api_tool["strict"] = True
            api_tools.append(api_tool)

        cache_control = self._cache_control()
        if cache_control and api_tools:
            api_tools[-1]["cache_control"] = cache_control
        return api_tools

    def _convert_response_format(self, response_format: dict | None) -> dict | None:
        """OpenAI 风格 response_format -> Anthropic output_config."""
        if not response_format:
            return None
        if (
            response_format.get("type") == "json_schema"
            and isinstance(response_format.get("json_schema"), dict)
        ):
            json_schema = response_format["json_schema"]
            return {
                "format": {
                    "type": "json_schema",
                    "schema": json_schema["schema"],
                }
            }
        raise LLMError(
            "Anthropic 当前只支持 json_schema response_format，"
            f"收到: {response_format!r}"
        )

    def _usage_from_response(self, resp: object) -> TokenUsage:
        """从 Anthropic usage 提取统一用量."""
        usage = getattr(resp, "usage", None)
        if usage is None:
            return TokenUsage()
        return TokenUsage(
            input_tokens=getattr(usage, "input_tokens", 0) or 0,
            output_tokens=getattr(usage, "output_tokens", 0) or 0,
            cache_creation_input_tokens=getattr(
                usage, "cache_creation_input_tokens", 0
            )
            or 0,
            cache_read_input_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
        )

    def _check_structured_output_stop(
        self,
        resp: object,
        response_format: dict | None,
    ) -> None:
        """把 structured output 失败原因转为可读错误."""
        if not response_format:
            return
        stop_reason = getattr(resp, "stop_reason", None)
        if stop_reason == "refusal":
            raise LLMError(
                "Anthropic structured output 被安全策略拒答，响应可能不满足 schema"
            )
        if stop_reason == "max_tokens":
            raise LLMError(
                "Anthropic structured output 因 max_tokens 截断，"
                "请提高 max_tokens 或重试生成"
            )

    def _parse_tool_call_arguments(self, raw_arguments: str) -> tuple[dict, str | None]:
        """解析 streaming tool 参数 JSON."""
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

    def _parse_response(
        self,
        resp: anthropic.types.Message,
        response_format: dict | None = None,
    ) -> LLMResponse:
        """Anthropic 响应 -> 统一 LLMResponse."""
        self._check_structured_output_stop(resp, response_format)
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

        usage = self._usage_from_response(resp)
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
        response_format: dict | None = None,
    ) -> LLMResponse:
        trace_id = self._trace_start_llm_call(
            mode="chat",
            messages=messages,
            tools=tools,
            response_format=response_format,
        )
        system, api_messages = self._convert_messages(messages)
        api_tools = self._convert_tools(tools)
        output_config = self._convert_response_format(response_format)

        kwargs: dict = {
            "model": self.model,
            "max_tokens": 4096,
            "messages": api_messages,
        }
        if system:
            kwargs["system"] = system
        if api_tools:
            kwargs["tools"] = api_tools
        if output_config:
            kwargs["output_config"] = output_config

        self._trace_provider_request(trace_id, kwargs)

        async def _do_call():
            try:
                return await self._client.messages.create(**kwargs)
            except anthropic.AuthenticationError as e:
                self._trace_finish_llm_call(trace_id, error=e)
                raise LLMAuthError(f"Anthropic 认证失败: {e}") from e
            except anthropic.RateLimitError as e:
                raise LLMRateLimitError(f"Anthropic 速率限制: {e}") from e
            except anthropic.BadRequestError as e:
                self._trace_finish_llm_call(trace_id, error=e)
                if response_format and "schema" in str(e).lower():
                    raise LLMError(
                        "Anthropic structured output schema 不兼容，"
                        "请简化 JSON Schema 或关闭 strict/structured outputs"
                    ) from e
                raise LLMError(f"Anthropic 请求错误: {e}") from e
            except anthropic.APIError as e:
                self._trace_finish_llm_call(trace_id, error=e)
                raise LLMError(f"Anthropic API 错误: {e}") from e

        try:
            resp = await self._call_with_retry(_do_call)
        except LLMRateLimitError as e:
            self._trace_finish_llm_call(trace_id, error=e)
            raise

        self._trace_raw_response(trace_id, resp)
        try:
            parsed = self._parse_response(resp, response_format=response_format)
        except Exception as e:
            self._trace_finish_llm_call(trace_id, error=e)
            raise
        self._trace_parsed_response(trace_id, parsed)
        self._trace_finish_llm_call(trace_id, response=parsed)
        return parsed

    async def chat_stream(
        self,
        messages: list[Message],
        tools: list[ToolParam] | None = None,
        response_format: dict | None = None,
    ) -> AsyncIterator[StreamDelta]:
        trace_id = self._trace_start_llm_call(
            mode="stream",
            messages=messages,
            tools=tools,
            response_format=response_format,
        )
        system, api_messages = self._convert_messages(messages)
        api_tools = self._convert_tools(tools)
        output_config = self._convert_response_format(response_format)

        kwargs: dict = {
            "model": self.model,
            "max_tokens": 4096,
            "messages": api_messages,
        }
        if system:
            kwargs["system"] = system
        if api_tools:
            kwargs["tools"] = api_tools
        if output_config:
            kwargs["output_config"] = output_config

        self._trace_provider_request(trace_id, kwargs)
        try:
            async with self._client.messages.stream(**kwargs) as stream:
                current_tool_id = ""
                current_tool_name = ""
                tool_args_buffer = ""
                full_content = ""
                completed_tool_calls: list[ToolCall] = []

                async for event in stream:
                    self._trace_stream_event(trace_id, event)
                    # --- 文本增量 ---
                    if event.type == "content_block_delta":
                        delta = event.delta
                        if delta.type == "text_delta":
                            full_content += delta.text
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
                            completed_tool_calls.append(
                                self._tool_call_from_parts(
                                    tool_call_id=current_tool_id,
                                    name=current_tool_name,
                                    raw_arguments=tool_args_buffer,
                                )
                            )
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
                self._check_structured_output_stop(final, response_format)
                self._trace_raw_response(
                    trace_id,
                    final,
                    label="response.final.raw",
                )
                usage = self._usage_from_response(final)
                self._accumulate_usage(usage)
                parsed = LLMResponse(
                    content=full_content,
                    tool_calls=completed_tool_calls,
                    usage=usage,
                    raw={"stream": True},
                )
                self._trace_parsed_response(trace_id, parsed)
                self._trace_finish_llm_call(trace_id, response=parsed)

                yield StreamDelta(
                    type=StreamDeltaType.FINISH,
                    usage=usage,
                )

        except anthropic.AuthenticationError as e:
            self._trace_finish_llm_call(trace_id, error=e)
            raise LLMAuthError(f"Anthropic 认证失败: {e}") from e
        except anthropic.RateLimitError as e:
            self._trace_finish_llm_call(trace_id, error=e)
            raise LLMRateLimitError(f"Anthropic 速率限制: {e}") from e
        except anthropic.BadRequestError as e:
            self._trace_finish_llm_call(trace_id, error=e)
            if response_format and "schema" in str(e).lower():
                raise LLMError(
                    "Anthropic structured output schema 不兼容，"
                    "请简化 JSON Schema 或关闭 strict/structured outputs"
                ) from e
            raise LLMError(f"Anthropic 请求错误: {e}") from e
        except anthropic.APIError as e:
            self._trace_finish_llm_call(trace_id, error=e)
            raise LLMError(f"Anthropic API 错误: {e}") from e
