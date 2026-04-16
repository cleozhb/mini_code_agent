"""LLM 客户端集成测试.

需要设置环境变量:
  - ANTHROPIC_API_KEY
  - OPENAI_API_KEY

运行:
  uv run pytest tests/test_llm.py -xvs
"""

from __future__ import annotations

import os

import pytest
import pytest_asyncio

from mini_code_agent.llm import (
    Message,
    StreamDeltaType,
    create_client,
)


# ------------------------------------------------------------------
# 跳过条件
# ------------------------------------------------------------------

skip_no_anthropic_key = pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY 未设置",
)

skip_no_openai_key = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY 未设置",
)


# ------------------------------------------------------------------
# Anthropic 测试
# ------------------------------------------------------------------


@skip_no_anthropic_key
@pytest.mark.asyncio
async def test_claude_chat():
    """Claude 客户端: 发送一条消息，验证能收到回复."""
    client = create_client("anthropic")
    messages = [Message.user("请用一句话介绍你自己。")]

    resp = await client.chat(messages)

    assert resp.content, "应该收到非空回复"
    assert resp.usage.total_tokens > 0, "应该有 token 计数"
    print(f"\n[Claude] 回复: {resp.content}")
    print(f"[Claude] Token 用量: {resp.usage}")


@skip_no_anthropic_key
@pytest.mark.asyncio
async def test_claude_chat_stream():
    """Claude 客户端: streaming 模式."""
    client = create_client("anthropic")
    messages = [Message.user("用一句话说 Hello。")]

    chunks: list[str] = []
    async for delta in client.chat_stream(messages):
        if delta.type == StreamDeltaType.TEXT:
            chunks.append(delta.content)
        elif delta.type == StreamDeltaType.FINISH:
            assert delta.usage is not None
            assert delta.usage.total_tokens > 0

    full_text = "".join(chunks)
    assert full_text, "streaming 应该收到文本"
    print(f"\n[Claude stream] 回复: {full_text}")


# ------------------------------------------------------------------
# OpenAI 测试
# ------------------------------------------------------------------


@skip_no_openai_key
@pytest.mark.asyncio
async def test_openai_chat():
    """OpenAI 客户端: 发送一条消息，验证能收到回复."""
    client = create_client("openai")
    messages = [Message.user("请用一句话介绍你自己。")]

    resp = await client.chat(messages)

    assert resp.content, "应该收到非空回复"
    assert resp.usage.total_tokens > 0, "应该有 token 计数"
    print(f"\n[OpenAI] 回复: {resp.content}")
    print(f"[OpenAI] Token 用量: {resp.usage}")


@skip_no_openai_key
@pytest.mark.asyncio
async def test_openai_chat_stream():
    """OpenAI 客户端: streaming 模式."""
    client = create_client("openai")
    messages = [Message.user("用一句话说 Hello。")]

    chunks: list[str] = []
    async for delta in client.chat_stream(messages):
        if delta.type == StreamDeltaType.TEXT:
            chunks.append(delta.content)
        elif delta.type == StreamDeltaType.FINISH:
            assert delta.usage is not None
            assert delta.usage.total_tokens > 0

    full_text = "".join(chunks)
    assert full_text, "streaming 应该收到文本"
    print(f"\n[OpenAI stream] 回复: {full_text}")


# ------------------------------------------------------------------
# 工厂函数测试（不需要 API Key）
# ------------------------------------------------------------------


def test_factory_invalid_provider():
    """工厂函数: 不支持的 provider 应该抛异常."""
    with pytest.raises(Exception, match="不支持的 provider"):
        create_client("unknown")


def test_factory_creates_claude():
    """工厂函数: 创建 Anthropic 客户端."""
    client = create_client("anthropic", model="claude-sonnet-4-20250514")
    assert client.model == "claude-sonnet-4-20250514"


def test_factory_creates_openai():
    """工厂函数: 创建 OpenAI 客户端."""
    client = create_client("openai", model="gpt-4o")
    assert client.model == "gpt-4o"
