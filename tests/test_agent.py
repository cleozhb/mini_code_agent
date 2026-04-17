"""Agent 核心循环测试.

包含：
  - 单元测试（Mock LLM，验证循环逻辑）
  - 集成测试（真实 LLM 调用，需要 API Key）

运行:
  uv run pytest tests/test_agent.py -xvs
"""

from __future__ import annotations

import os
import textwrap
from pathlib import Path

import pytest

from mini_code_agent.core import Agent, AgentResult, build_system_prompt
from mini_code_agent.llm import (
    LLMClient,
    LLMResponse,
    Message,
    TokenUsage,
    ToolCall,
    ToolParam,
    create_client,
)
from mini_code_agent.tools import (
    ReadFileTool,
    WriteFileTool,
    BashTool,
    GrepTool,
    ListDirTool,
    ToolRegistry,
)


# ==================================================================
# 辅助：Mock LLM 客户端
# ==================================================================


class MockLLMClient(LLMClient):
    """可预设响应序列的 Mock LLM 客户端."""

    def __init__(self, responses: list[LLMResponse]) -> None:
        super().__init__(model="mock")
        self._responses = list(responses)
        self._call_index = 0

    async def chat(
        self,
        messages: list[Message],
        tools: list[ToolParam] | None = None,
    ) -> LLMResponse:
        if self._call_index >= len(self._responses):
            raise RuntimeError("Mock 响应已用尽")
        resp = self._responses[self._call_index]
        self._call_index += 1
        self._accumulate_usage(resp.usage)
        return resp

    def chat_stream(self, messages, tools=None):
        raise NotImplementedError("Mock 不支持 stream")


# ==================================================================
# 单元测试：纯文本回复
# ==================================================================


@pytest.mark.asyncio
async def test_agent_simple_text_response():
    """Agent 收到纯文本回复时直接返回."""
    mock_client = MockLLMClient([
        LLMResponse(content="你好！我是编程助手。", usage=TokenUsage(100, 20)),
    ])
    registry = ToolRegistry()
    agent = Agent(mock_client, registry, "你是助手")

    result = await agent.run("你好")

    assert result.content == "你好！我是编程助手。"
    assert result.tool_calls_count == 0
    assert result.usage.total_tokens == 120


# ==================================================================
# 单元测试：工具调用 → 文本回复
# ==================================================================


@pytest.mark.asyncio
async def test_agent_tool_call_then_text(tmp_path: Path):
    """Agent 执行工具调用后收到文本回复."""
    # 准备测试文件
    test_file = tmp_path / "hello.txt"
    test_file.write_text("Hello, World!")

    mock_client = MockLLMClient([
        # 第 1 轮：LLM 请求读取文件
        LLMResponse(
            content="",
            tool_calls=[
                ToolCall(
                    id="call_1",
                    name="read_file",
                    arguments={"path": str(test_file)},
                ),
            ],
            usage=TokenUsage(150, 30),
        ),
        # 第 2 轮：LLM 拿到文件内容后回复
        LLMResponse(
            content="文件内容是：Hello, World!",
            usage=TokenUsage(200, 40),
        ),
    ])

    registry = ToolRegistry()
    registry.register(ReadFileTool())

    agent = Agent(mock_client, registry, "你是助手")
    result = await agent.run("读取 hello.txt")

    assert "Hello, World!" in result.content
    assert result.tool_calls_count == 1
    assert result.usage.input_tokens == 350
    assert result.usage.output_tokens == 70


# ==================================================================
# 单元测试：多个工具调用
# ==================================================================


@pytest.mark.asyncio
async def test_agent_multiple_tool_calls(tmp_path: Path):
    """Agent 单轮返回多个 tool_calls 时按顺序全部执行."""
    file_a = tmp_path / "a.txt"
    file_a.write_text("AAA")
    file_b = tmp_path / "b.txt"
    file_b.write_text("BBB")

    mock_client = MockLLMClient([
        # LLM 一次请求读两个文件
        LLMResponse(
            content="",
            tool_calls=[
                ToolCall(id="c1", name="read_file", arguments={"path": str(file_a)}),
                ToolCall(id="c2", name="read_file", arguments={"path": str(file_b)}),
            ],
            usage=TokenUsage(100, 20),
        ),
        # 拿到结果后回复
        LLMResponse(
            content="a.txt 是 AAA，b.txt 是 BBB",
            usage=TokenUsage(200, 40),
        ),
    ])

    registry = ToolRegistry()
    registry.register(ReadFileTool())

    agent = Agent(mock_client, registry, "你是助手")
    result = await agent.run("读取两个文件")

    assert result.tool_calls_count == 2
    assert "AAA" in result.content
    assert "BBB" in result.content


# ==================================================================
# 单元测试：工具不存在时返回错误给 LLM
# ==================================================================


@pytest.mark.asyncio
async def test_agent_unknown_tool():
    """调用不存在的工具时，Agent 将错误信息返回给 LLM 而非崩溃."""
    mock_client = MockLLMClient([
        LLMResponse(
            content="",
            tool_calls=[
                ToolCall(id="c1", name="nonexistent_tool", arguments={}),
            ],
            usage=TokenUsage(100, 20),
        ),
        # LLM 收到错误后给出文本回复
        LLMResponse(
            content="抱歉，该工具不可用。",
            usage=TokenUsage(150, 25),
        ),
    ])

    registry = ToolRegistry()
    agent = Agent(mock_client, registry, "你是助手")
    result = await agent.run("调用一个不存在的工具")

    assert result.tool_calls_count == 1
    assert "不可用" in result.content


# ==================================================================
# 单元测试：工具执行异常不会崩溃
# ==================================================================


@pytest.mark.asyncio
async def test_agent_tool_execution_error():
    """工具执行抛异常时，Agent 把错误信息返回给 LLM."""
    mock_client = MockLLMClient([
        LLMResponse(
            content="",
            tool_calls=[
                ToolCall(
                    id="c1",
                    name="read_file",
                    arguments={"path": "/nonexistent/path/xyz.txt"},
                ),
            ],
            usage=TokenUsage(100, 20),
        ),
        LLMResponse(
            content="文件不存在。",
            usage=TokenUsage(150, 25),
        ),
    ])

    registry = ToolRegistry()
    registry.register(ReadFileTool())

    agent = Agent(mock_client, registry, "你是助手")
    result = await agent.run("读取不存在的文件")

    assert result.tool_calls_count == 1
    # Agent 没有崩溃，正常返回
    assert result.content == "文件不存在。"


# ==================================================================
# 单元测试：循环保护（最多 25 轮）
# ==================================================================


@pytest.mark.asyncio
async def test_agent_max_rounds_protection():
    """超过 MAX_TOOL_ROUNDS 轮后强制收尾."""
    # 构造 25 轮 tool call（刚好用满上限）+ 1 轮收尾
    responses = []
    for i in range(25):
        responses.append(
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(id=f"c{i}", name="list_dir", arguments={}),
                ],
                usage=TokenUsage(10, 5),
            )
        )
    # 收尾回复
    responses.append(
        LLMResponse(content="好的，以上是结果。", usage=TokenUsage(20, 10))
    )

    mock_client = MockLLMClient(responses)
    registry = ToolRegistry()
    registry.register(ListDirTool())

    agent = Agent(mock_client, registry, "你是助手")
    result = await agent.run("一直列目录")

    # 最多执行 25 轮
    assert result.tool_calls_count == 25
    assert result.content == "好的，以上是结果。"
    assert result.stop_reason == "max_rounds"


# ==================================================================
# 单元测试：stop_reason — token 硬超限 / 墙钟超时
# ==================================================================


@pytest.mark.asyncio
async def test_agent_stop_reason_ok_on_normal_finish():
    """正常结束应返回 stop_reason='ok'."""
    mock_client = MockLLMClient([
        LLMResponse(content="搞定。", usage=TokenUsage(10, 5)),
    ])
    agent = Agent(mock_client, ToolRegistry(), "你是助手")
    result = await agent.run("来一个")
    assert result.stop_reason == "ok"


@pytest.mark.asyncio
async def test_agent_stop_reason_max_tokens():
    """LoopGuard 的 max_tokens 硬超限时应中断并标记 stop_reason='max_tokens'."""
    from mini_code_agent.safety import LoopGuard

    # 第一轮就超 1000 预算
    mock_client = MockLLMClient([
        LLMResponse(
            content="",
            tool_calls=[ToolCall(id="c1", name="list_dir", arguments={})],
            usage=TokenUsage(800, 300),
        ),
        # 不应被调用到；若调用会 RuntimeError Mock 用尽
    ])
    registry = ToolRegistry()
    registry.register(ListDirTool())
    lg = LoopGuard(max_tokens=1000, max_rounds=50)
    agent = Agent(mock_client, registry, "你是助手", loop_guard=lg)

    result = await agent.run("test")
    assert result.stop_reason == "max_tokens"


@pytest.mark.asyncio
async def test_agent_tool_calls_errors_counted(tmp_path: Path):
    """tool_result.is_error=True 的次数应被汇总到 tool_calls_errors."""
    # 让 ReadFile 读不存在的文件 → is_error=True
    missing = tmp_path / "nope.txt"
    assert not missing.exists()

    mock_client = MockLLMClient([
        # 1: 连读两个，第一个存在、第二个不存在
        LLMResponse(
            content="",
            tool_calls=[
                ToolCall(id="c1", name="ReadFile", arguments={"path": str(missing)}),
                ToolCall(id="c2", name="ReadFile", arguments={"path": str(missing)}),
            ],
            usage=TokenUsage(50, 10),
        ),
        # 2: 收尾
        LLMResponse(content="读不到", usage=TokenUsage(20, 5)),
    ])
    registry = ToolRegistry()
    registry.register(ReadFileTool())
    agent = Agent(mock_client, registry, "你是助手")

    result = await agent.run("读一下")
    assert result.tool_calls_count == 2
    assert result.tool_calls_errors == 2  # 两次都是 is_error


@pytest.mark.asyncio
async def test_agent_verifier_first_pass_no_retry(tmp_path: Path):
    """改了文件后 verifier 一把过 — first_passed / final_passed 都 True，attempts=1."""
    from mini_code_agent.core import Verifier

    class AlwaysPassVerifier(Verifier):
        async def verify_code_change(self, files_changed, project_path):
            from mini_code_agent.core.verifier import VerificationResult
            return VerificationResult(passed=True)

    target = tmp_path / "x.py"

    mock_client = MockLLMClient([
        # 1: 写文件
        LLMResponse(
            content="",
            tool_calls=[
                ToolCall(
                    id="c1",
                    name="WriteFile",
                    arguments={"path": str(target), "content": "x = 1\n"},
                ),
            ],
            usage=TokenUsage(10, 5),
        ),
        # 2: 收尾
        LLMResponse(content="已写", usage=TokenUsage(10, 5)),
    ])
    registry = ToolRegistry()
    registry.register(WriteFileTool())
    agent = Agent(
        mock_client, registry, "你是助手",
        verifier=AlwaysPassVerifier(),
        project_path=str(tmp_path),
    )
    result = await agent.run("写个文件")
    assert result.verifier_attempts == 1
    assert result.verifier_first_passed is True
    assert result.verifier_final_passed is True


@pytest.mark.asyncio
async def test_agent_verifier_recovery_tracked(tmp_path: Path):
    """首次 verifier 失败、重试后 pass — first=False, final=True, attempts=2."""
    from mini_code_agent.core import Verifier
    from mini_code_agent.core.verifier import VerificationResult

    class FailThenPassVerifier(Verifier):
        def __init__(self):
            super().__init__()
            self._calls = 0

        async def verify_code_change(self, files_changed, project_path):
            self._calls += 1
            if self._calls == 1:
                return VerificationResult(passed=False, errors=["first fail"])
            return VerificationResult(passed=True)

    target = tmp_path / "y.py"

    mock_client = MockLLMClient([
        # 1: 写文件
        LLMResponse(
            content="",
            tool_calls=[
                ToolCall(
                    id="c1",
                    name="WriteFile",
                    arguments={"path": str(target), "content": "y = 1\n"},
                ),
            ],
            usage=TokenUsage(10, 5),
        ),
        # 2: 收尾（第一次）
        LLMResponse(content="已写", usage=TokenUsage(10, 5)),
        # 3: verifier 失败后，retry prompt 触发再一次修复
        LLMResponse(
            content="",
            tool_calls=[
                ToolCall(
                    id="c2",
                    name="WriteFile",
                    arguments={"path": str(target), "content": "y = 2\n"},
                ),
            ],
            usage=TokenUsage(10, 5),
        ),
        # 4: 收尾（修好）
        LLMResponse(content="已修", usage=TokenUsage(10, 5)),
    ])
    registry = ToolRegistry()
    registry.register(WriteFileTool())
    agent = Agent(
        mock_client, registry, "你是助手",
        verifier=FailThenPassVerifier(),
        project_path=str(tmp_path),
    )
    result = await agent.run("写个文件")
    assert result.verifier_attempts == 2
    assert result.verifier_first_passed is False
    assert result.verifier_final_passed is True


@pytest.mark.asyncio
async def test_agent_verifier_not_triggered(tmp_path: Path):
    """没改文件时 verifier 不触发，字段保持默认 None/0."""
    mock_client = MockLLMClient([
        LLMResponse(content="不需要改", usage=TokenUsage(10, 5)),
    ])
    agent = Agent(mock_client, ToolRegistry(), "你是助手")
    result = await agent.run("只是问个问题")
    assert result.verifier_attempts == 0
    assert result.verifier_first_passed is None
    assert result.verifier_final_passed is None


@pytest.mark.asyncio
async def test_agent_stop_reason_timeout():
    """设置了 max_wall_time_seconds 且真的超时后，应返回 stop_reason='timeout'."""
    import asyncio

    class SlowMockClient(LLMClient):
        def __init__(self) -> None:
            super().__init__(model="slow-mock")

        async def chat(self, messages, tools=None):
            await asyncio.sleep(5)  # 远超 timeout
            return LLMResponse(content="太慢了", usage=TokenUsage(10, 5))

        def chat_stream(self, messages, tools=None):
            raise NotImplementedError

    agent = Agent(
        SlowMockClient(),
        ToolRegistry(),
        "你是助手",
        max_wall_time_seconds=0.2,
    )
    result = await agent.run("test")
    assert result.stop_reason == "timeout"
    assert "超时" in result.content


# ==================================================================
# 单元测试：reset 清空对话历史
# ==================================================================


@pytest.mark.asyncio
async def test_agent_reset():
    """reset() 清空对话历史但保留 system prompt."""
    mock_client = MockLLMClient([
        LLMResponse(content="回复1", usage=TokenUsage(10, 5)),
        LLMResponse(content="回复2", usage=TokenUsage(10, 5)),
    ])
    agent = Agent(mock_client, ToolRegistry(), "你是助手")

    await agent.run("消息1")
    assert len(agent.messages) == 3  # system + user + assistant

    agent.reset()
    assert len(agent.messages) == 1  # 只剩 system
    assert agent.messages[0].content == "你是助手"


# ==================================================================
# 单元测试：build_system_prompt
# ==================================================================


def test_build_system_prompt_default():
    """默认 system prompt 包含核心指导原则."""
    prompt = build_system_prompt()
    assert "编程助手" in prompt
    assert "read_file" in prompt


def test_build_system_prompt_with_project_info():
    """注入项目信息后 system prompt 包含对应内容."""
    prompt = build_system_prompt({
        "cwd": "/tmp/my_project",
        "project_name": "测试项目",
        "tech_stack": "Python + FastAPI",
    })
    assert "/tmp/my_project" in prompt
    assert "测试项目" in prompt
    assert "Python + FastAPI" in prompt


# ==================================================================
# 集成测试：真实 LLM（需要 API Key）
# ==================================================================


skip_no_openai_key = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY 未设置",
)


@skip_no_openai_key
@pytest.mark.asyncio
async def test_agent_integration_read_file(tmp_path: Path):
    """集成测试：Agent 用真实 LLM 读取文件.

    在 tmp_path 下创建 README.md，让 Agent 读取并返回内容。
    """
    # 准备测试工作区
    readme = tmp_path / "README.md"
    readme.write_text(
        textwrap.dedent("""\
        # Test Project
        This is a test README for the mini_code_agent integration test.
        Version: 1.0.0
        """)
    )

    # 创建 Agent
    client = create_client("openai")
    registry = ToolRegistry()
    registry.register(ReadFileTool())
    registry.register(ListDirTool())

    system_prompt = build_system_prompt({"cwd": str(tmp_path)})
    agent = Agent(client, registry, system_prompt)

    # 运行
    result = await agent.run(f"请读取 {readme} 文件并告诉我项目版本号。")

    print(f"\n[集成测试] Agent 回复: {result.content}")
    print(f"[集成测试] 工具调用次数: {result.tool_calls_count}")
    print(f"[集成测试] Token 用量: {result.usage}")

    # 验证
    assert result.tool_calls_count >= 1, "应该至少调用一次工具"
    assert "1.0.0" in result.content, "应该能正确提取版本号"
