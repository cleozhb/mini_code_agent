"""Verifier / RetryController 及错误自愈闭环测试.

运行:
    uv run pytest tests/test_verifier.py -xvs
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from mini_code_agent.core import (
    Agent,
    RetryController,
    VerificationResult,
    Verifier,
)
from mini_code_agent.core.retry import AttemptRecord
from mini_code_agent.llm import (
    LLMClient,
    LLMResponse,
    Message,
    TokenUsage,
    ToolCall,
    ToolParam,
)
from mini_code_agent.tools import ToolRegistry, WriteFileTool


# ==================================================================
# MockLLM（允许按需扩展响应列表）
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
# Verifier 单元测试：语法检查
# ==================================================================


@pytest.mark.asyncio
async def test_verifier_detects_python_syntax_error(tmp_path: Path):
    """Verifier 对语法有问题的 Python 文件能报错."""
    buggy = tmp_path / "buggy.py"
    buggy.write_text("def broken(:\n    return 1\n")

    verifier = Verifier()
    result = await verifier.verify_code_change([str(buggy)], tmp_path)

    assert not result.passed
    assert any("语法错误" in e for e in result.errors)
    assert result.suggestions  # 应该给出修复建议


@pytest.mark.asyncio
async def test_verifier_passes_on_valid_python(tmp_path: Path):
    """语法正确且无测试关联的文件应该通过验证."""
    good = tmp_path / "good.py"
    good.write_text("def greet(name):\n    return f'hi {name}'\n")

    verifier = Verifier()
    result = await verifier.verify_code_change([str(good)], tmp_path)

    assert result.passed
    assert result.errors == []


@pytest.mark.asyncio
async def test_verifier_runs_related_pytest_and_detects_failure(tmp_path: Path):
    """改动的源文件若有对应 test_*.py，应自动跑测试并反馈失败."""
    # 源文件：故意让函数返回错误值
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    (src_dir / "__init__.py").write_text("")
    (src_dir / "mathy.py").write_text(
        textwrap.dedent("""\
        def add(a, b):
            return a - b  # BUG: 应该是 a + b
        """)
    )

    # 对应测试
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_mathy.py").write_text(
        textwrap.dedent("""\
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
        from mathy import add

        def test_add():
            assert add(1, 2) == 3
        """)
    )

    verifier = Verifier()
    result = await verifier.verify_code_change(
        [str(src_dir / "mathy.py")], tmp_path
    )

    assert not result.passed
    assert any("pytest" in e for e in result.errors)


@pytest.mark.asyncio
async def test_verifier_ignores_unknown_file(tmp_path: Path):
    """不存在的文件不会让验证器崩溃."""
    verifier = Verifier()
    result = await verifier.verify_code_change(
        [str(tmp_path / "nope.py")], tmp_path
    )
    assert result.passed
    assert result.errors == []


# ==================================================================
# RetryController 单元测试
# ==================================================================


def test_retry_controller_tracks_attempts():
    rc = RetryController(max_retries=3)
    assert rc.can_retry()
    assert rc.attempts_count == 0

    rc.record_attempt(["err1"], "fix1")
    rc.record_attempt(["err2"], "fix2")
    assert rc.attempts_count == 2
    assert rc.can_retry()

    rc.record_attempt(["err3"], "fix3")
    assert rc.attempts_count == 3
    assert not rc.can_retry()


def test_retry_controller_retry_prompt_includes_history():
    rc = RetryController(max_retries=3)
    rc.record_attempt(["first error"], "first fix attempt")

    prompt = rc.build_retry_prompt(["new error"])
    assert "new error" in prompt
    assert "first error" in prompt
    assert "first fix attempt" in prompt
    # 应提醒换思路
    assert "不要重复" in prompt or "换个" in prompt


def test_retry_controller_giveup_summary():
    rc = RetryController(max_retries=2)
    rc.record_attempt(["e1"], "f1")
    rc.record_attempt(["e2"], "f2")

    text = rc.build_giveup_summary()
    assert "尝试了 2 次" in text
    assert "f1" in text
    assert "f2" in text
    assert "建议你检查" in text


def test_retry_controller_reset():
    rc = RetryController()
    rc.record_attempt(["e"], "f")
    rc.reset()
    assert rc.attempts_count == 0
    assert isinstance(rc.attempts, list)
    assert all(isinstance(a, AttemptRecord) for a in rc.attempts)  # 空列表也符合


# ==================================================================
# Agent 集成测试：mock LLM 驱动自愈闭环
# ==================================================================


@pytest.mark.asyncio
async def test_agent_auto_heals_syntax_error(tmp_path: Path):
    """场景：LLM 第一次写入有语法错误 → Verifier 报错 →
    Agent 把错误反馈给 LLM → LLM 修复 → 再次验证通过.
    """
    target = tmp_path / "hello.py"

    bad_content = "def greet(:\n    return 'hi'\n"  # 语法错误
    good_content = "def greet():\n    return 'hi'\n"

    responses = [
        # 第 1 轮：LLM 先写入一个有 bug 的文件
        LLMResponse(
            content="",
            tool_calls=[
                ToolCall(
                    id="c1",
                    name="WriteFile",
                    arguments={"path": str(target), "content": bad_content},
                ),
            ],
            usage=TokenUsage(100, 20),
        ),
        # 第 2 轮：LLM 给出最终文本（认为任务完成）
        LLMResponse(
            content="我已经写好了 hello.py。",
            usage=TokenUsage(120, 15),
        ),
        # 验证失败 → Agent 把错误回传 → 第 3 轮：LLM 写入正确版本
        LLMResponse(
            content="",
            tool_calls=[
                ToolCall(
                    id="c2",
                    name="WriteFile",
                    arguments={"path": str(target), "content": good_content},
                ),
            ],
            usage=TokenUsage(130, 20),
        ),
        # 第 4 轮：LLM 再次声明完成
        LLMResponse(
            content="已修复语法错误。",
            usage=TokenUsage(140, 18),
        ),
    ]

    mock_client = MockLLMClient(responses)
    registry = ToolRegistry()
    # 使用 AUTO 权限避免确认回调
    write_tool = WriteFileTool()
    write_tool.permission_level = write_tool.permission_level.__class__.AUTO
    registry.register(write_tool)

    verifier = Verifier()
    retry_ctrl = RetryController(max_retries=3)

    agent = Agent(
        mock_client,
        registry,
        "你是助手",
        verifier=verifier,
        retry_controller=retry_ctrl,
        project_path=str(tmp_path),
    )

    result = await agent.run("写一个 hello.py，包含 greet()")

    # 文件最终语法正确
    assert target.read_text() == good_content

    # Agent 执行了 2 次写入（第一次 buggy，第二次 fixed）
    assert result.tool_calls_count == 2
    assert "修复" in result.content or "已修复" in result.content

    # 有 1 次失败尝试被记录
    assert retry_ctrl.attempts_count == 1


@pytest.mark.asyncio
async def test_agent_gives_up_after_max_retries(tmp_path: Path):
    """LLM 怎么修都还是语法错误 → 超过重试次数后给用户汇总."""
    target = tmp_path / "stubborn.py"

    bad_content = "def broken(:\n    pass\n"

    def make_write_response(call_id: str) -> LLMResponse:
        return LLMResponse(
            content="",
            tool_calls=[
                ToolCall(
                    id=call_id,
                    name="WriteFile",
                    arguments={"path": str(target), "content": bad_content},
                ),
            ],
            usage=TokenUsage(50, 10),
        )

    def make_claim_done(msg: str) -> LLMResponse:
        return LLMResponse(content=msg, usage=TokenUsage(60, 10))

    # 4 轮都写入相同的坏内容（1 次正常 + 3 次重试），每次都声称完成
    responses = [
        make_write_response("c1"),
        make_claim_done("第 1 次：已经写好了"),
        make_write_response("c2"),
        make_claim_done("第 2 次：修好了"),
        make_write_response("c3"),
        make_claim_done("第 3 次：修好了"),
        make_write_response("c4"),
        make_claim_done("第 4 次：修好了"),
    ]

    mock_client = MockLLMClient(responses)
    registry = ToolRegistry()
    write_tool = WriteFileTool()
    write_tool.permission_level = write_tool.permission_level.__class__.AUTO
    registry.register(write_tool)

    verifier = Verifier()
    retry_ctrl = RetryController(max_retries=3)

    agent = Agent(
        mock_client,
        registry,
        "你是助手",
        verifier=verifier,
        retry_controller=retry_ctrl,
        project_path=str(tmp_path),
    )

    result = await agent.run("写 stubborn.py")

    # 达到上限
    assert retry_ctrl.attempts_count == 3
    # 最终回复是 give-up summary
    assert "尝试了 3 次" in result.content
    assert "建议你检查" in result.content


@pytest.mark.asyncio
async def test_agent_without_verifier_behaves_normally(tmp_path: Path):
    """没有注入 verifier 的 Agent 不应触发验证循环."""
    target = tmp_path / "x.py"

    responses = [
        LLMResponse(
            content="",
            tool_calls=[
                ToolCall(
                    id="c1",
                    name="WriteFile",
                    arguments={
                        "path": str(target),
                        "content": "def broken(:\n    pass\n",  # 坏也没人管
                    },
                ),
            ],
            usage=TokenUsage(10, 5),
        ),
        LLMResponse(content="完成", usage=TokenUsage(10, 5)),
    ]

    mock_client = MockLLMClient(responses)
    registry = ToolRegistry()
    write_tool = WriteFileTool()
    write_tool.permission_level = write_tool.permission_level.__class__.AUTO
    registry.register(write_tool)

    # 不传 verifier / retry_controller / project_path
    agent = Agent(mock_client, registry, "你是助手")
    result = await agent.run("写一个文件")

    assert result.content == "完成"
    assert result.tool_calls_count == 1
