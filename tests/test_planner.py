"""Planner 测试 — 验证计划生成格式和解析鲁棒性.

运行:
    uv run pytest tests/test_planner.py -xvs
"""

from __future__ import annotations

import json

import pytest

from mini_code_agent.core.planner import (
    Plan,
    Planner,
    PlannerError,
    PlanStep,
    _extract_json_object,
    _parse_plan,
)
from mini_code_agent.llm import (
    LLMClient,
    LLMResponse,
    Message,
    TokenUsage,
    ToolParam,
)


# ==================================================================
# MockLLM
# ==================================================================


class MockLLMClient(LLMClient):
    """返回预设文本的 Mock LLM."""

    def __init__(self, content: str) -> None:
        super().__init__(model="mock")
        self._content = content
        self.last_messages: list[Message] | None = None
        self.last_tools: list[ToolParam] | None = None

    async def chat(
        self,
        messages: list[Message],
        tools: list[ToolParam] | None = None,
    ) -> LLMResponse:
        self.last_messages = messages
        self.last_tools = tools
        return LLMResponse(content=self._content, usage=TokenUsage(10, 10))

    def chat_stream(self, messages, tools=None):
        raise NotImplementedError


# ==================================================================
# JSON 提取
# ==================================================================


def test_extract_json_pure():
    raw = '{"goal": "x", "steps": []}'
    assert _extract_json_object(raw) == raw


def test_extract_json_fenced():
    raw = '```json\n{"goal": "x"}\n```'
    out = _extract_json_object(raw)
    assert out == '{"goal": "x"}'


def test_extract_json_fenced_no_tag():
    raw = '```\n{"goal": "y"}\n```'
    out = _extract_json_object(raw)
    assert out == '{"goal": "y"}'


def test_extract_json_with_leading_text():
    raw = "下面是计划：\n{\"goal\": \"z\", \"steps\": []}\n希望对你有帮助。"
    out = _extract_json_object(raw)
    parsed = json.loads(out)
    assert parsed["goal"] == "z"


def test_extract_json_empty_raises():
    with pytest.raises(PlannerError):
        _extract_json_object("   ")


def test_extract_json_garbage_raises():
    with pytest.raises(PlannerError):
        _extract_json_object("纯文字，没有 JSON")


# ==================================================================
# _parse_plan
# ==================================================================


def _make_valid_json(
    steps: int = 2,
    complexity: str = "medium",
) -> str:
    data = {
        "goal": "写一个 hello world",
        "estimated_complexity": complexity,
        "steps": [
            {
                "description": f"步骤 {i + 1}",
                "files_involved": [f"f{i}.py"],
                "tools_needed": ["WriteFile"],
                "verification": "python 语法正确",
            }
            for i in range(steps)
        ],
    }
    return json.dumps(data, ensure_ascii=False)


def test_parse_plan_happy_path():
    raw = _make_valid_json(steps=3, complexity="complex")
    plan = _parse_plan("原始目标", raw)

    assert isinstance(plan, Plan)
    assert plan.goal == "写一个 hello world"
    assert plan.estimated_complexity == "complex"
    assert len(plan.steps) == 3
    assert all(isinstance(s, PlanStep) for s in plan.steps)
    assert plan.steps[0].description == "步骤 1"
    assert plan.steps[0].files_involved == ["f0.py"]
    assert plan.steps[0].tools_needed == ["WriteFile"]
    assert plan.steps[0].verification == "python 语法正确"


def test_parse_plan_fallback_complexity():
    raw = _make_valid_json(complexity="weird-value")
    plan = _parse_plan("目标", raw)
    assert plan.estimated_complexity == "medium"


def test_parse_plan_fallback_goal_when_missing():
    data = {
        "steps": [{"description": "干活"}],
    }
    plan = _parse_plan("用户输入的目标", json.dumps(data))
    assert plan.goal == "用户输入的目标"
    assert plan.steps[0].description == "干活"


def test_parse_plan_missing_steps_raises():
    data = {"goal": "x", "steps": []}
    with pytest.raises(PlannerError):
        _parse_plan("goal", json.dumps(data))


def test_parse_plan_step_missing_description_raises():
    data = {
        "goal": "x",
        "steps": [{"files_involved": ["a.py"]}],
    }
    with pytest.raises(PlannerError):
        _parse_plan("goal", json.dumps(data))


def test_parse_plan_tolerates_missing_optional_fields():
    data = {
        "goal": "x",
        "steps": [{"description": "只有描述"}],
    }
    plan = _parse_plan("goal", json.dumps(data))
    step = plan.steps[0]
    assert step.files_involved == []
    assert step.tools_needed == []
    assert step.verification == ""


def test_parse_plan_invalid_json_raises():
    with pytest.raises(PlannerError):
        _parse_plan("goal", "{not a json")


# ==================================================================
# Planner.plan 集成测试（Mock LLM）
# ==================================================================


@pytest.mark.asyncio
async def test_planner_generates_plan_from_mock_llm():
    raw = _make_valid_json(steps=2)
    client = MockLLMClient(raw)
    planner = Planner(client)

    plan = await planner.plan("帮我实现登录接口")

    assert len(plan.steps) == 2
    assert plan.estimated_complexity in ("simple", "medium", "complex")
    # Planner 阶段不应向 LLM 传工具
    assert client.last_tools is None
    # System prompt 应该来自 planner
    assert client.last_messages is not None
    assert client.last_messages[0].role.value == "system"
    assert "规划" in (client.last_messages[0].content or "")


@pytest.mark.asyncio
async def test_planner_rejects_empty_goal():
    client = MockLLMClient(_make_valid_json())
    planner = Planner(client)
    with pytest.raises(PlannerError):
        await planner.plan("   ")


@pytest.mark.asyncio
async def test_planner_handles_fenced_output():
    raw = "这是你的计划：\n```json\n" + _make_valid_json(steps=1) + "\n```\n希望可以帮到你。"
    client = MockLLMClient(raw)
    planner = Planner(client)
    plan = await planner.plan("做一件事")
    assert len(plan.steps) == 1


@pytest.mark.asyncio
async def test_planner_raises_on_unparseable_output():
    client = MockLLMClient("抱歉我无法给出计划")
    planner = Planner(client)
    with pytest.raises(PlannerError):
        await planner.plan("做一件事")


# ==================================================================
# Plan.format_for_prompt
# ==================================================================


def test_plan_format_for_prompt_contains_all_info():
    plan = Plan(
        goal="重构 auth 模块",
        estimated_complexity="complex",
        steps=[
            PlanStep(
                description="读取现有 auth.py",
                files_involved=["src/auth.py"],
                tools_needed=["ReadFile"],
                verification="文件被完整读取",
            ),
            PlanStep(description="拆分为 service/repository"),
        ],
    )
    text = plan.format_for_prompt()
    assert "重构 auth 模块" in text
    assert "complex" in text
    assert "读取现有 auth.py" in text
    assert "src/auth.py" in text
    assert "ReadFile" in text
    assert "拆分为 service/repository" in text


def test_plan_to_dict_roundtrip():
    plan = Plan(
        goal="g",
        estimated_complexity="simple",
        steps=[PlanStep(description="d", files_involved=["a"], tools_needed=["t"])],
    )
    d = plan.to_dict()
    assert d["goal"] == "g"
    assert d["estimated_complexity"] == "simple"
    assert d["steps"][0]["description"] == "d"
    assert d["steps"][0]["files_involved"] == ["a"]
    assert d["steps"][0]["tools_needed"] == ["t"]
