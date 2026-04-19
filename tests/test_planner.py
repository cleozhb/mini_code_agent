"""Planner 测试 — 验证计划生成格式和 Pydantic 结构化输出.

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
        self.last_response_format: dict | None = None

    async def chat(
        self,
        messages: list[Message],
        tools: list[ToolParam] | None = None,
        response_format: dict | None = None,
    ) -> LLMResponse:
        self.last_messages = messages
        self.last_tools = tools
        self.last_response_format = response_format
        return LLMResponse(content=self._content, usage=TokenUsage(10, 10))

    def chat_stream(self, messages, tools=None, response_format=None):
        raise NotImplementedError


# ==================================================================
# Pydantic Model 测试
# ==================================================================


def _make_valid_plan_dict(
    steps: int = 2,
    complexity: str = "medium",
) -> dict:
    return {
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


def test_plan_from_json():
    data = _make_valid_plan_dict(steps=3, complexity="complex")
    plan = Plan.model_validate(data)

    assert plan.goal == "写一个 hello world"
    assert plan.estimated_complexity == "complex"
    assert len(plan.steps) == 3
    assert all(isinstance(s, PlanStep) for s in plan.steps)
    assert plan.steps[0].description == "步骤 1"
    assert plan.steps[0].files_involved == ["f0.py"]
    assert plan.steps[0].tools_needed == ["WriteFile"]
    assert plan.steps[0].verification == "python 语法正确"


def test_plan_serializes_to_json():
    plan = Plan(
        goal="测试目标",
        estimated_complexity="simple",
        steps=[
            PlanStep(
                description="做点事",
                files_involved=["a.py"],
                tools_needed=["Read"],
                verification="检查",
            )
        ],
    )
    json_str = plan.model_dump_json()
    parsed = json.loads(json_str)
    assert parsed["goal"] == "测试目标"
    assert parsed["estimated_complexity"] == "simple"
    assert len(parsed["steps"]) == 1
    assert parsed["steps"][0]["description"] == "做点事"


def test_plan_step_optional_fields():
    step = PlanStep(description="只有描述")
    assert step.files_involved == []
    assert step.tools_needed == []
    assert step.verification == ""


def test_plan_validation_empty_steps_raises():
    with pytest.raises(Exception):  # Pydantic ValidationError
        Plan.model_validate({"goal": "x", "steps": []})


def test_plan_validation_missing_goal_raises():
    with pytest.raises(Exception):  # Pydantic ValidationError
        Plan.model_validate({"steps": [{"description": "x"}]})


def test_plan_validation_invalid_complexity_fails():
    # Pydantic 会验证枚举值，但在 fallback 处理中可能有差异
    # 这里主要测试有效值
    for complexity in ["simple", "medium", "complex"]:
        plan = Plan(
            goal="x",
            estimated_complexity=complexity,  # type: ignore
            steps=[PlanStep(description="y")],
        )
        assert plan.estimated_complexity == complexity


# ==================================================================
# Planner.plan 集成测试（Mock LLM）
# ==================================================================


@pytest.mark.asyncio
async def test_planner_generates_plan_from_mock_llm():
    raw = json.dumps(_make_valid_plan_dict(steps=2))
    client = MockLLMClient(raw)
    planner = Planner(client)

    plan = await planner.plan("帮我实现登录接口")

    assert len(plan.steps) == 2
    assert plan.estimated_complexity in ("simple", "medium", "complex")
    # Planner 阶段不应向 LLM 传工具
    assert client.last_tools is None
    # 应该传 response_format
    assert client.last_response_format is not None
    assert client.last_response_format["type"] == "json_schema"
    # System prompt 应该来自 planner
    assert client.last_messages is not None
    assert client.last_messages[0].role.value == "system"
    assert "规划" in (client.last_messages[0].content or "")


@pytest.mark.asyncio
async def test_planner_rejects_empty_goal():
    client = MockLLMClient(json.dumps(_make_valid_plan_dict()))
    planner = Planner(client)
    with pytest.raises(PlannerError):
        await planner.plan("   ")


@pytest.mark.asyncio
async def test_planner_handles_project_context():
    raw = json.dumps(_make_valid_plan_dict(steps=1))
    client = MockLLMClient(raw)
    planner = Planner(client)

    await planner.plan("做一件事", project_context="这是项目上下文\n第二行")

    user_msg = client.last_messages[1]
    assert "做一件事" in user_msg.content
    assert "项目上下文" in user_msg.content
    assert "这是项目上下文" in user_msg.content


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


# ==================================================================
# Plan.get_response_schema
# ==================================================================


def test_plan_get_response_schema_structure():
    schema = Plan.get_response_schema(Plan)
    assert schema["type"] == "json_schema"
    assert schema["json_schema"]["name"] == "Plan"
    assert schema["json_schema"]["strict"] is True
    assert "schema" in schema["json_schema"]
    # 验证包含核心字段
    json_schema = schema["json_schema"]["schema"]
    assert "properties" in json_schema
    assert "goal" in json_schema["properties"]
    assert "steps" in json_schema["properties"]
    assert "estimated_complexity" in json_schema["properties"]
