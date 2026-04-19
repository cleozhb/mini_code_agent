"""任务规划器 — 在复杂任务前先让 LLM 产出结构化执行计划."""

from __future__ import annotations

import logging
from typing import Literal

from pydantic import BaseModel, Field

from ..llm.base import LLMClient, Message

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 数据结构（Pydantic BaseModel，支持 JSON Schema）
# ---------------------------------------------------------------------------


Complexity = Literal["simple", "medium", "complex"]


class PlanStep(BaseModel):
    """计划中的单个步骤."""

    description: str = Field(..., description="这一步要做什么，一句话说清楚")
    files_involved: list[str] = Field(
        default_factory=list,
        description="可能涉及到的文件或目录，相对路径",
    )
    tools_needed: list[str] = Field(
        default_factory=list,
        description="预期会用到的工具名，比如 ReadFile/WriteFile/EditFile/Bash/Grep/ListDir",
    )
    verification: str = Field(default="", description="怎么确认这一步完成了")


class Plan(BaseModel):
    """完整的执行计划."""

    goal: str = Field(..., description="对用户需求的一句话概括")
    steps: list[PlanStep] = Field(..., min_length=1, description="执行步骤列表")
    estimated_complexity: Complexity = Field(
        default="medium", description="simple | medium | complex"
    )

    def format_for_prompt(self) -> str:
        """格式化计划成纯文本，便于把计划作为上下文回传给执行 LLM."""
        lines = [f"任务目标：{self.goal}", f"复杂度：{self.estimated_complexity}", "执行步骤："]
        for i, step in enumerate(self.steps, 1):
            lines.append(f"{i}. {step.description}")
            if step.files_involved:
                lines.append(f"   涉及文件：{', '.join(step.files_involved)}")
            if step.tools_needed:
                lines.append(f"   需要工具：{', '.join(step.tools_needed)}")
            if step.verification:
                lines.append(f"   验证方式：{step.verification}")
        return "\n".join(lines)

    def get_response_schema(self) -> dict:
        """获取 OpenAI response_format 的 JSON Schema."""
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "Plan",
                "strict": True,
                "schema": Plan.model_json_schema(),
            },
        }


# ---------------------------------------------------------------------------
# 异常
# ---------------------------------------------------------------------------


class PlannerError(Exception):
    """规划器相关错误."""


# ---------------------------------------------------------------------------
# 规划用 System Prompt
# ---------------------------------------------------------------------------


PLANNER_SYSTEM_PROMPT = """\
你是一个任务规划者。分析用户的需求，输出结构化的执行计划。
**不要执行任何操作**，也不要调用任何工具，只输出计划本身。

规划时请考虑：
- 需要读取/修改哪些文件、修改顺序
- 依赖关系：哪一步必须先于哪一步
- 每一步如何验证（测试、lint、运行等）
- 简单任务 2-3 步，中等 3-6 步，复杂 6+ 步

复杂度判断：
- simple：单文件小改动、查询类任务
- medium：多文件协同、需要先读后改、新增模块
- complex：跨模块重构、引入新依赖、需要写测试配合

请按照提供的 JSON Schema 输出执行计划。
"""


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------


class Planner:
    """调用 LLM 产出结构化执行计划."""

    def __init__(
        self,
        llm_client: LLMClient,
        system_prompt: str | None = None,
    ) -> None:
        self.llm_client = llm_client
        self.system_prompt = system_prompt or PLANNER_SYSTEM_PROMPT

    async def plan(self, goal: str, project_context: str = "") -> Plan:
        """根据目标生成执行计划.

        Args:
            goal: 用户的任务目标 / 需求原文
            project_context: 可选的项目上下文（文件树摘要、语言、记忆等）

        Returns:
            解析后的 Plan 对象

        Raises:
            PlannerError: 当 LLM 输出无法解析为合法计划时
        """
        if not goal.strip():
            raise PlannerError("goal 不能为空")

        user_lines = [f"用户需求：{goal.strip()}"]
        if project_context.strip():
            user_lines.append("")
            user_lines.append("项目上下文：")
            user_lines.append(project_context.strip())
        user_message = "\n".join(user_lines)

        messages = [
            Message.system(self.system_prompt),
            Message.user(user_message),
        ]

        # 使用 Pydantic 原生结构化输出
        response_format = Plan.get_response_schema(Plan)
        response = await self.llm_client.chat(
            messages=messages,
            tools=None,
            response_format=response_format,
        )
        logger.debug("Planner 原始输出: %s", response.content)

        # Pydantic 自动解析和校验
        try:
            plan = Plan.model_validate_json(response.content)
            return plan
        except Exception as e:
            raise PlannerError(f"Plan 解析失败: {e}") from e
