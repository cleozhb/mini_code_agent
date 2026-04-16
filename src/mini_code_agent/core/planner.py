"""任务规划器 — 在复杂任务前先让 LLM 产出结构化执行计划."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Literal

from ..llm.base import LLMClient, Message

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------


Complexity = Literal["simple", "medium", "complex"]


@dataclass
class PlanStep:
    """计划中的单个步骤."""

    description: str
    files_involved: list[str] = field(default_factory=list)
    tools_needed: list[str] = field(default_factory=list)
    verification: str = ""

    def to_dict(self) -> dict:
        return {
            "description": self.description,
            "files_involved": list(self.files_involved),
            "tools_needed": list(self.tools_needed),
            "verification": self.verification,
        }


@dataclass
class Plan:
    """完整的执行计划."""

    goal: str
    steps: list[PlanStep] = field(default_factory=list)
    estimated_complexity: Complexity = "medium"

    def to_dict(self) -> dict:
        return {
            "goal": self.goal,
            "steps": [s.to_dict() for s in self.steps],
            "estimated_complexity": self.estimated_complexity,
        }

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

请以 JSON 对象的形式返回计划，严格遵守以下格式：

{
  "goal": "对用户需求的一句话概括",
  "estimated_complexity": "simple | medium | complex",
  "steps": [
    {
      "description": "这一步要做什么，一句话说清楚",
      "files_involved": ["可能涉及到的文件或目录，相对路径"],
      "tools_needed": ["预期会用到的工具名，比如 ReadFile/WriteFile/EditFile/Bash/Grep/ListDir"],
      "verification": "怎么确认这一步完成了（比如：文件编译通过、测试通过、命令返回 0）"
    }
  ]
}

规划时请考虑：
- 需要读取/修改哪些文件、修改顺序
- 依赖关系：哪一步必须先于哪一步
- 每一步如何验证（测试、lint、运行等）
- 简单任务 2-3 步，中等 3-6 步，复杂 6+ 步

复杂度判断：
- simple：单文件小改动、查询类任务
- medium：多文件协同、需要先读后改、新增模块
- complex：跨模块重构、引入新依赖、需要写测试配合

只输出 JSON，不要添加任何解释性文字；不要用 Markdown 代码块包裹。
"""


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------


_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def _extract_json_object(text: str) -> str:
    """尽量从 LLM 输出里摘出一个 JSON 对象串.

    支持：纯 JSON / ```json ``` 包裹 / 前后带说明文字.
    """
    text = text.strip()
    if not text:
        raise PlannerError("LLM 返回空响应")

    # 1) 直接就是 JSON
    if text.startswith("{") and text.rstrip().endswith("}"):
        return text

    # 2) ```json ... ``` / ``` ... ``` 包裹
    m = _JSON_FENCE_RE.search(text)
    if m:
        return m.group(1)

    # 3) 取第一个 { 到最后一个 }
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]

    raise PlannerError(f"无法从响应中解析出 JSON：{text[:200]}")


def _parse_plan(goal: str, raw_text: str) -> Plan:
    """把 LLM 的原始输出解析为 Plan."""
    json_text = _extract_json_object(raw_text)
    try:
        data = json.loads(json_text)
    except json.JSONDecodeError as e:
        raise PlannerError(f"JSON 解析失败: {e}; 内容片段: {json_text[:200]}") from e

    if not isinstance(data, dict):
        raise PlannerError(f"期望 JSON 对象，实际得到 {type(data).__name__}")

    steps_raw = data.get("steps", [])
    if not isinstance(steps_raw, list) or not steps_raw:
        raise PlannerError("计划缺少 steps 或 steps 为空")

    steps: list[PlanStep] = []
    for i, s in enumerate(steps_raw):
        if not isinstance(s, dict):
            raise PlannerError(f"第 {i + 1} 步不是对象")
        description = str(s.get("description", "")).strip()
        if not description:
            raise PlannerError(f"第 {i + 1} 步缺少 description")
        steps.append(
            PlanStep(
                description=description,
                files_involved=[str(x) for x in s.get("files_involved") or []],
                tools_needed=[str(x) for x in s.get("tools_needed") or []],
                verification=str(s.get("verification", "")).strip(),
            )
        )

    complexity = str(data.get("estimated_complexity", "medium")).strip().lower()
    if complexity not in ("simple", "medium", "complex"):
        complexity = "medium"

    plan_goal = str(data.get("goal") or goal).strip() or goal
    return Plan(goal=plan_goal, steps=steps, estimated_complexity=complexity)  # type: ignore[arg-type]


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
        user_lines.append("")
        user_lines.append("请输出 JSON 格式的执行计划。")
        user_message = "\n".join(user_lines)

        messages = [
            Message.system(self.system_prompt),
            Message.user(user_message),
        ]

        # 规划阶段不带工具，纯文本输出
        response = await self.llm_client.chat(messages=messages, tools=None)
        logger.debug("Planner 原始输出: %s", response.content)
        return _parse_plan(goal, response.content)
