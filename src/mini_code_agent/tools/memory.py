"""记忆工具：AddMemoryTool, RecallMemoryTool."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field

from .base import PermissionLevel, Tool, ToolResult


# ---------------------------------------------------------------------------
# AddMemoryTool
# ---------------------------------------------------------------------------


class AddMemoryInput(BaseModel):
    type: Literal["convention", "decision", "known_issue"] = Field(
        description="记忆类型：convention=项目约定, decision=技术决策, known_issue=已知问题"
    )
    content: str = Field(description="记忆内容（约定文本 / 决策描述 / 问题描述）")
    reason: str | None = Field(
        default=None, description="原因说明（仅 decision 类型需要）"
    )
    solution: str | None = Field(
        default=None, description="解决方案（仅 known_issue 类型需要）"
    )


@dataclass
class AddMemoryTool(Tool):
    """向项目长期记忆中添加一条记录."""

    InputModel: ClassVar[type[BaseModel]] = AddMemoryInput

    name: str = "add_memory"
    description: str = (
        "向项目长期记忆中添加一条记录。"
        "支持三种类型：convention（项目约定）、decision（技术决策）、"
        "known_issue（已知问题及解法）。记忆会持久化到磁盘。"
    )
    permission_level: PermissionLevel = PermissionLevel.AUTO

    # 由外部注入的 ProjectMemory 实例
    _project_memory: Any = field(default=None, repr=False)

    async def execute(self, **kwargs: Any) -> ToolResult:
        if self._project_memory is None:
            return ToolResult(output="", error="项目记忆未初始化")

        mem_type: str = kwargs["type"]
        content: str = kwargs["content"]

        if mem_type == "convention":
            self._project_memory.add_convention(content)
            return ToolResult(output=f"已添加项目约定: {content}")

        if mem_type == "decision":
            reason = kwargs.get("reason") or ""
            self._project_memory.add_decision(content, reason)
            return ToolResult(output=f"已添加技术决策: {content}")

        if mem_type == "known_issue":
            solution = kwargs.get("solution") or ""
            self._project_memory.add_known_issue(content, solution)
            return ToolResult(output=f"已添加已知问题: {content}")

        return ToolResult(output="", error=f"未知记忆类型: {mem_type}")


# ---------------------------------------------------------------------------
# RecallMemoryTool
# ---------------------------------------------------------------------------


class RecallMemoryInput(BaseModel):
    keyword: str = Field(description="搜索关键词")


@dataclass
class RecallMemoryTool(Tool):
    """搜索项目长期记忆."""

    InputModel: ClassVar[type[BaseModel]] = RecallMemoryInput

    name: str = "recall_memory"
    description: str = (
        "搜索项目长期记忆，查找包含关键词的约定、决策、已知问题。"
        "返回匹配到的所有记忆条目。"
    )
    permission_level: PermissionLevel = PermissionLevel.AUTO

    # 由外部注入的 ProjectMemory 实例
    _project_memory: Any = field(default=None, repr=False)

    async def execute(self, **kwargs: Any) -> ToolResult:
        if self._project_memory is None:
            return ToolResult(output="", error="项目记忆未初始化")

        keyword: str = kwargs["keyword"]
        results = self._project_memory.recall(keyword)

        if not results:
            return ToolResult(output=f"未找到包含 '{keyword}' 的记忆")

        output = f"找到 {len(results)} 条相关记忆:\n" + "\n".join(results)
        return ToolResult(output=output)
