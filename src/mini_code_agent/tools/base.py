"""工具系统基础设施：基类、注册表、结果类型、权限级别."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, ClassVar

from pydantic import BaseModel, ValidationError

from ..llm.base import ToolParam


# ---------------------------------------------------------------------------
# 权限级别
# ---------------------------------------------------------------------------


class PermissionLevel(str, Enum):
    """工具执行权限级别."""

    AUTO = "auto"  # 自动执行，无需确认
    CONFIRM = "confirm"  # 需要用户确认
    DENY = "deny"  # 禁止执行


# ---------------------------------------------------------------------------
# 工具执行结果
# ---------------------------------------------------------------------------


@dataclass
class ToolResult:
    """工具执行后的结果."""

    output: str
    error: str | None = None
    exit_code: int | None = None

    @property
    def is_error(self) -> bool:
        return self.error is not None


# ---------------------------------------------------------------------------
# 工具基类
# ---------------------------------------------------------------------------


@dataclass
class Tool(ABC):
    """所有工具的抽象基类.

    子类需实现 execute() 和在类属性中定义 name / description /
    parameters / permission_level。

    子类可定义 InputModel (Pydantic BaseModel) 来自动生成 parameters
    JSON Schema 并在执行前校验输入参数。
    """

    InputModel: ClassVar[type[BaseModel] | None] = None

    name: str = ""
    description: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)  # JSON Schema
    permission_level: PermissionLevel = PermissionLevel.AUTO

    def __post_init__(self) -> None:
        """如果子类定义了 InputModel 且 parameters 为空，自动生成 JSON Schema."""
        if self.InputModel is not None and not self.parameters:
            schema = self.InputModel.model_json_schema()
            schema.pop("title", None)
            self.parameters = schema

    @abstractmethod
    async def execute(self, **kwargs: Any) -> ToolResult:
        """执行工具，返回结果."""
        ...

    async def run(self, arguments: dict[str, Any]) -> ToolResult:
        """校验参数并执行工具.

        如果定义了 InputModel，先做 Pydantic 校验：
        - 校验失败 → 返回 ToolResult(error=...) 让 LLM 自我修正
        - 校验通过 → 调用 execute(**validated_data)

        未定义 InputModel 时退化为直接调用 execute(**arguments)。
        """
        if self.InputModel is not None:
            try:
                validated = self.InputModel.model_validate(arguments)
            except ValidationError as e:
                return ToolResult(
                    output="",
                    error=f"参数校验失败:\n{e}",
                )
            return await self.execute(**validated.model_dump())
        return await self.execute(**arguments)

    def to_schema(self) -> dict[str, Any]:
        """返回 OpenAI function calling 格式的 JSON Schema.

        格式:
        {
            "type": "function",
            "function": {
                "name": "...",
                "description": "...",
                "parameters": { ... JSON Schema ... }
            }
        }
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def to_tool_param(self) -> ToolParam:
        """转换为 LLM 客户端使用的 ToolParam."""
        return ToolParam(
            name=self.name,
            description=self.description,
            parameters=self.parameters,
        )


# ---------------------------------------------------------------------------
# 工具注册表
# ---------------------------------------------------------------------------


class ToolRegistry:
    """管理所有已注册工具的注册表."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """注册一个工具."""
        if not tool.name:
            raise ValueError("工具必须有 name")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        """根据名称查找工具."""
        return self._tools.get(name)

    def list_tools(self) -> list[Tool]:
        """返回所有已注册的工具."""
        return list(self._tools.values())

    def to_schemas(self) -> list[dict[str, Any]]:
        """导出所有工具的 OpenAI function calling schema."""
        return [tool.to_schema() for tool in self._tools.values()]

    def to_tool_params(self) -> list[ToolParam]:
        """导出所有工具的 ToolParam（传给 LLM 客户端）."""
        return [tool.to_tool_param() for tool in self._tools.values()]
