"""测试 Pydantic InputModel 校验机制."""

from __future__ import annotations

import pytest

from mini_code_agent.tools.base import Tool, ToolResult
from mini_code_agent.tools.shell import BashTool, BashInput
from mini_code_agent.tools.search import GrepTool, GrepInput, ListDirTool, ListDirInput
from mini_code_agent.tools.file_ops import ReadFileTool, ReadFileInput, WriteFileTool, WriteFileInput
from mini_code_agent.tools.edit import EditFileTool, EditFileInput
from mini_code_agent.tools.memory import AddMemoryTool, AddMemoryInput, RecallMemoryTool, RecallMemoryInput


# ---------------------------------------------------------------------------
# 基础机制测试
# ---------------------------------------------------------------------------


class TestToolRunValidation:
    """测试 Tool.run() 的校验逻辑."""

    @pytest.mark.asyncio
    async def test_run_valid_args(self, tmp_path):
        """有效参数 → 正常执行."""
        f = tmp_path / "hello.txt"
        f.write_text("hello world", encoding="utf-8")

        tool = ReadFileTool()
        result = await tool.run({"path": str(f)})
        assert not result.is_error
        assert "hello world" in result.output

    @pytest.mark.asyncio
    async def test_run_missing_required_field(self):
        """缺少必填字段 → 返回 error ToolResult."""
        tool = BashTool()
        result = await tool.run({})
        assert result.is_error
        assert "参数校验失败" in result.error
        assert "command" in result.error

    @pytest.mark.asyncio
    async def test_run_wrong_type(self):
        """类型错误 → 返回 error ToolResult."""
        tool = ReadFileTool()
        result = await tool.run({"path": 123})
        assert result.is_error
        assert "参数校验失败" in result.error

    @pytest.mark.asyncio
    async def test_run_fills_defaults(self):
        """可选字段使用默认值."""
        tool = ListDirTool()
        # 不传任何参数，path 和 max_depth 都有默认值
        result = await tool.run({})
        # 不应报校验错误（可能 "." 不存在但那是执行逻辑的问题）
        assert not result.is_error or "参数校验失败" not in (result.error or "")

    @pytest.mark.asyncio
    async def test_run_legacy_tool_without_input_model(self):
        """没有 InputModel 的工具退化为直接执行."""
        from dataclasses import dataclass
        from typing import Any

        @dataclass
        class LegacyTool(Tool):
            name: str = "Legacy"
            description: str = "test"

            async def execute(self, **kwargs: Any) -> ToolResult:
                return ToolResult(output=f"got {kwargs['x']}")

        tool = LegacyTool()
        result = await tool.run({"x": 42})
        assert not result.is_error
        assert "got 42" in result.output


# ---------------------------------------------------------------------------
# Schema 生成测试
# ---------------------------------------------------------------------------


class TestSchemaGeneration:
    """测试 InputModel 自动生成 parameters JSON Schema."""

    def test_schema_has_type_object(self):
        """生成的 schema 包含 type: object."""
        tool = BashTool()
        assert tool.parameters["type"] == "object"
        assert "properties" in tool.parameters

    def test_schema_required_fields(self):
        """required 列表匹配模型中无默认值的字段."""
        tool = BashTool()
        assert "command" in tool.parameters["required"]

    def test_schema_optional_fields_not_required(self):
        """有默认值的字段不在 required 中."""
        tool = GrepTool()
        required = tool.parameters.get("required", [])
        assert "pattern" in required
        assert "path" not in required

    def test_schema_all_optional(self):
        """所有字段都有默认值时 required 为空或不存在."""
        tool = ListDirTool()
        required = tool.parameters.get("required", [])
        assert required == []

    def test_schema_no_title_at_top_level(self):
        """顶层不应有 title 键."""
        tool = ReadFileTool()
        assert "title" not in tool.parameters

    def test_edit_tool_schema_required(self):
        """EditFileTool 应有 3 个必填字段."""
        tool = EditFileTool()
        required = tool.parameters["required"]
        assert set(required) == {"path", "old_content", "new_content"}

    def test_add_memory_schema_enum(self):
        """AddMemoryTool 的 type 字段应有 enum 约束."""
        tool = AddMemoryTool()
        type_schema = tool.parameters["properties"]["type"]
        assert "enum" in type_schema
        assert set(type_schema["enum"]) == {"convention", "decision", "known_issue"}


# ---------------------------------------------------------------------------
# 各工具 InputModel 校验测试
# ---------------------------------------------------------------------------


class TestInputModels:
    """测试每个工具的 InputModel 校验行为."""

    def test_bash_input_valid(self):
        m = BashInput.model_validate({"command": "echo hi"})
        assert m.command == "echo hi"

    def test_bash_input_missing(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            BashInput.model_validate({})

    def test_grep_input_default_path(self):
        m = GrepInput.model_validate({"pattern": "foo"})
        assert m.path == "."

    def test_read_file_input_optional_lines(self):
        m = ReadFileInput.model_validate({"path": "/tmp/x"})
        assert m.start_line is None
        assert m.end_line is None

    def test_read_file_input_with_lines(self):
        m = ReadFileInput.model_validate({"path": "/tmp/x", "start_line": 5, "end_line": 10})
        assert m.start_line == 5
        assert m.end_line == 10

    def test_write_file_input_valid(self):
        m = WriteFileInput.model_validate({"path": "/tmp/x", "content": "hello"})
        assert m.path == "/tmp/x"
        assert m.content == "hello"

    def test_edit_file_input_valid(self):
        m = EditFileInput.model_validate({"path": "/tmp/x", "old_content": "a", "new_content": "b"})
        assert m.old_content == "a"

    def test_add_memory_input_invalid_type(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            AddMemoryInput.model_validate({"type": "invalid", "content": "x"})

    def test_add_memory_input_valid(self):
        m = AddMemoryInput.model_validate({"type": "decision", "content": "use pydantic", "reason": "type safety"})
        assert m.type == "decision"
        assert m.content == "use pydantic"
        assert m.reason == "type safety"
        assert m.solution is None

    def test_recall_memory_input(self):
        m = RecallMemoryInput.model_validate({"keyword": "test"})
        assert m.keyword == "test"
