"""工具系统测试."""

from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path

import pytest

from mini_code_agent.tools.base import PermissionLevel, Tool, ToolRegistry, ToolResult
from mini_code_agent.tools.file_ops import ReadFileTool, WriteFileTool
from mini_code_agent.tools.shell import BashTool
from mini_code_agent.tools.search import GrepTool, ListDirTool


# ===========================================================================
# base.py 测试
# ===========================================================================


class TestPermissionLevel:
    def test_values(self) -> None:
        assert PermissionLevel.AUTO == "auto"
        assert PermissionLevel.CONFIRM == "confirm"
        assert PermissionLevel.DENY == "deny"


class TestToolResult:
    def test_success(self) -> None:
        r = ToolResult(output="hello")
        assert r.output == "hello"
        assert r.error is None
        assert not r.is_error

    def test_error(self) -> None:
        r = ToolResult(output="", error="boom")
        assert r.is_error
        assert r.error == "boom"

    def test_exit_code(self) -> None:
        r = ToolResult(output="", exit_code=1)
        assert r.exit_code == 1


class TestToolSchema:
    def test_to_schema(self) -> None:
        tool = ReadFileTool()
        schema = tool.to_schema()
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "ReadFile"
        assert "path" in schema["function"]["parameters"]["properties"]

    def test_to_tool_param(self) -> None:
        tool = ReadFileTool()
        param = tool.to_tool_param()
        assert param.name == "ReadFile"
        assert param.description
        assert "path" in param.parameters["properties"]


class TestToolRegistry:
    def test_register_and_get(self) -> None:
        registry = ToolRegistry()
        tool = ReadFileTool()
        registry.register(tool)
        assert registry.get("ReadFile") is tool
        assert registry.get("NotExist") is None

    def test_register_no_name_raises(self) -> None:
        registry = ToolRegistry()

        class BadTool(Tool):
            async def execute(self, **kwargs):
                return ToolResult(output="")

        with pytest.raises(ValueError, match="name"):
            registry.register(BadTool())

    def test_list_tools(self) -> None:
        registry = ToolRegistry()
        registry.register(ReadFileTool())
        registry.register(WriteFileTool())
        assert len(registry.list_tools()) == 2

    def test_to_schemas(self) -> None:
        registry = ToolRegistry()
        registry.register(ReadFileTool())
        registry.register(BashTool())
        schemas = registry.to_schemas()
        assert len(schemas) == 2
        names = {s["function"]["name"] for s in schemas}
        assert names == {"ReadFile", "Bash"}

    def test_to_tool_params(self) -> None:
        registry = ToolRegistry()
        registry.register(GrepTool())
        params = registry.to_tool_params()
        assert len(params) == 1
        assert params[0].name == "Grep"


# ===========================================================================
# ReadFileTool 测试
# ===========================================================================


class TestReadFileTool:
    """ReadFileTool 全面测试."""

    @pytest.fixture
    def tool(self) -> ReadFileTool:
        return ReadFileTool()

    @pytest.fixture
    def small_file(self, tmp_path: Path) -> Path:
        """创建一个 10 行的小文件."""
        f = tmp_path / "small.txt"
        lines = [f"line {i + 1} content" for i in range(10)]
        f.write_text("\n".join(lines))
        return f

    @pytest.fixture
    def large_file(self, tmp_path: Path) -> Path:
        """创建一个 1000 行的大文件."""
        f = tmp_path / "large.txt"
        lines = [f"line {i + 1} content" for i in range(1000)]
        f.write_text("\n".join(lines))
        return f

    @pytest.fixture
    def exact_500_file(self, tmp_path: Path) -> Path:
        """刚好 500 行（边界）."""
        f = tmp_path / "exact500.txt"
        lines = [f"line {i + 1}" for i in range(500)]
        f.write_text("\n".join(lines))
        return f

    @pytest.fixture
    def exact_501_file(self, tmp_path: Path) -> Path:
        """501 行，刚超过阈值."""
        f = tmp_path / "exact501.txt"
        lines = [f"line {i + 1}" for i in range(501)]
        f.write_text("\n".join(lines))
        return f

    # --- 小文件完整读取 ---

    async def test_small_file_full_read(self, tool: ReadFileTool, small_file: Path) -> None:
        result = await tool.execute(path=str(small_file))
        assert not result.is_error
        lines = result.output.splitlines()
        assert len(lines) == 10
        # 每行都带行号前缀
        for i, line in enumerate(lines):
            parts = line.split("\t", 1)
            assert int(parts[0].strip()) == i + 1
            assert parts[1] == f"line {i + 1} content"

    async def test_500_lines_full_read(self, tool: ReadFileTool, exact_500_file: Path) -> None:
        """500 行的文件应该完整返回（不截断）."""
        result = await tool.execute(path=str(exact_500_file))
        assert not result.is_error
        lines = result.output.splitlines()
        assert len(lines) == 500

    # --- 大文件截断读取 ---

    async def test_large_file_truncated(self, tool: ReadFileTool, large_file: Path) -> None:
        result = await tool.execute(path=str(large_file))
        assert not result.is_error
        output = result.output
        # 验证截断提示
        assert "此处省略" in output
        assert "1000" in output  # 总行数
        assert "start_line/end_line" in output
        assert "GrepTool" in output

    async def test_501_lines_truncated(self, tool: ReadFileTool, exact_501_file: Path) -> None:
        """501 行刚超过阈值，应该截断."""
        result = await tool.execute(path=str(exact_501_file))
        assert not result.is_error
        assert "此处省略" in result.output
        assert "501" in result.output

    async def test_truncated_preserves_head_and_tail(
        self, tool: ReadFileTool, large_file: Path
    ) -> None:
        """截断后应保留前 200 行和末 50 行."""
        result = await tool.execute(path=str(large_file))
        output = result.output

        # 应包含第 1 行
        assert "line 1 content" in output
        # 应包含第 200 行
        assert "line 200 content" in output
        # 不应包含第 201 行的内容作为正常行
        # 但截断提示中会包含 "201"
        # 应包含第 951 行（倒数第 50 行）
        assert "line 951 content" in output
        # 应包含第 1000 行
        assert "line 1000 content" in output

    # --- 指定行号范围 ---

    async def test_start_and_end_line(self, tool: ReadFileTool, small_file: Path) -> None:
        result = await tool.execute(path=str(small_file), start_line=3, end_line=5)
        assert not result.is_error
        lines = result.output.splitlines()
        assert len(lines) == 3
        # 第 3 行
        assert "3" in lines[0].split("\t")[0]
        assert "line 3 content" in lines[0]
        # 第 5 行
        assert "5" in lines[2].split("\t")[0]
        assert "line 5 content" in lines[2]

    async def test_start_line_exceeds_total(self, tool: ReadFileTool, small_file: Path) -> None:
        """start_line 超出文件行数，应返回友好错误."""
        result = await tool.execute(path=str(small_file), start_line=100)
        assert result.is_error
        assert "start_line(100)" in result.error
        assert "10" in result.error  # 总行数

    async def test_end_line_exceeds_total(self, tool: ReadFileTool, small_file: Path) -> None:
        """end_line 超出文件行数，应截止到最后一行（不报错）."""
        result = await tool.execute(path=str(small_file), start_line=8, end_line=999)
        assert not result.is_error
        lines = result.output.splitlines()
        assert len(lines) == 3  # 第 8, 9, 10 行
        assert "line 10 content" in lines[-1]

    async def test_only_start_line(self, tool: ReadFileTool, small_file: Path) -> None:
        """只传 start_line，读取到文件末尾."""
        result = await tool.execute(path=str(small_file), start_line=8)
        assert not result.is_error
        lines = result.output.splitlines()
        assert len(lines) == 3  # 第 8, 9, 10 行
        assert "line 8 content" in lines[0]
        assert "line 10 content" in lines[-1]

    async def test_only_end_line(self, tool: ReadFileTool, small_file: Path) -> None:
        """只传 end_line，从第 1 行读到 end_line."""
        result = await tool.execute(path=str(small_file), end_line=3)
        assert not result.is_error
        lines = result.output.splitlines()
        assert len(lines) == 3  # 第 1, 2, 3 行
        assert "line 1 content" in lines[0]
        assert "line 3 content" in lines[-1]

    # --- 边界和错误 ---

    async def test_file_not_found(self, tool: ReadFileTool, tmp_path: Path) -> None:
        result = await tool.execute(path=str(tmp_path / "nonexistent.txt"))
        assert result.is_error
        assert "不存在" in result.error

    async def test_directory_not_file(self, tool: ReadFileTool, tmp_path: Path) -> None:
        result = await tool.execute(path=str(tmp_path))
        assert result.is_error
        assert "不是文件" in result.error

    async def test_empty_file(self, tool: ReadFileTool, tmp_path: Path) -> None:
        f = tmp_path / "empty.txt"
        f.write_text("")
        result = await tool.execute(path=str(f))
        assert not result.is_error

    async def test_permission_level(self, tool: ReadFileTool) -> None:
        assert tool.permission_level == PermissionLevel.AUTO


# ===========================================================================
# WriteFileTool 测试
# ===========================================================================


class TestWriteFileTool:
    @pytest.fixture
    def tool(self) -> WriteFileTool:
        return WriteFileTool()

    async def test_write_new_file(self, tool: WriteFileTool, tmp_path: Path) -> None:
        f = tmp_path / "new.txt"
        result = await tool.execute(path=str(f), content="hello world")
        assert not result.is_error
        assert f.read_text() == "hello world"
        assert "字符" in result.output

    async def test_overwrite_existing(self, tool: WriteFileTool, tmp_path: Path) -> None:
        f = tmp_path / "existing.txt"
        f.write_text("old content")
        result = await tool.execute(path=str(f), content="new content")
        assert not result.is_error
        assert f.read_text() == "new content"
        # 应记录原始内容
        assert tool._original_contents[str(f)] == "old content"

    async def test_create_parent_dirs(self, tool: WriteFileTool, tmp_path: Path) -> None:
        f = tmp_path / "a" / "b" / "c" / "file.txt"
        result = await tool.execute(path=str(f), content="deep")
        assert not result.is_error
        assert f.read_text() == "deep"

    async def test_permission_level(self, tool: WriteFileTool) -> None:
        assert tool.permission_level == PermissionLevel.CONFIRM


# ===========================================================================
# BashTool 测试
# ===========================================================================


class TestBashTool:
    @pytest.fixture
    def tool(self) -> BashTool:
        return BashTool()

    async def test_simple_command(self, tool: BashTool) -> None:
        result = await tool.execute(command="echo hello")
        assert not result.is_error
        assert "hello" in result.output
        assert result.exit_code == 0

    async def test_failing_command(self, tool: BashTool) -> None:
        result = await tool.execute(command="exit 42")
        assert result.is_error
        assert result.exit_code == 42

    async def test_stderr_merged(self, tool: BashTool) -> None:
        result = await tool.execute(command="echo err >&2")
        # stderr 合并到 stdout
        assert "err" in result.output

    async def test_timeout(self) -> None:
        tool = BashTool(timeout=1)
        result = await tool.execute(command="sleep 10")
        assert result.is_error
        assert "超时" in result.error

    async def test_output_truncation(self) -> None:
        tool = BashTool(max_output_lines=10, head_lines=3, tail_lines=3)
        # 生成 20 行输出
        result = await tool.execute(command="seq 1 20")
        assert not result.is_error
        assert "省略" in result.output

    async def test_permission_level(self, tool: BashTool) -> None:
        assert tool.permission_level == PermissionLevel.CONFIRM


# ===========================================================================
# GrepTool 测试
# ===========================================================================


class TestGrepTool:
    @pytest.fixture
    def tool(self) -> GrepTool:
        return GrepTool()

    @pytest.fixture
    def search_dir(self, tmp_path: Path) -> Path:
        """创建一组用于搜索测试的文件."""
        (tmp_path / "foo.py").write_text("def hello():\n    print('world')\n")
        (tmp_path / "bar.py").write_text("class Foo:\n    pass\n")
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "baz.py").write_text("import hello\n")
        return tmp_path

    async def test_basic_search(self, tool: GrepTool, search_dir: Path) -> None:
        result = await tool.execute(pattern="hello", path=str(search_dir))
        assert not result.is_error
        assert "hello" in result.output
        # 应该匹配 foo.py 和 sub/baz.py
        assert "foo.py" in result.output

    async def test_no_match(self, tool: GrepTool, search_dir: Path) -> None:
        result = await tool.execute(pattern="zzzznotexist", path=str(search_dir))
        assert not result.is_error
        assert "未找到" in result.output

    async def test_invalid_path(self, tool: GrepTool) -> None:
        result = await tool.execute(pattern="foo", path="/nonexistent/dir")
        assert result.is_error

    async def test_permission_level(self, tool: GrepTool) -> None:
        assert tool.permission_level == PermissionLevel.AUTO


# ===========================================================================
# ListDirTool 测试
# ===========================================================================


class TestListDirTool:
    @pytest.fixture
    def tool(self) -> ListDirTool:
        return ListDirTool()

    @pytest.fixture
    def dir_tree(self, tmp_path: Path) -> Path:
        """创建测试目录树."""
        (tmp_path / "file1.py").write_text("")
        (tmp_path / "file2.txt").write_text("")
        sub = tmp_path / "subdir"
        sub.mkdir()
        (sub / "nested.py").write_text("")
        # 应被忽略的目录
        (tmp_path / "__pycache__").mkdir()
        (tmp_path / "__pycache__" / "foo.pyc").write_text("")
        (tmp_path / "node_modules").mkdir()
        return tmp_path

    async def test_basic_listing(self, tool: ListDirTool, dir_tree: Path) -> None:
        result = await tool.execute(path=str(dir_tree))
        assert not result.is_error
        assert "file1.py" in result.output
        assert "subdir" in result.output

    async def test_ignores_pycache(self, tool: ListDirTool, dir_tree: Path) -> None:
        result = await tool.execute(path=str(dir_tree))
        assert "__pycache__" not in result.output

    async def test_ignores_node_modules(self, tool: ListDirTool, dir_tree: Path) -> None:
        result = await tool.execute(path=str(dir_tree))
        # 只检查树形结构部分（跳过根目录名那一行）
        tree_lines = result.output.splitlines()[1:]
        assert all("node_modules" not in line for line in tree_lines)

    async def test_nested_files(self, tool: ListDirTool, dir_tree: Path) -> None:
        result = await tool.execute(path=str(dir_tree), max_depth=2)
        assert "nested.py" in result.output

    async def test_depth_limit(self, tool: ListDirTool, dir_tree: Path) -> None:
        result = await tool.execute(path=str(dir_tree), max_depth=0)
        # depth 0 不展开任何子项
        assert "nested.py" not in result.output
        # 但仍然显示根目录名
        assert dir_tree.name in result.output

    async def test_invalid_path(self, tool: ListDirTool) -> None:
        result = await tool.execute(path="/nonexistent/dir")
        assert result.is_error

    async def test_permission_level(self, tool: ListDirTool) -> None:
        assert tool.permission_level == PermissionLevel.AUTO
