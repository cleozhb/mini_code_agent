"""EditFileTool 测试."""

from __future__ import annotations

from pathlib import Path

import pytest

from mini_code_agent.tools.edit import EditFileTool


class TestEditFileTool:
    """EditFileTool 全面测试."""

    @pytest.fixture
    def tool(self) -> EditFileTool:
        return EditFileTool()

    @pytest.fixture
    def sample_file(self, tmp_path: Path) -> Path:
        """创建一个示例 Python 文件."""
        f = tmp_path / "sample.py"
        f.write_text(
            "def hello():\n"
            "    print('hello')\n"
            "\n"
            "def world():\n"
            "    print('world')\n"
        )
        return f

    # --- 正常替换 ---

    async def test_normal_replace(self, tool: EditFileTool, sample_file: Path) -> None:
        """单处匹配，正常替换."""
        result = await tool.execute(
            path=str(sample_file),
            old_content="    print('hello')",
            new_content="    print('hi there')",
        )
        assert not result.is_error
        assert "已编辑" in result.output
        # diff 输出
        assert "---" in result.output
        assert "+++" in result.output
        # 文件内容已更新
        content = sample_file.read_text()
        assert "print('hi there')" in content
        assert "print('hello')" not in content

    async def test_multiline_replace(self, tool: EditFileTool, sample_file: Path) -> None:
        """多行替换."""
        result = await tool.execute(
            path=str(sample_file),
            old_content="def hello():\n    print('hello')",
            new_content="def greet(name):\n    print(f'hello {name}')",
        )
        assert not result.is_error
        content = sample_file.read_text()
        assert "def greet(name):" in content
        assert "def hello():" not in content

    async def test_delete_content(self, tool: EditFileTool, sample_file: Path) -> None:
        """new_content 为空字符串，删除片段."""
        result = await tool.execute(
            path=str(sample_file),
            old_content="\ndef world():\n    print('world')\n",
            new_content="",
        )
        assert not result.is_error
        content = sample_file.read_text()
        assert "world" not in content
        assert "hello" in content

    # --- 找不到 ---

    async def test_not_found(self, tool: EditFileTool, sample_file: Path) -> None:
        """old_content 不存在于文件中."""
        result = await tool.execute(
            path=str(sample_file),
            old_content="this text does not exist anywhere",
            new_content="replacement",
        )
        assert result.is_error
        assert "not found" in result.error

    async def test_file_not_exist(self, tool: EditFileTool, tmp_path: Path) -> None:
        """文件不存在."""
        result = await tool.execute(
            path=str(tmp_path / "nope.txt"),
            old_content="x",
            new_content="y",
        )
        assert result.is_error
        assert "不存在" in result.error

    # --- 多处匹配 ---

    async def test_multiple_matches(self, tool: EditFileTool, tmp_path: Path) -> None:
        """old_content 在文件中出现多次."""
        f = tmp_path / "dup.py"
        f.write_text("print('a')\nprint('a')\nprint('a')\n")
        result = await tool.execute(
            path=str(f),
            old_content="print('a')",
            new_content="print('b')",
        )
        assert result.is_error
        assert "3" in result.error  # 匹配了 3 处
        assert "唯一匹配" in result.error

    # --- 容错：忽略行尾空白 ---

    async def test_trailing_whitespace_tolerance(
        self, tool: EditFileTool, tmp_path: Path
    ) -> None:
        """文件中有行尾空白，old_content 没有（或反之），应能容错匹配."""
        f = tmp_path / "ws.py"
        # 文件中 print 行末尾带空格
        f.write_text("def foo():  \n    return 42  \n")
        result = await tool.execute(
            path=str(f),
            old_content="def foo():\n    return 42",
            new_content="def foo():\n    return 0",
        )
        assert not result.is_error
        content = f.read_text()
        assert "return 0" in content

    # --- 容错：忽略缩进差异 ---

    async def test_indent_tolerance(self, tool: EditFileTool, tmp_path: Path) -> None:
        """old_content 缩进与文件不一致，应能容错匹配."""
        f = tmp_path / "indent.py"
        f.write_text("class Foo:\n    def bar(self):\n        pass\n")
        # old_content 没有缩进
        result = await tool.execute(
            path=str(f),
            old_content="def bar(self):\n    pass",
            new_content="    def bar(self):\n        return 1",
        )
        assert not result.is_error
        content = f.read_text()
        assert "return 1" in content

    # --- 容错：相似片段建议 ---

    async def test_similar_suggestion(self, tool: EditFileTool, sample_file: Path) -> None:
        """old_content 与文件内容相似但不完全匹配，应给出建议."""
        result = await tool.execute(
            path=str(sample_file),
            old_content="def hello():\n    print('helo')",  # typo: helo
            new_content="replacement",
        )
        assert result.is_error
        assert "Did you mean this?" in result.error

    # --- 权限级别 ---

    async def test_permission_level(self, tool: EditFileTool) -> None:
        from mini_code_agent.tools.base import PermissionLevel

        assert tool.permission_level == PermissionLevel.CONFIRM

    # --- Schema ---

    def test_schema(self, tool: EditFileTool) -> None:
        schema = tool.to_schema()
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "EditFile"
        props = schema["function"]["parameters"]["properties"]
        assert "path" in props
        assert "old_content" in props
        assert "new_content" in props
