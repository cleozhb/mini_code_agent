"""LSP 工具测试.

测试策略：
- LSPManager 使用 mock subprocess 模拟语言服务器
- Tool 类使用 mock LSPManager 测试参数处理和输出格式
- 降级场景使用 patch shutil.which
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from mini_code_agent.tools.base import PermissionLevel
from mini_code_agent.tools.lsp import (
    FindReferencesTool,
    GetDiagnosticsTool,
    GetHoverInfoTool,
    GotoDefinitionTool,
    LSPManager,
    _detect_language_from_ext,
    _detect_project_root,
    _find_symbol_column,
    _path_to_uri,
    _read_surrounding_lines,
    _uri_to_path,
)


# ===========================================================================
# 辅助函数测试
# ===========================================================================


class TestHelpers:
    def test_path_to_uri(self) -> None:
        assert _path_to_uri("/foo/bar.py") == "file:///foo/bar.py"

    def test_path_to_uri_relative(self) -> None:
        result = _path_to_uri("test.py")
        assert result.startswith("file:///")
        assert result.endswith("test.py")

    def test_uri_to_path(self) -> None:
        assert _uri_to_path("file:///foo/bar.py") == "/foo/bar.py"

    def test_uri_to_path_no_scheme(self) -> None:
        assert _uri_to_path("/foo/bar.py") == "/foo/bar.py"

    def test_detect_language_python(self) -> None:
        assert _detect_language_from_ext("foo.py") == "python"
        assert _detect_language_from_ext("bar.pyi") == "python"

    def test_detect_language_typescript(self) -> None:
        assert _detect_language_from_ext("app.ts") == "typescript"
        assert _detect_language_from_ext("app.tsx") == "typescript"

    def test_detect_language_javascript(self) -> None:
        assert _detect_language_from_ext("app.js") == "javascript"
        assert _detect_language_from_ext("app.jsx") == "javascript"

    def test_detect_language_go(self) -> None:
        assert _detect_language_from_ext("main.go") == "go"

    def test_detect_language_rust(self) -> None:
        assert _detect_language_from_ext("lib.rs") == "rust"

    def test_detect_language_unknown(self) -> None:
        assert _detect_language_from_ext("file.xyz") is None
        assert _detect_language_from_ext("noext") is None

    def test_detect_project_root(self, tmp_path: Path) -> None:
        (tmp_path / "pyproject.toml").write_text("[project]")
        sub = tmp_path / "src" / "pkg"
        sub.mkdir(parents=True)
        (sub / "main.py").write_text("x = 1")
        root = _detect_project_root(str(sub / "main.py"))
        assert root == str(tmp_path)

    def test_detect_project_root_no_config(self, tmp_path: Path) -> None:
        sub = tmp_path / "isolated"
        sub.mkdir()
        (sub / "test.py").write_text("")
        root = _detect_project_root(str(sub / "test.py"))
        assert isinstance(root, str)

    def test_read_surrounding_lines(self, tmp_path: Path) -> None:
        f = tmp_path / "sample.py"
        f.write_text("line1\nline2\nline3\nline4\nline5\nline6\nline7\n")
        result = _read_surrounding_lines(str(f), 4, context=2)
        assert "line2" in result
        assert "line3" in result
        assert "line4" in result
        assert "line5" in result
        assert "line6" in result
        assert " → " in result

    def test_read_surrounding_lines_nonexistent(self) -> None:
        result = _read_surrounding_lines("/nonexistent/path.py", 1)
        assert "不存在" in result


# ===========================================================================
# _find_symbol_column 测试
# ===========================================================================


class TestFindSymbolColumn:
    def test_find_function_name(self, tmp_path: Path) -> None:
        f = tmp_path / "test.py"
        f.write_text("def fibonacci_recursive(n):\n    pass\n")
        # 在第 1 行找 fibonacci_recursive
        col = _find_symbol_column(str(f), 1, "fibonacci_recursive")
        assert col == 5  # "def " 是 4 个字符，符号从第 5 列开始

    def test_find_class_name(self, tmp_path: Path) -> None:
        f = tmp_path / "test.py"
        f.write_text("class MyClass:\n    pass\n")
        col = _find_symbol_column(str(f), 1, "MyClass")
        assert col == 7  # "class " 是 6 个字符

    def test_find_variable(self, tmp_path: Path) -> None:
        f = tmp_path / "test.py"
        f.write_text("x = 1\nresult = foo(x)\n")
        col = _find_symbol_column(str(f), 2, "foo")
        assert col == 10  # "result = " 是 9 个字符

    def test_word_boundary_matching(self, tmp_path: Path) -> None:
        f = tmp_path / "test.py"
        f.write_text("foobar = 1\nfoo = 2\n")
        # 在第 1 行查找 "foo"——应该不匹配 "foobar"（完整单词匹配）
        # 但因为 line 1 没有完整的 "foo"，会 fallback 到子串匹配
        col = _find_symbol_column(str(f), 2, "foo")
        assert col == 1

    def test_symbol_not_found(self, tmp_path: Path) -> None:
        f = tmp_path / "test.py"
        f.write_text("x = 1\n")
        col = _find_symbol_column(str(f), 1, "nonexistent")
        assert col is None

    def test_line_out_of_range(self, tmp_path: Path) -> None:
        f = tmp_path / "test.py"
        f.write_text("x = 1\n")
        col = _find_symbol_column(str(f), 99, "x")
        assert col is None

    def test_file_not_found(self) -> None:
        col = _find_symbol_column("/nonexistent/file.py", 1, "x")
        assert col is None

    def test_prefers_word_boundary(self, tmp_path: Path) -> None:
        f = tmp_path / "test.py"
        f.write_text("foobar, foo = 1, 2\n")
        col = _find_symbol_column(str(f), 1, "foo")
        assert col == 9  # 匹配完整的 "foo"（第 9 列），而不是 "foobar" 的前 3 个字符


# ===========================================================================
# LSPManager 测试
# ===========================================================================


class TestLSPManager:
    @pytest.fixture
    def manager(self) -> LSPManager:
        return LSPManager()

    def test_initial_state(self, manager: LSPManager) -> None:
        assert not manager.is_ready()
        assert manager._proc is None
        assert manager._language is None
        assert manager._diagnostics == {}

    async def test_server_not_installed(self, manager: LSPManager) -> None:
        with patch("shutil.which", return_value=None):
            with pytest.raises(FileNotFoundError, match="未安装"):
                await manager.start_server("python", "/tmp/project")

    async def test_server_not_installed_with_fallback(self, manager: LSPManager) -> None:
        with patch("shutil.which", return_value=None):
            with pytest.raises(FileNotFoundError, match="未安装"):
                await manager.start_server("python", "/tmp/project")

    async def test_stop_when_not_started(self, manager: LSPManager) -> None:
        await manager.stop_server()
        assert not manager.is_ready()

    async def test_ensure_ready_unknown_language(self, manager: LSPManager) -> None:
        with pytest.raises(FileNotFoundError, match="无法识别"):
            await manager.ensure_ready("file.unknown_ext_xyz")

    async def test_ensure_ready_detects_language(
        self, manager: LSPManager, tmp_path: Path
    ) -> None:
        py_file = tmp_path / "test.py"
        py_file.write_text("x = 1")
        (tmp_path / "pyproject.toml").write_text("[project]")

        with patch.object(
            manager, "start_server", new_callable=AsyncMock
        ) as mock_start:
            await manager.ensure_ready(str(py_file))
            mock_start.assert_called_once_with("python", str(tmp_path))

    async def test_ensure_ready_skips_if_already_running(
        self, manager: LSPManager, tmp_path: Path
    ) -> None:
        manager._language = "python"
        manager._initialized = True
        mock_proc = AsyncMock()
        mock_proc.returncode = None
        manager._proc = mock_proc

        py_file = tmp_path / "test.py"
        py_file.write_text("x = 1")

        with patch.object(
            manager, "start_server", new_callable=AsyncMock
        ) as mock_start:
            await manager.ensure_ready(str(py_file))
            mock_start.assert_not_called()


# ===========================================================================
# GotoDefinitionTool 测试
# ===========================================================================


class TestGotoDefinitionTool:
    @pytest.fixture
    def tool(self) -> GotoDefinitionTool:
        return GotoDefinitionTool()

    def test_permission_level(self, tool: GotoDefinitionTool) -> None:
        assert tool.permission_level == PermissionLevel.AUTO

    def test_name(self, tool: GotoDefinitionTool) -> None:
        assert tool.name == "GotoDefinition"

    def test_schema_has_required_fields(self, tool: GotoDefinitionTool) -> None:
        schema = tool.to_schema()
        params = schema["function"]["parameters"]
        assert "file_path" in params["properties"]
        assert "line" in params["properties"]
        assert "symbol_name" in params["properties"]
        assert set(params["required"]) == {"file_path", "line", "symbol_name"}

    async def test_no_manager(self, tool: GotoDefinitionTool) -> None:
        result = await tool.execute(file_path="test.py", line=1, symbol_name="foo")
        assert result.is_error
        assert "未初始化" in result.error

    async def test_symbol_not_on_line(
        self, tool: GotoDefinitionTool, tmp_path: Path
    ) -> None:
        target_file = tmp_path / "test.py"
        target_file.write_text("x = 1\n")
        tool._lsp_manager = LSPManager()

        result = await tool.execute(
            file_path=str(target_file), line=1, symbol_name="nonexistent"
        )
        assert result.is_error
        assert "未找到符号" in result.error

    async def test_server_not_installed(
        self, tool: GotoDefinitionTool, tmp_path: Path
    ) -> None:
        target_file = tmp_path / "test.py"
        target_file.write_text("def hello(): pass\n")
        manager = LSPManager()
        tool._lsp_manager = manager
        with patch("shutil.which", return_value=None):
            result = await tool.execute(
                file_path=str(target_file), line=1, symbol_name="hello"
            )
            assert result.is_error
            assert "未安装" in result.error or "无法识别" in result.error

    async def test_successful_goto(
        self, tool: GotoDefinitionTool, tmp_path: Path
    ) -> None:
        target_file = tmp_path / "target.py"
        target_file.write_text("def hello():\n    pass\n\nhello()\n")

        mock_manager = AsyncMock(spec=LSPManager)
        mock_manager.is_ready.return_value = True
        mock_manager.open_document.return_value = f"file://{target_file}"
        mock_manager.request.return_value = {
            "uri": f"file://{target_file}",
            "range": {
                "start": {"line": 0, "character": 4},
                "end": {"line": 0, "character": 9},
            },
        }
        tool._lsp_manager = mock_manager

        result = await tool.execute(
            file_path=str(target_file), line=4, symbol_name="hello"
        )
        assert not result.is_error
        assert "target.py:1" in result.output
        assert "def hello" in result.output

    async def test_definition_not_found(
        self, tool: GotoDefinitionTool, tmp_path: Path
    ) -> None:
        f = tmp_path / "test.py"
        f.write_text("foo()\n")

        mock_manager = AsyncMock(spec=LSPManager)
        mock_manager.is_ready.return_value = True
        mock_manager.open_document.return_value = "file:///test.py"
        mock_manager.request.return_value = []
        tool._lsp_manager = mock_manager

        result = await tool.execute(file_path=str(f), line=1, symbol_name="foo")
        assert not result.is_error
        assert "未找到" in result.output

    async def test_definition_none_result(
        self, tool: GotoDefinitionTool, tmp_path: Path
    ) -> None:
        f = tmp_path / "test.py"
        f.write_text("foo()\n")

        mock_manager = AsyncMock(spec=LSPManager)
        mock_manager.is_ready.return_value = True
        mock_manager.open_document.return_value = "file:///test.py"
        mock_manager.request.return_value = None
        tool._lsp_manager = mock_manager

        result = await tool.execute(file_path=str(f), line=1, symbol_name="foo")
        assert not result.is_error
        assert "未找到" in result.output

    async def test_timeout_handling(
        self, tool: GotoDefinitionTool, tmp_path: Path
    ) -> None:
        f = tmp_path / "test.py"
        f.write_text("foo()\n")

        mock_manager = AsyncMock(spec=LSPManager)
        mock_manager.is_ready.return_value = True
        mock_manager.open_document.return_value = "file:///test.py"
        mock_manager.request.side_effect = asyncio.TimeoutError()
        tool._lsp_manager = mock_manager

        result = await tool.execute(file_path=str(f), line=1, symbol_name="foo")
        assert result.is_error
        assert "超时" in result.error

    async def test_location_link_format(
        self, tool: GotoDefinitionTool, tmp_path: Path
    ) -> None:
        src_file = tmp_path / "src.py"
        src_file.write_text("from mod import Foo\nx = Foo()\n")
        target_file = tmp_path / "mod.py"
        target_file.write_text("class Foo:\n    pass\n")

        mock_manager = AsyncMock(spec=LSPManager)
        mock_manager.is_ready.return_value = True
        mock_manager.open_document.return_value = "file:///src.py"
        mock_manager.request.return_value = [
            {
                "targetUri": f"file://{target_file}",
                "targetRange": {
                    "start": {"line": 0, "character": 6},
                    "end": {"line": 0, "character": 9},
                },
            }
        ]
        tool._lsp_manager = mock_manager

        result = await tool.execute(
            file_path=str(src_file), line=2, symbol_name="Foo"
        )
        assert not result.is_error
        assert "mod.py:1" in result.output


# ===========================================================================
# FindReferencesTool 测试
# ===========================================================================


class TestFindReferencesTool:
    @pytest.fixture
    def tool(self) -> FindReferencesTool:
        return FindReferencesTool()

    def test_permission_level(self, tool: FindReferencesTool) -> None:
        assert tool.permission_level == PermissionLevel.AUTO

    def test_name(self, tool: FindReferencesTool) -> None:
        assert tool.name == "FindReferences"

    async def test_no_manager(self, tool: FindReferencesTool) -> None:
        result = await tool.execute(file_path="test.py", line=1, symbol_name="foo")
        assert result.is_error
        assert "未初始化" in result.error

    async def test_symbol_not_on_line(
        self, tool: FindReferencesTool, tmp_path: Path
    ) -> None:
        f = tmp_path / "test.py"
        f.write_text("x = 1\n")
        tool._lsp_manager = LSPManager()

        result = await tool.execute(
            file_path=str(f), line=1, symbol_name="nonexistent"
        )
        assert result.is_error
        assert "未找到符号" in result.error

    async def test_references_found(
        self, tool: FindReferencesTool, tmp_path: Path
    ) -> None:
        f = tmp_path / "test.py"
        f.write_text("def foo(): pass\n")

        mock_manager = AsyncMock(spec=LSPManager)
        mock_manager.is_ready.return_value = True
        mock_manager.open_document.return_value = "file:///test.py"
        mock_manager.request.return_value = [
            {"uri": "file:///a.py", "range": {"start": {"line": 9, "character": 0}}},
            {"uri": "file:///b.py", "range": {"start": {"line": 19, "character": 5}}},
            {"uri": "file:///c.py", "range": {"start": {"line": 0, "character": 0}}},
        ]
        tool._lsp_manager = mock_manager

        result = await tool.execute(file_path=str(f), line=1, symbol_name="foo")
        assert not result.is_error
        assert "3 处引用" in result.output
        assert "/a.py:10" in result.output
        assert "/b.py:20" in result.output
        assert "/c.py:1" in result.output

    async def test_no_references(
        self, tool: FindReferencesTool, tmp_path: Path
    ) -> None:
        f = tmp_path / "test.py"
        f.write_text("def foo(): pass\n")

        mock_manager = AsyncMock(spec=LSPManager)
        mock_manager.is_ready.return_value = True
        mock_manager.open_document.return_value = "file:///test.py"
        mock_manager.request.return_value = []
        tool._lsp_manager = mock_manager

        result = await tool.execute(file_path=str(f), line=1, symbol_name="foo")
        assert not result.is_error
        assert "未找到" in result.output

    async def test_truncation_at_50(
        self, tool: FindReferencesTool, tmp_path: Path
    ) -> None:
        f = tmp_path / "test.py"
        f.write_text("def foo(): pass\n")

        mock_manager = AsyncMock(spec=LSPManager)
        mock_manager.is_ready.return_value = True
        mock_manager.open_document.return_value = "file:///test.py"
        mock_manager.request.return_value = [
            {"uri": f"file:///f{i}.py", "range": {"start": {"line": i}}}
            for i in range(60)
        ]
        tool._lsp_manager = mock_manager

        result = await tool.execute(file_path=str(f), line=1, symbol_name="foo")
        assert not result.is_error
        assert "60 处引用" in result.output
        assert "前 50 条" in result.output

    async def test_timeout_handling(
        self, tool: FindReferencesTool, tmp_path: Path
    ) -> None:
        f = tmp_path / "test.py"
        f.write_text("def foo(): pass\n")

        mock_manager = AsyncMock(spec=LSPManager)
        mock_manager.is_ready.return_value = True
        mock_manager.open_document.return_value = "file:///test.py"
        mock_manager.request.side_effect = asyncio.TimeoutError()
        tool._lsp_manager = mock_manager

        result = await tool.execute(file_path=str(f), line=1, symbol_name="foo")
        assert result.is_error
        assert "超时" in result.error


# ===========================================================================
# GetHoverInfoTool 测试
# ===========================================================================


class TestGetHoverInfoTool:
    @pytest.fixture
    def tool(self) -> GetHoverInfoTool:
        return GetHoverInfoTool()

    def test_permission_level(self, tool: GetHoverInfoTool) -> None:
        assert tool.permission_level == PermissionLevel.AUTO

    def test_name(self, tool: GetHoverInfoTool) -> None:
        assert tool.name == "GetHoverInfo"

    async def test_no_manager(self, tool: GetHoverInfoTool) -> None:
        result = await tool.execute(file_path="test.py", line=1, symbol_name="foo")
        assert result.is_error

    async def test_symbol_not_on_line(
        self, tool: GetHoverInfoTool, tmp_path: Path
    ) -> None:
        f = tmp_path / "test.py"
        f.write_text("x = 1\n")
        tool._lsp_manager = LSPManager()

        result = await tool.execute(
            file_path=str(f), line=1, symbol_name="nonexistent"
        )
        assert result.is_error
        assert "未找到符号" in result.error

    async def test_hover_markup_content(
        self, tool: GetHoverInfoTool, tmp_path: Path
    ) -> None:
        f = tmp_path / "test.py"
        f.write_text("def foo() -> int: return 1\n")

        mock_manager = AsyncMock(spec=LSPManager)
        mock_manager.is_ready.return_value = True
        mock_manager.open_document.return_value = "file:///test.py"
        mock_manager.request.return_value = {
            "contents": {
                "kind": "markdown",
                "value": "```python\ndef foo() -> int\n```\nReturns an integer.",
            },
        }
        tool._lsp_manager = mock_manager

        result = await tool.execute(file_path=str(f), line=1, symbol_name="foo")
        assert not result.is_error
        assert "def foo" in result.output
        assert "int" in result.output

    async def test_hover_plain_string(
        self, tool: GetHoverInfoTool, tmp_path: Path
    ) -> None:
        f = tmp_path / "test.py"
        f.write_text("some_variable: int = 1\n")

        mock_manager = AsyncMock(spec=LSPManager)
        mock_manager.is_ready.return_value = True
        mock_manager.open_document.return_value = "file:///test.py"
        mock_manager.request.return_value = {"contents": "some_variable: int"}
        tool._lsp_manager = mock_manager

        result = await tool.execute(
            file_path=str(f), line=1, symbol_name="some_variable"
        )
        assert not result.is_error
        assert "some_variable: int" in result.output

    async def test_hover_marked_string_with_language(
        self, tool: GetHoverInfoTool, tmp_path: Path
    ) -> None:
        f = tmp_path / "test.py"
        f.write_text("def bar(x: str) -> None: pass\n")

        mock_manager = AsyncMock(spec=LSPManager)
        mock_manager.is_ready.return_value = True
        mock_manager.open_document.return_value = "file:///test.py"
        mock_manager.request.return_value = {
            "contents": {"language": "python", "value": "def bar(x: str) -> None"},
        }
        tool._lsp_manager = mock_manager

        result = await tool.execute(file_path=str(f), line=1, symbol_name="bar")
        assert not result.is_error
        assert "def bar" in result.output
        assert "```python" in result.output

    async def test_hover_list_format(
        self, tool: GetHoverInfoTool, tmp_path: Path
    ) -> None:
        f = tmp_path / "test.py"
        f.write_text("class Foo: pass\n")

        mock_manager = AsyncMock(spec=LSPManager)
        mock_manager.is_ready.return_value = True
        mock_manager.open_document.return_value = "file:///test.py"
        mock_manager.request.return_value = {
            "contents": [
                {"language": "python", "value": "class Foo"},
                "A useful class.",
            ],
        }
        tool._lsp_manager = mock_manager

        result = await tool.execute(file_path=str(f), line=1, symbol_name="Foo")
        assert not result.is_error
        assert "class Foo" in result.output
        assert "A useful class" in result.output

    async def test_hover_null_result(
        self, tool: GetHoverInfoTool, tmp_path: Path
    ) -> None:
        f = tmp_path / "test.py"
        f.write_text("foo()\n")

        mock_manager = AsyncMock(spec=LSPManager)
        mock_manager.is_ready.return_value = True
        mock_manager.open_document.return_value = "file:///test.py"
        mock_manager.request.return_value = None
        tool._lsp_manager = mock_manager

        result = await tool.execute(file_path=str(f), line=1, symbol_name="foo")
        assert not result.is_error
        assert "没有可用" in result.output

    async def test_timeout_handling(
        self, tool: GetHoverInfoTool, tmp_path: Path
    ) -> None:
        f = tmp_path / "test.py"
        f.write_text("foo()\n")

        mock_manager = AsyncMock(spec=LSPManager)
        mock_manager.is_ready.return_value = True
        mock_manager.open_document.return_value = "file:///test.py"
        mock_manager.request.side_effect = asyncio.TimeoutError()
        tool._lsp_manager = mock_manager

        result = await tool.execute(file_path=str(f), line=1, symbol_name="foo")
        assert result.is_error
        assert "超时" in result.error


# ===========================================================================
# GetDiagnosticsTool 测试
# ===========================================================================


class TestGetDiagnosticsTool:
    @pytest.fixture
    def tool(self) -> GetDiagnosticsTool:
        return GetDiagnosticsTool()

    def test_permission_level(self, tool: GetDiagnosticsTool) -> None:
        assert tool.permission_level == PermissionLevel.AUTO

    def test_name(self, tool: GetDiagnosticsTool) -> None:
        assert tool.name == "GetDiagnostics"

    async def test_no_manager(self, tool: GetDiagnosticsTool) -> None:
        result = await tool.execute(file_path="test.py")
        assert result.is_error
        assert "未初始化" in result.error

    async def test_no_diagnostics(self, tool: GetDiagnosticsTool) -> None:
        mock_manager = AsyncMock(spec=LSPManager)
        mock_manager.is_ready.return_value = True
        mock_manager.open_document.return_value = "file:///test.py"
        mock_manager._diagnostics = {}
        tool._lsp_manager = mock_manager

        result = await tool.execute(file_path="test.py")
        assert not result.is_error
        assert "没有发现" in result.output

    async def test_with_diagnostics(self, tool: GetDiagnosticsTool) -> None:
        mock_manager = AsyncMock(spec=LSPManager)
        mock_manager.is_ready.return_value = True
        mock_manager.open_document.return_value = "file:///test.py"
        mock_manager._diagnostics = {
            "file:///test.py": [
                {
                    "range": {"start": {"line": 11, "character": 0}},
                    "severity": 1,
                    "message": "Undefined variable 'foo'",
                },
                {
                    "range": {"start": {"line": 24, "character": 0}},
                    "severity": 2,
                    "message": "Unused import 'os'",
                },
            ]
        }
        tool._lsp_manager = mock_manager

        result = await tool.execute(file_path="test.py")
        assert not result.is_error
        assert "2 个诊断问题" in result.output
        assert "L12" in result.output
        assert "[error]" in result.output
        assert "Undefined variable" in result.output
        assert "L25" in result.output
        assert "[warning]" in result.output
        assert "Unused import" in result.output

    async def test_diagnostics_all_files(self, tool: GetDiagnosticsTool) -> None:
        mock_manager = AsyncMock(spec=LSPManager)
        mock_manager.is_ready.return_value = True
        mock_manager._diagnostics = {
            "file:///a.py": [
                {"range": {"start": {"line": 0}}, "severity": 1, "message": "Error in a"},
            ],
            "file:///b.py": [
                {"range": {"start": {"line": 4}}, "severity": 3, "message": "Info in b"},
            ],
        }
        tool._lsp_manager = mock_manager

        result = await tool.execute(file_path=None)
        assert not result.is_error
        assert "Error in a" in result.output
        assert "Info in b" in result.output
        assert "[info]" in result.output

    async def test_diagnostics_server_not_ready(self, tool: GetDiagnosticsTool) -> None:
        mock_manager = AsyncMock(spec=LSPManager)
        mock_manager.is_ready.return_value = False
        tool._lsp_manager = mock_manager

        result = await tool.execute(file_path=None)
        assert not result.is_error
        assert "未运行" in result.output

    async def test_severity_mapping(self, tool: GetDiagnosticsTool) -> None:
        mock_manager = AsyncMock(spec=LSPManager)
        mock_manager.is_ready.return_value = True
        mock_manager.open_document.return_value = "file:///test.py"
        mock_manager._diagnostics = {
            "file:///test.py": [
                {"range": {"start": {"line": 0}}, "severity": 1, "message": "e"},
                {"range": {"start": {"line": 1}}, "severity": 2, "message": "w"},
                {"range": {"start": {"line": 2}}, "severity": 3, "message": "i"},
                {"range": {"start": {"line": 3}}, "severity": 4, "message": "h"},
            ]
        }
        tool._lsp_manager = mock_manager

        result = await tool.execute(file_path="test.py")
        assert "[error]" in result.output
        assert "[warning]" in result.output
        assert "[info]" in result.output
        assert "[hint]" in result.output


# ===========================================================================
# 降级行为测试
# ===========================================================================


class TestDegradation:
    async def test_goto_server_not_installed(self, tmp_path: Path) -> None:
        f = tmp_path / "test.py"
        f.write_text("def hello(): pass\n")
        manager = LSPManager()
        tool = GotoDefinitionTool()
        tool._lsp_manager = manager
        with patch("shutil.which", return_value=None):
            result = await tool.execute(
                file_path=str(f), line=1, symbol_name="hello"
            )
            assert result.is_error
            assert "未安装" in result.error or "无法识别" in result.error

    async def test_find_refs_server_not_installed(self, tmp_path: Path) -> None:
        f = tmp_path / "app.ts"
        f.write_text("function foo() {}\n")
        manager = LSPManager()
        tool = FindReferencesTool()
        tool._lsp_manager = manager
        with patch("shutil.which", return_value=None):
            result = await tool.execute(
                file_path=str(f), line=1, symbol_name="foo"
            )
            assert result.is_error
            assert "未安装" in result.error

    async def test_hover_server_not_installed(self, tmp_path: Path) -> None:
        f = tmp_path / "main.go"
        f.write_text("func main() {}\n")
        manager = LSPManager()
        tool = GetHoverInfoTool()
        tool._lsp_manager = manager
        with patch("shutil.which", return_value=None):
            result = await tool.execute(
                file_path=str(f), line=1, symbol_name="main"
            )
            assert result.is_error
            assert "未安装" in result.error

    async def test_unknown_file_type(self, tmp_path: Path) -> None:
        f = tmp_path / "data.csv"
        f.write_text("a,b,c\n")
        manager = LSPManager()
        tool = GotoDefinitionTool()
        tool._lsp_manager = manager
        result = await tool.execute(file_path=str(f), line=1, symbol_name="a")
        assert result.is_error
        assert "无法识别" in result.error
