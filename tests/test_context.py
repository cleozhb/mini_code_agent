"""项目上下文感知模块测试."""

from __future__ import annotations

import json
import os
import textwrap
from pathlib import Path

import pytest

from mini_code_agent.context import (
    ContextBudget,
    ContextBuilder,
    ProjectInfo,
    build_repo_map,
    build_repo_map_paths_only,
    detect_project_type,
    estimate_tokens,
    get_directory_tree,
    get_key_files,
    summarize_file,
)
from mini_code_agent.core.system_prompt import (
    DEFAULT_SYSTEM_PROMPT,
    build_system_prompt,
    build_system_prompt_with_context,
)


# ---------------------------------------------------------------------------
# Fixtures: 测试项目目录
# ---------------------------------------------------------------------------


@pytest.fixture
def small_python_project(tmp_path: Path) -> Path:
    """创建一个小型 Python 项目 (<20 文件)."""
    root = tmp_path / "my_project"
    root.mkdir()

    # pyproject.toml
    (root / "pyproject.toml").write_text(textwrap.dedent("""\
        [project]
        name = "my-project"
        version = "0.1.0"
        description = "A test project"
        requires-python = ">=3.12"
        dependencies = ["fastapi>=0.100.0", "uvicorn"]

        [build-system]
        requires = ["hatchling"]
        build-backend = "hatchling.build"
    """))

    # uv.lock (触发 uv 检测)
    (root / "uv.lock").write_text("# lock file")

    # CLAUDE.md
    (root / "CLAUDE.md").write_text("# Project Instructions\n\nUse uv for deps.\n")

    # README.md
    (root / "README.md").write_text("# My Project\n\nA test project.\n")

    # main.py
    (root / "main.py").write_text(textwrap.dedent("""\
        from my_project.app import create_app

        def main():
            app = create_app()
            app.run()

        if __name__ == "__main__":
            main()
    """))

    # src/my_project/
    pkg = root / "src" / "my_project"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text('"""my_project package."""\n')

    # src/my_project/app.py
    (pkg / "app.py").write_text(textwrap.dedent("""\
        class App:
            def __init__(self, config):
                self.config = config

            def run(self):
                pass

            def stop(self):
                pass

        def create_app():
            return App({})
    """))

    # src/my_project/models.py
    (pkg / "models.py").write_text(textwrap.dedent("""\
        from dataclasses import dataclass

        @dataclass
        class User:
            name: str
            email: str

        @dataclass
        class Project:
            title: str
            owner: User
    """))

    # src/my_project/utils.py
    (pkg / "utils.py").write_text(textwrap.dedent("""\
        def format_name(name: str) -> str:
            return name.strip().title()

        def validate_email(email: str) -> bool:
            return "@" in email
    """))

    # tests/
    tests = root / "tests"
    tests.mkdir()
    (tests / "__init__.py").write_text("")
    (tests / "test_app.py").write_text(textwrap.dedent("""\
        def test_create_app():
            from my_project.app import create_app
            app = create_app()
            assert app is not None
    """))

    # .gitignore
    (root / ".gitignore").write_text("__pycache__/\n*.pyc\n.venv/\n")

    # .env.example
    (root / ".env.example").write_text("API_KEY=xxx\n")

    return root


@pytest.fixture
def large_project(tmp_path: Path) -> Path:
    """创建一个大型项目 (>500 文件)."""
    root = tmp_path / "large_project"
    root.mkdir()

    (root / "pyproject.toml").write_text(textwrap.dedent("""\
        [project]
        name = "large-project"
        version = "1.0.0"
        description = "A very large project"
    """))

    (root / "README.md").write_text("# Large Project\n")
    (root / "main.py").write_text("def main(): pass\n")

    # 创建 >500 个 Python 文件
    for i in range(520):
        pkg_dir = root / "src" / f"module_{i // 50}"
        pkg_dir.mkdir(parents=True, exist_ok=True)
        init_file = pkg_dir / "__init__.py"
        if not init_file.exists():
            init_file.write_text("")

        (pkg_dir / f"file_{i}.py").write_text(textwrap.dedent(f"""\
            class Handler{i}:
                def process(self):
                    pass

            def helper_{i}():
                pass
        """))

    return root


@pytest.fixture
def node_project(tmp_path: Path) -> Path:
    """创建一个 Node.js/TypeScript 项目."""
    root = tmp_path / "node_app"
    root.mkdir()

    (root / "package.json").write_text(json.dumps({
        "name": "node-app",
        "version": "2.0.0",
        "description": "A Node.js app",
        "dependencies": {"react": "^18.0.0", "next": "^14.0.0"},
        "devDependencies": {"typescript": "^5.0.0"},
    }))
    (root / "tsconfig.json").write_text("{}")
    (root / "yarn.lock").write_text("")

    src = root / "src"
    src.mkdir()
    (src / "index.ts").write_text(textwrap.dedent("""\
        export function main() {
            console.log("hello");
        }

        export class App {
            start() {}
        }
    """))

    return root


@pytest.fixture
def rust_project(tmp_path: Path) -> Path:
    """创建一个 Rust 项目."""
    root = tmp_path / "rust_app"
    root.mkdir()

    (root / "Cargo.toml").write_text(textwrap.dedent("""\
        [package]
        name = "rust-app"
        version = "0.3.0"
    """))

    src = root / "src"
    src.mkdir()
    (src / "main.rs").write_text(textwrap.dedent("""\
        pub fn greet(name: &str) -> String {
            format!("Hello, {}!", name)
        }

        pub struct Config {
            pub debug: bool,
        }

        fn main() {
            println!("Hello world");
        }
    """))

    return root


@pytest.fixture
def go_project(tmp_path: Path) -> Path:
    """创建一个 Go 项目."""
    root = tmp_path / "go_app"
    root.mkdir()

    (root / "go.mod").write_text("module github.com/user/go-app\n\ngo 1.21\n")
    (root / "main.go").write_text(textwrap.dedent("""\
        package main

        func main() {
            println("hello")
        }

        type Server struct {
            Port int
        }

        func NewServer() *Server {
            return &Server{Port: 8080}
        }
    """))

    return root


# ===========================================================================
# 1. project_analyzer 测试
# ===========================================================================


class TestDetectProjectType:
    """测试 detect_project_type."""

    def test_python_project(self, small_python_project: Path) -> None:
        info = detect_project_type(small_python_project)
        assert info.project_type == "python"
        assert info.language == "Python"
        assert info.package_manager == "uv"  # 有 uv.lock
        assert info.name == "my-project"
        assert info.version == "0.1.0"
        assert info.framework == "fastapi"
        assert "main.py" in info.entry_points

    def test_node_project(self, node_project: Path) -> None:
        info = detect_project_type(node_project)
        assert info.project_type == "node"
        assert info.language == "TypeScript"  # 有 tsconfig.json
        assert info.package_manager == "yarn"  # 有 yarn.lock
        assert info.name == "node-app"
        assert info.framework == "react"  # react in deps

    def test_rust_project(self, rust_project: Path) -> None:
        info = detect_project_type(rust_project)
        assert info.project_type == "rust"
        assert info.language == "Rust"
        assert info.package_manager == "cargo"
        assert info.name == "rust-app"
        assert "src/main.rs" in info.entry_points

    def test_go_project(self, go_project: Path) -> None:
        info = detect_project_type(go_project)
        assert info.project_type == "go"
        assert info.language == "Go"
        assert info.name == "github.com/user/go-app"
        assert "main.go" in info.entry_points

    def test_unknown_project(self, tmp_path: Path) -> None:
        info = detect_project_type(tmp_path)
        assert info.project_type == "unknown"
        assert info.language == "Unknown"

    def test_java_maven_project(self, tmp_path: Path) -> None:
        (tmp_path / "pom.xml").write_text("<project></project>")
        info = detect_project_type(tmp_path)
        assert info.project_type == "java"
        assert info.package_manager == "maven"

    def test_java_gradle_project(self, tmp_path: Path) -> None:
        (tmp_path / "build.gradle").write_text("plugins {}")
        info = detect_project_type(tmp_path)
        assert info.project_type == "java"
        assert info.package_manager == "gradle"


class TestGetDirectoryTree:
    """测试 get_directory_tree."""

    def test_basic_tree(self, small_python_project: Path) -> None:
        tree = get_directory_tree(small_python_project, max_depth=2)
        assert "my_project/" in tree
        assert "src/" in tree
        assert "pyproject.toml" in tree
        assert "main.py" in tree

    def test_ignores_hidden_dirs(self, tmp_path: Path) -> None:
        (tmp_path / ".git").mkdir()
        (tmp_path / ".git" / "config").write_text("")
        (tmp_path / "visible.txt").write_text("")

        tree = get_directory_tree(tmp_path, max_depth=2)
        assert ".git" not in tree
        assert "visible.txt" in tree

    def test_ignores_pycache(self, tmp_path: Path) -> None:
        (tmp_path / "__pycache__").mkdir()
        (tmp_path / "__pycache__" / "foo.pyc").write_text("")
        (tmp_path / "real.py").write_text("")

        tree = get_directory_tree(tmp_path, max_depth=2)
        assert "__pycache__" not in tree
        assert "real.py" in tree

    def test_max_depth_control(self, small_python_project: Path) -> None:
        tree_d1 = get_directory_tree(small_python_project, max_depth=1)
        tree_d3 = get_directory_tree(small_python_project, max_depth=3)
        # 深度 3 应包含更多内容
        assert len(tree_d3) > len(tree_d1)

    def test_not_a_directory(self, tmp_path: Path) -> None:
        f = tmp_path / "file.txt"
        f.write_text("hello")
        result = get_directory_tree(f)
        assert "不是目录" in result


class TestGetKeyFiles:
    """测试 get_key_files."""

    def test_python_project_key_files(self, small_python_project: Path) -> None:
        key_files = get_key_files(small_python_project)
        assert "CLAUDE.md" in key_files
        assert "README.md" in key_files
        assert "pyproject.toml" in key_files
        assert "main.py" in key_files
        assert ".env.example" in key_files

    def test_node_project_key_files(self, node_project: Path) -> None:
        key_files = get_key_files(node_project)
        assert "package.json" in key_files
        assert "tsconfig.json" in key_files

    def test_empty_project(self, tmp_path: Path) -> None:
        key_files = get_key_files(tmp_path)
        assert key_files == []


class TestSummarizeFile:
    """测试 summarize_file."""

    def test_python_file_with_class_and_functions(self, small_python_project: Path) -> None:
        summary = summarize_file(small_python_project / "src" / "my_project" / "app.py")
        assert "App" in summary
        assert "run()" in summary
        assert "stop()" in summary
        assert "create_app" in summary

    def test_python_file_functions_only(self, small_python_project: Path) -> None:
        summary = summarize_file(small_python_project / "src" / "my_project" / "utils.py")
        assert "format_name" in summary
        assert "validate_email" in summary

    def test_python_file_dataclasses(self, small_python_project: Path) -> None:
        summary = summarize_file(small_python_project / "src" / "my_project" / "models.py")
        assert "User" in summary
        assert "Project" in summary

    def test_typescript_file(self, node_project: Path) -> None:
        summary = summarize_file(node_project / "src" / "index.ts")
        assert "main" in summary
        assert "App" in summary

    def test_rust_file(self, rust_project: Path) -> None:
        summary = summarize_file(rust_project / "src" / "main.rs")
        assert "greet" in summary
        assert "Config" in summary

    def test_go_file(self, go_project: Path) -> None:
        summary = summarize_file(go_project / "main.go")
        assert "Server" in summary
        assert "NewServer" in summary

    def test_nonexistent_file(self, tmp_path: Path) -> None:
        result = summarize_file(tmp_path / "nonexistent.py")
        assert "nonexistent.py" in result

    def test_non_code_file(self, tmp_path: Path) -> None:
        f = tmp_path / "data.csv"
        f.write_text("a,b,c\n1,2,3\n")
        result = summarize_file(f)
        # 非代码文件只返回路径
        assert str(f) in result
        assert "—" not in result


# ===========================================================================
# 2. repo_map 测试
# ===========================================================================


class TestBuildRepoMap:
    """测试 build_repo_map."""

    def test_small_project_has_signatures(self, small_python_project: Path) -> None:
        repo_map = build_repo_map(small_python_project)
        # 应该包含文件路径和签名摘要
        assert "app.py" in repo_map
        assert "App" in repo_map
        assert "models.py" in repo_map
        assert "utils.py" in repo_map

    def test_large_project_paths_only(self, large_project: Path) -> None:
        """超过 max_files 阈值时应降级为只列路径."""
        repo_map = build_repo_map(large_project, max_files=100)
        # 文件数 >100，应该不包含签名摘要
        # 每行只是路径，不应包含 " — "
        lines_with_summary = [l for l in repo_map.splitlines() if " — " in l]
        assert len(lines_with_summary) == 0

    def test_paths_only_mode(self, small_python_project: Path) -> None:
        repo_map = build_repo_map_paths_only(small_python_project)
        lines = repo_map.splitlines()
        # 应该只有路径，没有摘要
        for line in lines:
            assert " — " not in line

    def test_ignores_hidden_and_cache_dirs(self, small_python_project: Path) -> None:
        # 创建应该被忽略的目录
        (small_python_project / ".git").mkdir()
        (small_python_project / ".git" / "config").write_text("")
        (small_python_project / "__pycache__").mkdir()
        (small_python_project / "__pycache__" / "foo.pyc").write_text("")

        repo_map = build_repo_map(small_python_project)
        # .git 目录本身不应出现（.gitignore 是文件，可以出现）
        assert ".git/" not in repo_map
        assert ".git/config" not in repo_map
        assert "__pycache__" not in repo_map
        assert "foo.pyc" not in repo_map


# ===========================================================================
# 3. context_builder 测试
# ===========================================================================


class TestEstimateTokens:
    """测试 token 估算."""

    def test_empty_string(self) -> None:
        assert estimate_tokens("") == 0

    def test_english_text(self) -> None:
        text = "Hello world, this is a test."
        tokens = estimate_tokens(text)
        # ~28 chars / 4 ≈ 7 tokens
        assert 5 <= tokens <= 10

    def test_chinese_text(self) -> None:
        text = "你好世界这是一个测试"
        tokens = estimate_tokens(text)
        # 10 chars / 1.5 ≈ 7 tokens
        assert 5 <= tokens <= 10

    def test_mixed_text(self) -> None:
        text = "Hello 你好 world 世界"
        tokens = estimate_tokens(text)
        assert tokens > 0


class TestContextBudget:
    """测试 ContextBudget."""

    def test_default_values(self) -> None:
        budget = ContextBudget()
        assert budget.model_context_limit == 200_000
        assert budget.reserved_for_output == 8192
        assert budget.available_for_context > 0
        assert budget.initial_context_budget > 0
        assert budget.task_context_budget > 0

    def test_budget_math(self) -> None:
        budget = ContextBudget(
            model_context_limit=100_000,
            reserved_for_output=10_000,
            reserved_for_conversation_ratio=0.5,
        )
        # usable = 100000 - 10000 = 90000
        # conversation = 90000 * 0.5 = 45000
        assert budget.reserved_for_conversation == 45_000
        # available = 100000 - 10000 - 45000 = 45000
        assert budget.available_for_context == 45_000
        # initial = 45000 * 0.4 = 18000
        assert budget.initial_context_budget == 18_000
        # task = 45000 * 0.6 = 27000
        assert budget.task_context_budget == 27_000

    def test_budget_components_sum_correctly(self) -> None:
        budget = ContextBudget()
        total = (
            budget.reserved_for_output
            + budget.reserved_for_conversation
            + budget.available_for_context
        )
        assert total == budget.model_context_limit


class TestContextBuilder:
    """测试 ContextBuilder."""

    def test_analyze_project(self, small_python_project: Path) -> None:
        builder = ContextBuilder(project_path=small_python_project)
        info = builder.analyze_project()
        assert info.project_type == "python"
        assert info.name == "my-project"

    def test_build_initial_context_small_project(self, small_python_project: Path) -> None:
        """小项目 (<20 文件) 的 initial context 应包含完整 repo map."""
        builder = ContextBuilder(project_path=small_python_project)
        context = builder.build_initial_context("You are a coding assistant.")

        # 应包含基础指令
        assert "You are a coding assistant." in context
        # 应包含项目指令
        assert "Project Instructions" in context
        # 应包含项目元信息
        assert "my-project" in context
        assert "Python" in context
        # 应包含目录树
        assert "directory-tree" in context
        # 应包含 repo map
        assert "repo-map" in context
        # 小项目的 repo map 应包含签名
        assert "App" in context

    def test_build_initial_context_large_project(self, large_project: Path) -> None:
        """大项目 (>500 文件) 的 repo map 应被降级或截断."""
        # 设一个合理但不太大的 budget
        budget = ContextBudget(model_context_limit=50_000)
        builder = ContextBuilder(
            project_path=large_project,
            budget=budget,
        )
        context = builder.build_initial_context("You are a coding assistant.")

        # 应包含基础指令
        assert "You are a coding assistant." in context
        # 总 token 不应超过 budget
        tokens = estimate_tokens(context)
        assert tokens <= budget.initial_context_budget

    def test_build_initial_context_very_small_budget(self, small_python_project: Path) -> None:
        """很小的 budget 应触发降级逻辑，但不报错."""
        budget = ContextBudget(
            model_context_limit=2000,
            reserved_for_output=200,
            reserved_for_conversation_ratio=0.5,
        )
        builder = ContextBuilder(
            project_path=small_python_project,
            budget=budget,
        )
        context = builder.build_initial_context("You are a coding assistant.")

        # 应该至少包含基础指令
        assert "You are a coding assistant." in context
        # 不应报错

    def test_prefix_stability(self, small_python_project: Path) -> None:
        """验证输出的前缀部分是稳定不变的内容."""
        builder = ContextBuilder(project_path=small_python_project)
        context1 = builder.build_initial_context("Base instructions.")
        context2 = builder.build_initial_context("Base instructions.")

        # 完全一致
        assert context1 == context2

        # 前缀部分（基础指令 + 项目指令）应该是 KV cache 友好的
        stats = builder.get_context_stats()
        assert stats.cache_friendly_prefix_tokens > 0

    def test_xml_tag_structure(self, small_python_project: Path) -> None:
        """验证输出使用 XML 标签清晰分隔各 section."""
        builder = ContextBuilder(project_path=small_python_project)
        context = builder.build_initial_context("Base.")

        # 应使用标签分隔
        assert "<project-instructions>" in context
        assert "</project-instructions>" in context
        assert "<project-meta>" in context
        assert "</project-meta>" in context
        assert "<directory-tree>" in context
        assert "</directory-tree>" in context

    def test_context_stats(self, small_python_project: Path) -> None:
        builder = ContextBuilder(project_path=small_python_project)
        builder.build_initial_context("Base.")

        stats = builder.get_context_stats()
        assert stats.total_budget > 0
        assert stats.initial_context_tokens > 0
        assert stats.remaining_tokens > 0
        assert stats.remaining_tokens < stats.total_budget


class TestTaskContext:
    """测试 build_task_context."""

    def test_basic_task_context(self, small_python_project: Path) -> None:
        builder = ContextBuilder(project_path=small_python_project)
        context = builder.build_task_context([
            "src/my_project/app.py",
            "src/my_project/models.py",
        ])

        # 应包含文件内容
        assert "app.py" in context
        assert "class App" in context
        assert "models.py" in context
        assert "class User" in context

        # 应有行号标注
        assert "===" in context

    def test_task_context_with_oversized_files(self, tmp_path: Path) -> None:
        """传入超大文件应触发截断."""
        root = tmp_path / "project"
        root.mkdir()

        # 创建一个超大文件
        big_file = root / "big.py"
        lines = [f"def func_{i}(): pass  # line {i}" for i in range(2000)]
        big_file.write_text("\n".join(lines))

        # 创建一个小文件
        small_file = root / "small.py"
        small_file.write_text("def hello(): pass\n")

        # 用小 budget 测试
        budget = ContextBudget(
            model_context_limit=5000,
            reserved_for_output=500,
            reserved_for_conversation_ratio=0.5,
        )
        builder = ContextBuilder(project_path=root, budget=budget)

        context = builder.build_task_context(
            ["big.py", "small.py"],
            budget=budget,
        )

        # 大文件应被截断
        total_tokens = estimate_tokens(context)
        assert total_tokens <= budget.task_context_budget

    def test_task_context_priority_order(self, small_python_project: Path) -> None:
        """文件按传入顺序（优先级）处理."""
        budget = ContextBudget(
            model_context_limit=3000,
            reserved_for_output=300,
            reserved_for_conversation_ratio=0.5,
        )
        builder = ContextBuilder(
            project_path=small_python_project,
            budget=budget,
        )

        # app.py 优先于 models.py
        context = builder.build_task_context(
            ["src/my_project/app.py", "src/my_project/models.py"],
            budget=budget,
        )

        # 至少第一个文件应该在里面
        assert "app.py" in context

    def test_task_context_nonexistent_file(self, small_python_project: Path) -> None:
        """不存在的文件应被跳过."""
        builder = ContextBuilder(project_path=small_python_project)
        context = builder.build_task_context([
            "nonexistent.py",
            "src/my_project/app.py",
        ])

        assert "nonexistent" not in context
        assert "app.py" in context

    def test_task_context_omitted_files_hint(self, tmp_path: Path) -> None:
        """超出预算的文件应显示省略提示."""
        root = tmp_path / "project"
        root.mkdir()

        # 创建多个文件
        for i in range(10):
            f = root / f"module_{i}.py"
            f.write_text(f"class Module{i}:\n    pass\n" * 50)

        # 极小 budget
        budget = ContextBudget(
            model_context_limit=2000,
            reserved_for_output=200,
            reserved_for_conversation_ratio=0.5,
        )
        builder = ContextBuilder(project_path=root, budget=budget)

        files = [f"module_{i}.py" for i in range(10)]
        context = builder.build_task_context(files, budget=budget)

        # 应包含省略提示
        assert "已省略" in context or "ReadFile" in context


# ===========================================================================
# 4. system_prompt 集成测试
# ===========================================================================


class TestBuildSystemPrompt:
    """测试 build_system_prompt 集成."""

    def test_backward_compatible_dict_mode(self) -> None:
        """旧的 dict 方式仍然工作."""
        prompt = build_system_prompt({
            "cwd": "/tmp/test",
            "project_name": "test",
        })
        assert "工作目录" in prompt
        assert "/tmp/test" in prompt

    def test_new_project_path_mode(self, small_python_project: Path) -> None:
        """新的 project_path 方式."""
        prompt = build_system_prompt(project_path=small_python_project)
        assert DEFAULT_SYSTEM_PROMPT in prompt
        assert "my-project" in prompt
        assert "Python" in prompt

    def test_build_system_prompt_with_context(self, small_python_project: Path) -> None:
        prompt, builder = build_system_prompt_with_context(small_python_project)
        assert "coding assistant" in prompt or "编程助手" in prompt
        assert builder.project_info.project_type == "python"

        stats = builder.get_context_stats()
        assert stats.initial_context_tokens > 0

    def test_prompt_ordering(self, small_python_project: Path) -> None:
        """验证拼接顺序：基础指令 → 项目指令 → 元信息 → 目录树 → repo map."""
        prompt = build_system_prompt(project_path=small_python_project)

        # 找到各部分的位置
        base_pos = prompt.find("编程助手")
        instructions_pos = prompt.find("<project-instructions>")
        meta_pos = prompt.find("<project-meta>")
        tree_pos = prompt.find("<directory-tree>")
        map_pos = prompt.find("<repo-map>")

        # 验证顺序
        assert base_pos < instructions_pos, "基础指令应在项目指令之前"
        assert instructions_pos < meta_pos, "项目指令应在元信息之前"
        assert meta_pos < tree_pos, "元信息应在目录树之前"
        assert tree_pos < map_pos, "目录树应在 repo map 之前"
