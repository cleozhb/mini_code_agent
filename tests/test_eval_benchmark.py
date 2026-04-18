"""Benchmark 任务与任务集测试."""

from __future__ import annotations

from pathlib import Path

import pytest

from mini_code_agent.eval.benchmark import (
    BenchmarkSuite,
    BenchmarkTask,
    compute_suite_hash,
    compute_task_hash,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
TASKS_DIR = REPO_ROOT / "eval" / "tasks"


# ---------------------------------------------------------------------------
# 小工具：在 tmp_path 里造一个 minimal 合法 task
# ---------------------------------------------------------------------------


def _write_minimal_task(
    task_dir: Path,
    *,
    task_id: str = "T-fake",
    level: int = 1,
    tags: list[str] | None = None,
    workspace_file_content: str = "print('hi')\n",
) -> Path:
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "workspace").mkdir(exist_ok=True)
    (task_dir / "workspace" / "main.py").write_text(workspace_file_content)
    (task_dir / "validate.py").write_text("print('{\"passed\": true}')\n")
    yaml_tags = "\n".join(f"  - {t}" for t in (tags or ["python"]))
    (task_dir / "task.yaml").write_text(
        f"""id: {task_id}
level: {level}
description: |
  fake task
expected_files:
  - main.py
max_steps: 5
max_tokens: 1000
max_wall_time_seconds: 60
tags:
{yaml_tags}
"""
    )
    return task_dir


# ---------------------------------------------------------------------------
# BenchmarkTask.load
# ---------------------------------------------------------------------------


class TestBenchmarkTaskLoad:
    def test_load_real_L1(self) -> None:
        task = BenchmarkTask.load(TASKS_DIR / "L1-add-function")
        assert task.id == "L1-add-function"
        assert task.level == 1
        assert "时间戳" in task.description or "Unix" in task.description
        assert task.expected_files == ("src/utils.py", "tests/test_utils.py")
        # 无 "|" 时 expected_file_groups 每项都是单元素组
        assert task.expected_file_groups == (
            ("src/utils.py",),
            ("tests/test_utils.py",),
        )
        assert task.max_steps == 20
        assert task.max_tokens == 50_000
        assert task.max_wall_time_seconds == 300
        assert "single-file" in task.tags and "python" in task.tags
        assert task.workspace_dir.is_dir()
        assert task.validate_script.is_file()
        assert len(task.task_hash) == 64  # sha256 hex

    def test_load_expected_files_with_alternative_groups(self, tmp_path: Path) -> None:
        """expected_files 里的 "a|b" 该被拆成 alternative 组 ("a", "b")."""
        task_dir = tmp_path / "T-alt"
        task_dir.mkdir()
        (task_dir / "workspace").mkdir()
        (task_dir / "workspace" / "main.py").write_text("x")
        (task_dir / "validate.py").write_text("print('{}')")
        (task_dir / "task.yaml").write_text(
            "id: T-alt\n"
            "level: 1\n"
            "description: d\n"
            "expected_files:\n"
            "  - config.py\n"
            "  - test_config.py|tests/test_config.py\n"
            "max_steps: 5\n"
            "max_tokens: 1000\n"
            "max_wall_time_seconds: 60\n"
        )
        task = BenchmarkTask.load(task_dir)
        # expected_files 保留原始字符串（便于 human read + hash 稳定）
        assert task.expected_files == (
            "config.py",
            "test_config.py|tests/test_config.py",
        )
        # 派生的 groups 才是评分用的
        assert task.expected_file_groups == (
            ("config.py",),
            ("test_config.py", "tests/test_config.py"),
        )

    def test_load_expected_files_trims_whitespace_in_alternatives(
        self, tmp_path: Path
    ) -> None:
        """'a | b' 的空白应被 strip 掉；不是扩充新路径语义，纯容错."""
        task_dir = tmp_path / "T-alt-ws"
        task_dir.mkdir()
        (task_dir / "workspace").mkdir()
        (task_dir / "workspace" / "main.py").write_text("x")
        (task_dir / "validate.py").write_text("print('{}')")
        (task_dir / "task.yaml").write_text(
            "id: T-alt-ws\n"
            "level: 1\n"
            "description: d\n"
            "expected_files:\n"
            "  - '  a.py  |  b.py  '\n"
            "max_steps: 5\n"
            "max_tokens: 1000\n"
            "max_wall_time_seconds: 60\n"
        )
        task = BenchmarkTask.load(task_dir)
        assert task.expected_file_groups == (("a.py", "b.py"),)

    def test_load_expected_files_rejects_empty_alternative(
        self, tmp_path: Path
    ) -> None:
        """纯空串的 expected_files 条目应报错，不让它静默变成空组."""
        task_dir = tmp_path / "T-empty"
        task_dir.mkdir()
        (task_dir / "workspace").mkdir()
        (task_dir / "workspace" / "main.py").write_text("x")
        (task_dir / "validate.py").write_text("print('{}')")
        (task_dir / "task.yaml").write_text(
            "id: T-empty\n"
            "level: 1\n"
            "description: d\n"
            "expected_files:\n"
            "  - ''\n"
            "max_steps: 5\n"
            "max_tokens: 1000\n"
            "max_wall_time_seconds: 60\n"
        )
        with pytest.raises(ValueError, match="expected_files"):
            BenchmarkTask.load(task_dir)

    def test_load_missing_yaml(self, tmp_path: Path) -> None:
        (tmp_path / "workspace").mkdir()
        (tmp_path / "validate.py").write_text("")
        with pytest.raises(FileNotFoundError):
            BenchmarkTask.load(tmp_path)

    def test_load_missing_workspace(self, tmp_path: Path) -> None:
        (tmp_path / "task.yaml").write_text(
            "id: x\nlevel: 1\ndescription: d\n"
            "max_steps: 1\nmax_tokens: 1\nmax_wall_time_seconds: 1\n"
        )
        (tmp_path / "validate.py").write_text("")
        with pytest.raises(FileNotFoundError):
            BenchmarkTask.load(tmp_path)

    def test_load_missing_validate(self, tmp_path: Path) -> None:
        (tmp_path / "workspace").mkdir()
        (tmp_path / "task.yaml").write_text(
            "id: x\nlevel: 1\ndescription: d\n"
            "max_steps: 1\nmax_tokens: 1\nmax_wall_time_seconds: 1\n"
        )
        with pytest.raises(FileNotFoundError):
            BenchmarkTask.load(tmp_path)

    def test_load_missing_required_field(self, tmp_path: Path) -> None:
        (tmp_path / "workspace").mkdir()
        (tmp_path / "validate.py").write_text("")
        # 缺 max_tokens
        (tmp_path / "task.yaml").write_text(
            "id: x\nlevel: 1\ndescription: d\n"
            "max_steps: 1\nmax_wall_time_seconds: 1\n"
        )
        with pytest.raises(ValueError, match="缺少必需字段"):
            BenchmarkTask.load(tmp_path)

    def test_load_malformed_yaml(self, tmp_path: Path) -> None:
        (tmp_path / "workspace").mkdir()
        (tmp_path / "validate.py").write_text("")
        (tmp_path / "task.yaml").write_text("::: not yaml :::\n")
        with pytest.raises(ValueError):
            BenchmarkTask.load(tmp_path)


# ---------------------------------------------------------------------------
# task_hash 稳定性
# ---------------------------------------------------------------------------


class TestTaskHash:
    def test_identical_content_same_hash(self, tmp_path: Path) -> None:
        a = _write_minimal_task(tmp_path / "a")
        b = _write_minimal_task(tmp_path / "b")
        assert compute_task_hash(a) == compute_task_hash(b)

    def test_one_byte_change_differs(self, tmp_path: Path) -> None:
        task_dir = _write_minimal_task(tmp_path / "t")
        h1 = compute_task_hash(task_dir)
        (task_dir / "workspace" / "main.py").write_text("print('changed')\n")
        h2 = compute_task_hash(task_dir)
        assert h1 != h2

    def test_rename_file_differs(self, tmp_path: Path) -> None:
        task_dir = _write_minimal_task(tmp_path / "t")
        h1 = compute_task_hash(task_dir)
        # 内容一样但文件名变了 → hash 必须变（relpath 进 hash 了）
        (task_dir / "workspace" / "main.py").rename(
            task_dir / "workspace" / "renamed.py"
        )
        h2 = compute_task_hash(task_dir)
        assert h1 != h2

    def test_ignores_pycache_and_pyc(self, tmp_path: Path) -> None:
        task_dir = _write_minimal_task(tmp_path / "t")
        h1 = compute_task_hash(task_dir)
        # 加 __pycache__/foo.pyc 和一个散落的 .pyc
        (task_dir / "workspace" / "__pycache__").mkdir()
        (task_dir / "workspace" / "__pycache__" / "foo.pyc").write_bytes(b"garbage")
        (task_dir / "workspace" / "stray.pyc").write_bytes(b"other garbage")
        (task_dir / "workspace" / ".DS_Store").write_bytes(b"\x00")
        h2 = compute_task_hash(task_dir)
        assert h1 == h2

    def test_real_L1_hash_stable(self) -> None:
        # 同一目录连算两次必须一致
        d = TASKS_DIR / "L1-add-function"
        assert compute_task_hash(d) == compute_task_hash(d)


# ---------------------------------------------------------------------------
# BenchmarkSuite & suite_hash
# ---------------------------------------------------------------------------


class TestBenchmarkSuite:
    def test_load_from_dir(self, tmp_path: Path) -> None:
        _write_minimal_task(tmp_path / "T-b", task_id="T-b", level=2, tags=["x"])
        _write_minimal_task(tmp_path / "T-a", task_id="T-a", level=1, tags=["y"])
        suite = BenchmarkSuite.load_from_dir(tmp_path)
        assert len(suite) == 2
        # 目录是 sorted 遍历：T-a 在前
        assert suite.tasks[0].id == "T-a"
        assert suite.tasks[1].id == "T-b"
        assert len(suite.suite_hash) == 64

    def test_load_from_dir_skips_non_task_subdirs(self, tmp_path: Path) -> None:
        _write_minimal_task(tmp_path / "T-a", task_id="T-a")
        (tmp_path / "not-a-task").mkdir()  # 没有 task.yaml → 跳过
        suite = BenchmarkSuite.load_from_dir(tmp_path)
        assert len(suite) == 1

    def test_load_real_tasks_dir(self) -> None:
        suite = BenchmarkSuite.load_from_dir(TASKS_DIR)
        # 目前只有一个任务
        assert len(suite) >= 1
        assert suite.get("L1-add-function") is not None
        assert suite.get("non-existent") is None

    def test_filter_by_level(self, tmp_path: Path) -> None:
        _write_minimal_task(tmp_path / "T-a", task_id="T-a", level=1)
        _write_minimal_task(tmp_path / "T-b", task_id="T-b", level=2)
        _write_minimal_task(tmp_path / "T-c", task_id="T-c", level=1)
        suite = BenchmarkSuite.load_from_dir(tmp_path)
        l1 = suite.filter_by_level(1)
        assert {t.id for t in l1.tasks} == {"T-a", "T-c"}
        assert l1.suite_hash != suite.suite_hash  # 子集 hash 不同

    def test_filter_by_tag(self, tmp_path: Path) -> None:
        _write_minimal_task(tmp_path / "T-a", task_id="T-a", tags=["python", "fast"])
        _write_minimal_task(tmp_path / "T-b", task_id="T-b", tags=["python"])
        _write_minimal_task(tmp_path / "T-c", task_id="T-c", tags=["shell"])
        suite = BenchmarkSuite.load_from_dir(tmp_path)
        fast = suite.filter_by_tag("fast")
        assert [t.id for t in fast.tasks] == ["T-a"]
        shell = suite.filter_by_tag("shell")
        assert [t.id for t in shell.tasks] == ["T-c"]

    def test_suite_hash_order_insensitive(self, tmp_path: Path) -> None:
        # 同样的 (id, task_hash) 集合无论顺序，suite_hash 都一样
        _write_minimal_task(tmp_path / "Ta", task_id="Ta")
        _write_minimal_task(tmp_path / "Tb", task_id="Tb")
        suite = BenchmarkSuite.load_from_dir(tmp_path)

        reversed_tasks = list(reversed(suite.tasks))
        assert compute_suite_hash(reversed_tasks) == suite.suite_hash

    def test_suite_hash_changes_when_task_changes(self, tmp_path: Path) -> None:
        _write_minimal_task(tmp_path / "Ta", task_id="Ta")
        suite1 = BenchmarkSuite.load_from_dir(tmp_path)
        # 改 workspace 一个字节
        (tmp_path / "Ta" / "workspace" / "main.py").write_text("# edited\n")
        suite2 = BenchmarkSuite.load_from_dir(tmp_path)
        assert suite1.suite_hash != suite2.suite_hash
